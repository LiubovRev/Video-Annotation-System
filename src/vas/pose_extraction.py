"""Step 2 — read psifx pose archives into one frame-level table.

psifx writes one ``<track_id>.tar.gz`` per SAM3 track into ``PosesDir``.
Each archive holds one JSON per frame (``<frame>.json``) plus
``edges.json``. Archives are streamed directly — nothing is extracted to
disk.

Output: ``outputs/<session>/processed_data.parquet`` with one row per
(person, frame): ``session, person_label, track_id, frame, time_s,
avg_pose_conf, <joint>_x, <joint>_y ...``
"""

from __future__ import annotations

import json
import logging
import re
import tarfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .config import project_config, resolve_paths
from .keypoints import add_virtual_joints, is_legacy_table, joint_names, remap_legacy_columns
from .utils import probe_fps, read_table, write_table

logger = logging.getLogger(__name__)

_FRAME_RE = re.compile(r"(\d+)\s*$")
DEFAULT_FPS = 15.0


def track_id_from_archive(path: Path) -> str:
    """``3.tar.gz`` → ``"3"``."""
    return path.name.split(".")[0]


def read_pose_archive(path: Path, n_joints: int) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(frames[int], keypoints[n, n_joints, 3])`` for one archive.

    Frames where the pose is missing (all zeros) are dropped.
    """
    frames, kps = [], []
    with tarfile.open(path, "r:gz") as tar:
        for member in tar:
            if not member.isfile() or not member.name.endswith(".json"):
                continue
            stem = Path(member.name).stem
            m = _FRAME_RE.search(stem)
            if not m:  # edges.json and friends
                continue
            data = json.load(tar.extractfile(member))
            arr = np.asarray(data.get("pose_keypoints_2d") or [], dtype=float)
            if arr.size == 0 or not np.any(arr):
                continue
            arr = arr.reshape(-1, 3)
            if arr.shape[0] < n_joints:
                arr = np.vstack([arr, np.zeros((n_joints - arr.shape[0], 3))])
            frames.append(int(m.group(1)))
            kps.append(arr[:n_joints])
    if not frames:
        return np.empty(0, dtype=int), np.empty((0, n_joints, 3))
    order = np.argsort(frames)
    return np.asarray(frames)[order], np.stack(kps)[order]


def keypoints_to_frame(frames: np.ndarray, kps: np.ndarray, joints: list[str], conf_threshold: float) -> pd.DataFrame:
    vis = kps[:, :, 2]
    xy = kps[:, :, :2].copy()
    if conf_threshold > 0:
        xy[vis < conf_threshold] = np.nan
    cols = {"frame": frames, "avg_pose_conf": vis.mean(axis=1)}
    for j, name in enumerate(joints):
        cols[f"{name}_x"] = xy[:, j, 0]
        cols[f"{name}_y"] = xy[:, j, 1]
    return pd.DataFrame(cols)


def _label_map(id_to_label: dict) -> dict[str, str]:
    """Accept keys like 1, "1" or "id_1"."""
    out = {}
    for k, v in (id_to_label or {}).items():
        out[str(k).removeprefix("id_")] = v
    return out


def resolve_fps(cfg: dict, project_dir: Path) -> float:
    fps = cfg["pose_extraction"].get("fps")
    if fps:
        return float(fps)
    probed = probe_fps(Path(project_dir) / "processed_video.mp4")
    if probed:
        logger.info("Probed fps=%.3f from processed_video.mp4", probed)
        return probed
    logger.warning("fps not set and could not be probed — using %s", DEFAULT_FPS)
    return DEFAULT_FPS


def extract_poses(poses_dir: Path, cfg: dict, fps: float, session: str) -> pd.DataFrame:
    ec = cfg["pose_extraction"]
    joints = joint_names(ec.get("keypoint_format", "mediapipe"))
    labels = _label_map(ec.get("id_to_label"))
    threshold = float(ec.get("confidence_threshold", 0.5))

    parts = []
    for archive in sorted(Path(poses_dir).glob("*.tar.gz")):
        tid = track_id_from_archive(archive)
        frames, kps = read_pose_archive(archive, len(joints))
        if len(frames) == 0:
            logger.warning("No poses in %s", archive.name)
            continue
        df = keypoints_to_frame(frames, kps, joints, threshold)
        df.insert(0, "track_id", tid)
        label = labels.get(tid)
        if label is None:
            label = f"Track{tid}"
            logger.warning("Track %s has no entry in pose_extraction.id_to_label — labelled '%s'", tid, label)
        df.insert(0, "person_label", label)
        parts.append(df)
        logger.info("  %s → %s: %d frames", archive.name, label, len(df))

    if not parts:
        return pd.DataFrame()
    df = pd.concat(parts, ignore_index=True)
    df = add_virtual_joints(df)
    df.insert(0, "session", session)
    df.insert(df.columns.get_loc("frame") + 1, "time_s", df["frame"] / fps)
    return df.sort_values(["time_s", "person_label"]).reset_index(drop=True)


def find_legacy_table(project_dir: Path, poses_dir: Path) -> Path | None:
    candidates = [poses_dir / "processed_data.csv", *sorted(poses_dir.glob("processed_data__*.parquet")),
                  *sorted(project_dir.glob("processed_data__*.parquet")), project_dir / "processed_data.csv"]
    return next((c for c in candidates if c.exists()), None)


def load_legacy_table(path: Path, cfg: dict, session: str) -> pd.DataFrame:
    df = read_table(path)
    logger.info("Using existing pose table %s (%d rows)", path.name, len(df))
    if is_legacy_table(df) and cfg["pose_extraction"].get("keypoint_format", "mediapipe") == "mediapipe":
        df = remap_legacy_columns(df)
    df = df.drop(columns=["time_min:s.ms", "wrist_dist"], errors="ignore")
    df["session"] = session
    return add_virtual_joints(df)


def plot_mass_centers(df: pd.DataFrame, save_path: Path, dpi: int = 150) -> None:
    xcols = [c for c in df.columns if c.endswith("_x")]
    ycols = [c for c in df.columns if c.endswith("_y")]
    fig, axes = plt.subplots(2, 1, figsize=(14, 7), sharex=True)
    for label, g in df.groupby("person_label"):
        axes[0].plot(g["time_s"], g[xcols].mean(axis=1), label=label, lw=0.8)
        axes[1].plot(g["time_s"], g[ycols].mean(axis=1), label=label, lw=0.8)
    for ax, t in zip(axes, ("x", "y")):
        ax.set_ylabel(f"mean keypoint {t} (px)")
        ax.grid(alpha=0.3)
        ax.legend(loc="upper right")
    axes[1].set_xlabel("time in processed video (s)")
    fig.suptitle("Centre of keypoints per person")
    fig.tight_layout()
    fig.savefig(save_path, dpi=dpi)
    plt.close(fig)


def run_pose_extraction(project_dir: Path, cfg: dict, force: bool = False) -> Path | None:
    """Build ``outputs/<session>/processed_data.parquet``; return its path."""
    project_dir = Path(project_dir)
    cfg = project_config(cfg, project_dir)
    session = project_dir.name
    out_dir = resolve_paths(cfg).outputs / session
    out_path = out_dir / "processed_data.parquet"
    if out_path.exists() and not force:
        logger.info("Pose table exists — %s", out_path)
        return out_path

    poses_dir = project_dir / cfg["pose_extraction"].get("directory", "PosesDir")
    if poses_dir.exists() and any(poses_dir.glob("*.tar.gz")):
        df = extract_poses(poses_dir, cfg, resolve_fps(cfg, project_dir), session)
    elif (legacy := find_legacy_table(project_dir, poses_dir)) is not None:
        df = load_legacy_table(legacy, cfg, session)
    else:
        logger.error("No pose archives (*.tar.gz) or pose table found in %s", poses_dir)
        return None

    if df.empty:
        logger.error("Pose extraction produced no rows for %s", session)
        return None

    write_table(df, out_path)
    logger.info("Saved %s (%d rows, persons: %s)", out_path.name, len(df), sorted(df["person_label"].unique()))
    try:
        plot_mass_centers(df, out_dir / "mass_centers.png", cfg.get("plotting", {}).get("dpi", 150))
    except Exception as e:  # plotting must never break the pipeline
        logger.warning("Mass-centre plot failed: %s", e)
    return out_path
