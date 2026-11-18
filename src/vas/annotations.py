"""Step 4 — align WAKEE annotation intervals with pose frames.

A WAKEE export line (tab separated)::

    tier  <blank>  hh:mm:ss.ms  start_s  hh:mm:ss.ms  end_s  hh:mm:ss.ms  dur_s  CODE

Annotation times refer to the *original* video, pose times to the trimmed
``processed_video.mp4``; pose times are shifted by ``start_trim_sec``.
Each code is attached to a role through ``label_prefix_map`` (longest
prefix wins: ``CST`` → ``C`` → Child) and only matched to frames of
persons with that role (``roles`` in the config).
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .config import project_config, resolve_paths
from .utils import read_table, write_table

logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# Parsing
# -----------------------------------------------------------------------------
def parse_annotations(path: Path) -> pd.DataFrame:
    """Parse a WAKEE .txt export → ``tier, start_time, end_time, label``."""
    rows, bad = [], 0
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            parts = [p for p in re.split(r"\s+", line.strip()) if p]
            if len(parts) < 8:
                if parts:
                    bad += 1
                continue
            try:
                rows.append((parts[0], float(parts[2]), float(parts[4]), parts[7]))
            except ValueError:
                bad += 1
    if bad:
        logger.warning("Skipped %d unparseable line(s) in %s", bad, Path(path).name)
    df = pd.DataFrame(rows, columns=["tier", "start_time", "end_time", "label"])
    logger.info("Parsed %d annotations from %s", len(df), Path(path).name)
    return df


def role_for_label(label: str, prefix_map: dict) -> str | None:
    for prefix in sorted(prefix_map, key=len, reverse=True):
        if label.upper().startswith(str(prefix).upper()):
            return prefix_map[prefix]
    return None


def prepare_annotations(ann: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    ac = cfg.get("annotation_alignment", {})
    mt = cfg.get("model_training", {})
    prefix_map = ac.get("label_prefix_map") or {"C": "Child", "T": "Therapist"}
    label_to_class = mt.get("label_to_class") or {}
    non_posture = set(mt.get("non_posture_classes") or [])
    exclude = set(ac.get("exclude_labels") or [])

    ann = ann.copy()
    ann["role"] = ann["label"].map(lambda s: role_for_label(s, prefix_map))
    ann["label_class"] = ann["label"].map(label_to_class)
    keep = ann["role"].notna() & ~ann["label"].isin(exclude) & ~ann["label_class"].isin(non_posture)
    dropped = ann.loc[~keep, "label"].value_counts()
    if len(dropped):
        logger.info("Ignoring %d annotation(s): %s", int(dropped.sum()), {k: int(v) for k, v in dropped.items()})
    ann = ann[keep & (ann["end_time"] > ann["start_time"])].reset_index(drop=True)
    ann["duration"] = ann["end_time"] - ann["start_time"]
    return ann


# -----------------------------------------------------------------------------
# Alignment
# -----------------------------------------------------------------------------
def map_annotations_to_frames(frames: pd.DataFrame, ann: pd.DataFrame, strategy: str = "shortest") -> pd.DataFrame:
    """Vectorised interval matching. Adds ``annotation_label`` and
    ``n_overlapping_labels`` (how many annotations covered the frame)."""
    out = frames.copy()
    labels = np.full(len(out), None, dtype=object)
    n_overlap = np.zeros(len(out), dtype=int)
    t = out["video_time_s"].to_numpy()
    roles = out["role"].to_numpy()

    for role, group in ann.groupby("role"):
        idx = np.flatnonzero(roles == role)
        if idx.size == 0:
            continue
        tr = t[idx]
        best = np.full(idx.size, np.inf)
        rows = group.sort_values("duration") if strategy == "shortest" else group.iloc[::-1]
        for start, end, label, dur in rows[["start_time", "end_time", "label", "duration"]].itertuples(index=False):
            hit = (tr >= start) & (tr < end)
            if not hit.any():
                continue
            n_overlap[idx[hit]] += 1
            if strategy == "shortest":
                take = hit & (dur < best)
                best[take] = dur
            else:  # "first" in file order: iterate reversed so earlier rows overwrite
                take = hit
            labels[idx[take]] = label
    out["annotation_label"] = labels
    out["n_overlapping_labels"] = n_overlap
    return out


def find_annotation_file(project_dir: Path, cfg: dict) -> Path | None:
    project_dir = Path(project_dir)
    ann_dir = resolve_paths(cfg).annotations
    override = cfg.get("annotation_alignment", {}).get("annotation_file")
    if override:
        for base in (project_dir, ann_dir, Path(".")):
            p = Path(override) if Path(override).is_absolute() else base / override
            if p.exists():
                return p
        logger.error("annotation_file '%s' not found", override)
        return None

    prefix = re.split(r"[\[(]", project_dir.name)[0]
    candidates = []
    for base in (project_dir, ann_dir):
        if base.exists():
            candidates += sorted(p for p in base.glob("*.txt") if p.name.startswith(prefix))
    if not candidates:
        return None
    if len(candidates) > 1:
        logger.warning("Several annotation files match '%s*': %s — using %s",
                       prefix, [c.name for c in candidates], candidates[0].name)
    return candidates[0]


def align_session(frames: pd.DataFrame, ann_path: Path, cfg: dict) -> pd.DataFrame:
    vp = cfg.get("video_processing", {})
    start_trim = float(vp.get("start_trim_sec") or 0)
    end_trim = vp.get("end_trim_sec")
    roles = cfg.get("roles") or {}

    df = frames.copy()
    df["role"] = df["person_label"].map(lambda p: roles.get(p, p))
    df["video_time_s"] = df["time_s"] + start_trim
    if end_trim is not None:
        df = df[df["video_time_s"] <= float(end_trim)]

    known = set(cfg.get("annotation_alignment", {}).get("label_prefix_map", {}).values())
    if not set(df["role"]).intersection(known):
        logger.error("No person maps to an annotated role %s. Check `roles` and `pose_extraction.id_to_label` "
                     "(persons found: %s).", sorted(known), sorted(df["person_label"].unique()))

    ann = prepare_annotations(parse_annotations(ann_path), cfg)
    strategy = cfg.get("annotation_alignment", {}).get("overlap_strategy", "shortest")
    df = map_annotations_to_frames(df, ann, strategy)
    labeled = df[df["annotation_label"].notna()].reset_index(drop=True)
    logger.info("Labeled %d / %d frames (%d classes); %.1f%% had overlapping annotations",
                len(labeled), len(df), labeled["annotation_label"].nunique(),
                100 * (labeled["n_overlapping_labels"] > 1).mean() if len(labeled) else 0)
    return labeled


def plot_label_distribution(labeled: pd.DataFrame, title: str, save_path: Path, dpi: int = 150) -> None:
    counts = labeled.groupby(["annotation_label", "role"]).size().unstack(fill_value=0)
    counts = counts.loc[counts.sum(axis=1).sort_values(ascending=False).index]
    ax = counts.plot(kind="bar", stacked=True, figsize=(12, 5), edgecolor="black", width=0.8)
    ax.set_title(f"Labeled frames per code — {title}")
    ax.set_xlabel("annotation code")
    ax.set_ylabel("frames")
    ax.grid(alpha=0.3, axis="y")
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(save_path, dpi=dpi)
    plt.close()


def run_annotation_alignment(project_dir: Path, cfg: dict, force: bool = False) -> Path | None:
    """Create ``outputs/<session>/labeled_features.parquet``; return its path."""
    project_dir = Path(project_dir)
    cfg = project_config(cfg, project_dir)
    out_dir = resolve_paths(cfg).outputs / project_dir.name
    out_path = out_dir / "labeled_features.parquet"
    if out_path.exists() and not force:
        logger.info("Labeled table exists — %s", out_path)
        return out_path

    frames_path = out_dir / "processed_data.parquet"
    if not frames_path.exists():
        logger.error("Pose table missing: %s (run pose extraction first)", frames_path)
        return None
    ann_path = find_annotation_file(project_dir, cfg)
    if ann_path is None:
        logger.warning("No annotation file for %s — skipping alignment.", project_dir.name)
        return None
    logger.info("Annotation file: %s", ann_path.name)

    labeled = align_session(read_table(frames_path), ann_path, cfg)
    if labeled.empty:
        logger.error("No frames could be labeled for %s", project_dir.name)
        return None
    write_table(labeled, out_path)
    try:
        plot_label_distribution(labeled, project_dir.name, out_dir / "annotation_stats.png",
                                cfg.get("plotting", {}).get("dpi", 150))
    except Exception as e:
        logger.warning("Annotation plot failed: %s", e)
    return out_path
