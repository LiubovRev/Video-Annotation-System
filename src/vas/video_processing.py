"""Step 1 — video preprocessing, SAM3 person tracking and MediaPipe pose.

Per session folder::

    <session>/camera_a.mkv              raw video (name configurable)
    <session>/processed_video.mp4       trimmed + cropped re-encode
    <session>/MaskDir/<id>.mp4          SAM3 masks, one per track
    <session>/PosesDir/<id>.tar.gz      MediaPipe poses, one per track
    <session>/Visualizations/*.mp4      optional QA renders

External tools: ``ffmpeg`` and ``psifx`` (https://github.com/psifx/psifx).
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
from pathlib import Path

from .config import project_config

logger = logging.getLogger(__name__)


class StepError(RuntimeError):
    """An external command failed."""


def run_command(cmd: list[str], name: str, env: dict | None = None) -> None:
    logger.info(">>> %s", name)
    logger.debug("$ %s", " ".join(map(str, cmd)))
    try:
        r = subprocess.run(list(map(str, cmd)), text=True, capture_output=True, env=env)
    except FileNotFoundError as e:
        raise StepError(f"{name}: executable not found ({e.filename}). Is it installed and on PATH?") from e
    if r.stdout:
        logger.debug(r.stdout.strip())
    if r.returncode != 0:
        logger.error(r.stderr.strip()[-4000:])
        raise StepError(f"{name} failed with exit code {r.returncode}")


def _reset_dir(p: Path) -> None:
    if p.exists():
        shutil.rmtree(p)
    p.mkdir(parents=True, exist_ok=True)


def build_env(max_objects: int | None = None) -> dict:
    env = os.environ.copy()
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    if max_objects:
        env["PSIFX_MAX_OBJECTS"] = str(max_objects)
    if "HF_TOKEN" in env and "HUGGINGFACE_HUB_TOKEN" not in env:
        env["HUGGINGFACE_HUB_TOKEN"] = env["HF_TOKEN"]
    return env


def build_ffmpeg_cmd(raw_video: Path, processed_video: Path, p: dict) -> list[str]:
    """ffmpeg trim + crop (+ optional resize) + clean H.264 re-encode."""
    x_min, y_min = int(p.get("x_min") or 0), int(p.get("y_min") or 0)
    x_max, y_max = p.get("x_max"), p.get("y_max")
    vf = []
    if x_max is not None and y_max is not None:
        w, h = int(x_max) - x_min, int(y_max) - y_min
        if w <= 0 or h <= 0:
            raise ValueError(f"Invalid crop box: x {x_min}-{x_max}, y {y_min}-{y_max}")
        vf.append(f"crop={w}:{h}:{x_min}:{y_min}")
    if p.get("resize_w") or p.get("resize_h"):
        vf.append(f"scale={p.get('resize_w') or -2}:{p.get('resize_h') or -2}")

    start, end = float(p.get("start_trim_sec") or 0), p.get("end_trim_sec")
    cmd = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-ss", str(start)]
    if end is not None:
        if float(end) <= start:
            raise ValueError(f"end_trim_sec ({end}) must be greater than start_trim_sec ({start})")
        cmd += ["-to", str(end)]
    cmd += ["-i", str(raw_video), "-map", "0:v:0"]
    if vf:
        cmd += ["-vf", ",".join(vf)]
    cmd += ["-c:v", "libx264", "-preset", "slow", "-crf", "18", "-pix_fmt", "yuv420p", str(processed_video)]
    return cmd


def build_psifx_cmds(
    processed_video: Path, mask_dir: Path, poses_dir: Path, vis_dir: Path, p: dict
) -> list[tuple[str, list[str]]]:
    device = p.get("device", "cuda")
    steps = [
        ("SAM3 tracking", [
            "psifx", "video", "tracking", "sam3", "inference",
            "--video", processed_video, "--mask_dir", mask_dir,
            "--text_prompt", p.get("text_prompt", "person"),
            "--chunk_size", p.get("chunk_size", 400),
            "--iou_threshold", p.get("iou_threshold", 0.15),
            "--device", device, *(p.get("sam3_extra_args") or []),
        ]),
    ]
    if p.get("visualizations", True):
        steps.append(("Tracking visualization", [
            "psifx", "video", "tracking", "visualization",
            "--video", processed_video, "--masks", mask_dir,
            "--visualization", vis_dir / "tracking.mp4", "--labels", "--color",
        ]))
    steps.append(("Pose inference", [
        "psifx", "video", "pose", "mediapipe", "multi-inference",
        "--video", processed_video, "--masks", mask_dir, "--poses_dir", poses_dir,
        "--mask_threshold", p.get("pose_mask_threshold", 0.0),
        "--model_complexity", p.get("pose_model_complexity", 2),
        "--smooth", "--device", device, *(p.get("pose_extra_args") or []),
    ]))
    if p.get("visualizations", True):
        steps.append(("Pose visualization", [
            "psifx", "video", "pose", "mediapipe", "visualization",
            "--video", processed_video, "--poses", poses_dir,
            "--visualization", vis_dir / "pose.mp4", "--confidence_threshold", "0.0",
        ]))
    return [(name, [str(c) for c in cmd]) for name, cmd in steps]


def outputs_exist(project_dir: Path, cfg: dict) -> bool:
    poses = Path(project_dir) / cfg["pose_extraction"].get("directory", "PosesDir")
    return poses.exists() and any(poses.glob("*.tar.gz"))


def run_video_processing(project_dir: Path, cfg: dict, force: bool = False) -> bool:
    """Run ffmpeg + psifx for one session. Returns True on success."""
    project_dir = Path(project_dir)
    cfg = project_config(cfg, project_dir)
    p = cfg["video_processing"]

    if outputs_exist(project_dir, cfg) and not force:
        logger.info("Pose archives already present — skipping video processing (use force to redo).")
        return True

    raw_video = project_dir / p.get("raw_video_filename", "camera_a.mkv")
    if not raw_video.exists():
        logger.error("Raw video not found: %s", raw_video)
        return False

    processed = project_dir / "processed_video.mp4"
    mask_dir = project_dir / "MaskDir"
    poses_dir = project_dir / cfg["pose_extraction"].get("directory", "PosesDir")
    vis_dir = project_dir / "Visualizations"
    for d in (mask_dir, poses_dir, vis_dir):
        _reset_dir(d)

    env = build_env(p.get("max_objects"))
    try:
        run_command(build_ffmpeg_cmd(raw_video, processed, p), "FFmpeg trim + crop", env)
        for name, cmd in build_psifx_cmds(processed, mask_dir, poses_dir, vis_dir, p):
            run_command(cmd, name, env)
    except (StepError, ValueError) as e:
        logger.error("Video processing failed for %s: %s", project_dir.name, e)
        return False

    logger.info("Video processing done: %s", project_dir.name)
    return True
