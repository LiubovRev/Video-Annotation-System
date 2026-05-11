#!/usr/bin/env python3
# coding: utf-8
"""
processing.py
-------------
Step 1 of the Video Annotation Pipeline: video preprocessing,
person tracking, and pose inference for a single project.

Stages per project:
  1. FFmpeg trim + crop + clean re-encode  -> processed_video.mp4
  2. SAM-based person tracking             -> MaskDir/
  3. Tracking visualization                -> Visualizations/tracking.mp4
  4. Multi-person MediaPipe pose inference -> PosesDir/
  5. Pose visualization                    -> Visualizations/pose.mp4

Can run standalone (single project via CLI) or be imported by
`full_pipeline.py` via `run_video_processing(project_dir, cfg)`.

External dependencies (must be installed and on PATH):
  - ffmpeg
  - psifx  (https://github.com/idiap/psifx)
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Optional

import yaml


# =============================================================================
# Helpers
# =============================================================================

def _run(cmd, name, env):
    """Run a subprocess, raising RuntimeError if it fails."""
    print(f"\n>>> {name}")
    print(" ".join(map(str, cmd)))

    r = subprocess.run(cmd, text=True, capture_output=True, env=env)

    if r.returncode != 0:
        print(r.stderr)
        raise RuntimeError(f"Step failed: {name}")

    if r.stdout:
        print(r.stdout)


def _wipe(p: Path):
    """Reset a directory (delete if exists, then create)."""
    if p.exists():
        shutil.rmtree(p)
    p.mkdir(parents=True, exist_ok=True)


def _build_env(max_objects: Optional[int] = None):
    """Build subprocess environment with CUDA + HF token forwarding."""
    env = os.environ.copy()
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    if max_objects:
        env["PSIFX_MAX_OBJECTS"] = str(max_objects)

    # Forward HF token to the variable psifx expects
    if "HF_TOKEN" in env:
        env["HUGGINGFACE_HUB_TOKEN"] = env["HF_TOKEN"]

    return env


def _build_ffmpeg_cmd(raw_video: Path, processed_video: Path, params: dict):
    """Construct ffmpeg command: trim + crop + clean re-encode."""
    x_min = params.get("x_min", 0)
    y_min = params.get("y_min", 0)
    x_max = params.get("x_max")
    y_max = params.get("y_max")
    start_sec = params.get("start_trim_sec", 0)
    end_sec = params.get("end_trim_sec")
    resize_w = params.get("resize_w")
    resize_h = params.get("resize_h")

    vf = []
    if x_max is not None and y_max is not None:
        vf.append(f"crop={x_max - x_min}:{y_max - y_min}:{x_min}:{y_min}")
    if resize_w or resize_h:
        vf.append(f"scale={resize_w or -1}:{resize_h or -1}")

    cmd = [
        "ffmpeg", "-y", "-hide_banner",
        "-ss", str(start_sec),
    ]
    if end_sec is not None:
        cmd += ["-to", str(end_sec)]
    cmd += [
        "-i", str(raw_video),
        "-map", "0:0",
    ]
    if vf:
        cmd += ["-vf", ",".join(vf)]
    cmd += [
        "-c:v", "libx264",
        "-preset", "slow",
        "-crf", "18",
        "-pix_fmt", "yuv420p",
        str(processed_video),
    ]
    return cmd


def _resolve_video_params(cfg: dict, project_dir: Path) -> dict:
    """
    Resolve video processing parameters with the following precedence:
      1. Per-project override file: <project_dir>/project_config.yaml
      2. Global config: cfg["video_processing"]
      3. Module defaults

    Returns a flat dict of parameters used downstream.
    """
    vp = (cfg.get("video_processing") or {}).copy()

    # Per-project override
    proj_cfg_path = project_dir / "project_config.yaml"
    if proj_cfg_path.exists():
        with open(proj_cfg_path) as f:
            proj_cfg = yaml.safe_load(f) or {}
        vp.update(proj_cfg.get("video_processing", {}))

    # Defaults
    vp.setdefault("raw_video_filename", "camera_a.mkv")
    vp.setdefault("device", "cuda")
    vp.setdefault("text_prompt", "person")
    vp.setdefault("chunk_size", 400)
    vp.setdefault("iou_threshold", 0.15)
    vp.setdefault("max_objects", 3)
    vp.setdefault("pose_mask_threshold", "0.0")
    vp.setdefault("pose_model_complexity", "2")
    vp.setdefault("start_trim_sec", 0)
    vp.setdefault("end_trim_sec", None)
    vp.setdefault("x_min", 0)
    vp.setdefault("y_min", 0)
    vp.setdefault("x_max", None)
    vp.setdefault("y_max", None)
    vp.setdefault("resize_w", None)
    vp.setdefault("resize_h", None)

    return vp


# =============================================================================
# Public API (called by full_pipeline.py)
# =============================================================================

def run_video_processing(project_dir: Path, cfg: dict) -> bool:
    """
    Run the full video processing stage for a single project directory.

    Expected layout:
        <project_dir>/
            <raw_video_filename>      (e.g. camera_a.mkv)
            [project_config.yaml]     (optional per-project overrides)

    Produces:
        <project_dir>/
            processed_video.mp4
            MaskDir/
            PosesDir/
            Visualizations/

    Parameters
    ----------
    project_dir : Path
        Directory containing the raw video for one recording session.
    cfg : dict
        Global config (the result of yaml.safe_load on config.yaml).

    Returns
    -------
    bool
        True if all steps succeeded, False otherwise.
    """
    project_dir = Path(project_dir)
    params = _resolve_video_params(cfg, project_dir)

    raw_video = project_dir / params["raw_video_filename"]
    if not raw_video.exists():
        print(f"  [video_processing] Raw video not found: {raw_video}")
        return False

    processed_video = project_dir / "processed_video.mp4"
    mask_dir = project_dir / "MaskDir"
    poses_dir = project_dir / "PosesDir"
    vis_dir = project_dir / "Visualizations"

    # Reset output dirs for a clean run
    _wipe(mask_dir)
    _wipe(poses_dir)
    _wipe(vis_dir)

    env = _build_env(max_objects=params.get("max_objects"))
    device = params["device"]

    try:
        # ---------- Step 1: clean processed video ----------
        _run(
            _build_ffmpeg_cmd(raw_video, processed_video, params),
            "FFmpeg trim + crop",
            env,
        )

        # ---------- Step 2: SAM tracking ----------
        _run(
            [
                "psifx", "video", "tracking", "sam3", "inference",
                "--video", str(processed_video),
                "--mask_dir", str(mask_dir),
                "--text_prompt", params["text_prompt"],
                "--chunk_size", str(params["chunk_size"]),
                "--iou_threshold", str(params["iou_threshold"]),
                "--device", device,
            ],
            "SAM tracking",
            env,
        )

        # ---------- Step 3: tracking visualization ----------
        _run(
            [
                "psifx", "video", "tracking", "visualization",
                "--video", str(processed_video),
                "--masks", str(mask_dir),
                "--visualization", str(vis_dir / "tracking.mp4"),
                "--labels", "--color",
            ],
            "Tracking visualization",
            env,
        )

        # ---------- Step 4: pose inference ----------
        _run(
            [
                "psifx", "video", "pose", "mediapipe", "multi-inference",
                "--video", str(processed_video),
                "--masks", str(mask_dir),
                "--poses_dir", str(poses_dir),
                "--mask_threshold", str(params["pose_mask_threshold"]),
                "--model_complexity", str(params["pose_model_complexity"]),
                "--smooth",
                "--device", device,
            ],
            "Pose inference",
            env,
        )

        # ---------- Step 5: pose visualization ----------
        _run(
            [
                "psifx", "video", "pose", "mediapipe", "visualization",
                "--video", str(processed_video),
                "--poses", str(poses_dir),
                "--visualization", str(vis_dir / "pose.mp4"),
                "--confidence_threshold", "0.0",
            ],
            "Pose visualization",
            env,
        )

    except RuntimeError as e:
        print(f"  [video_processing] {e}")
        return False
    except FileNotFoundError as e:
        # Most likely ffmpeg or psifx not on PATH
        print(f"  [video_processing] Missing external tool: {e}")
        return False

    print(f"  [video_processing] Done: {project_dir.name}")
    return True


# =============================================================================
# Standalone CLI entry point
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Run video preprocessing + tracking + pose inference for one project."
    )
    parser.add_argument(
        "--project_dir", type=str, required=True,
        help="Path to the project directory containing the raw video.",
    )
    parser.add_argument(
        "--config", type=str, default=None,
        help="Path to global config.yaml. Defaults to src/config/config.yaml.",
    )
    args = parser.parse_args()

    project_dir = Path(args.project_dir).resolve()
    if not project_dir.is_dir():
        sys.exit(f"Project directory not found: {project_dir}")

    if args.config:
        config_path = Path(args.config).resolve()
    else:
        # default: src/config/config.yaml relative to this file
        config_path = Path(__file__).resolve().parent.parent / "config" / "config.yaml"

    if not config_path.exists():
        sys.exit(f"Config file not found: {config_path}")

    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    ok = run_video_processing(project_dir, cfg)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
