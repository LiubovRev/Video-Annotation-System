#!/usr/bin/env python3
from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Optional

# ==========================================================
# ===================== SETTINGS ===========================
# ==========================================================

DEVICE = "cuda"

TEXT_PROMPT = "person"
CHUNK_SIZE = 400
IOU_THRESHOLD = 0.15
MAX_OBJECTS: Optional[int] = 3

START_TRIM_SEC = 0
END_TRIM_SEC = 100

# ----------------------------------------------------------

X_MIN = 40
Y_MIN = 40
X_MAX = 1240
Y_MAX = 700

# keep resolution HIGH
RESIZE_W = None
RESIZE_H = None

# ----------------------------------------------------------
POSE_MASK_THRESHOLD = "0.0"     #  critical for seated people
POSE_MODEL_COMPLEXITY = "2"

BASE_PATH = Path(
    "/home/liubov/Bureau/new/6-2-2024_#10_INDIVIDUAL_[12]"
)

RAW_VIDEO = BASE_PATH / "camera_a.mkv"
PROCESSED_VIDEO = BASE_PATH / "processed_video.mp4"

MASK_DIR = BASE_PATH / "MaskDir"
POSES_DIR = BASE_PATH / "PosesDir"
VIS_DIR = BASE_PATH / "Visualizations"
LOG_FILE = BASE_PATH / "processing_log.log"

# ==========================================================
# ===================== UTILITIES ==========================
# ==========================================================

def run(cmd, name, env):
    print(f"\n>>> {name}")
    print(" ".join(map(str, cmd)))

    r = subprocess.run(
        cmd,
        text=True,
        capture_output=True,
        env=env,
    )

    if r.returncode != 0:
        print(r.stderr)
        raise RuntimeError(name)

    print(r.stdout)


def wipe(p: Path):
    if p.exists():
        shutil.rmtree(p)
    p.mkdir(parents=True, exist_ok=True)


def build_env():
    env = os.environ.copy()

    env.setdefault(
        "PYTORCH_CUDA_ALLOC_CONF",
        "expandable_segments:True"
    )

    if MAX_OBJECTS:
        env["PSIFX_MAX_OBJECTS"] = str(MAX_OBJECTS)

    if "HF_TOKEN" in env:
        env["HUGGINGFACE_HUB_TOKEN"] = env["HF_TOKEN"]

    return env


# ===================== FFMPEG =============================
# ==========================================================
def build_ffmpeg():
    crop = f"crop={X_MAX-X_MIN}:{Y_MAX-Y_MIN}:{X_MIN}:{Y_MIN}"

    vf = [crop]

    if RESIZE_W or RESIZE_H:
        vf.append(
            f"scale={RESIZE_W or -1}:{RESIZE_H or -1}"
        )

    return [
        "ffmpeg","-y","-hide_banner",
        "-ss",str(START_TRIM_SEC),
        "-to",str(END_TRIM_SEC),
        "-i",str(RAW_VIDEO),
        "-map","0:0",
        "-vf",",".join(vf),

        #  CLEAN RE-ENCODE
        "-c:v","libx264",
        "-preset","slow",
        "-crf","18",
        "-pix_fmt","yuv420p",

        str(PROCESSED_VIDEO)
    ]


# ==========================================================
# ===================== PIPELINE ===========================
# ==========================================================

def main():

    env = build_env()

    wipe(MASK_DIR)
    wipe(POSES_DIR)
    wipe(VIS_DIR)

    # ------------------------------------------------------
    # STEP 1 — Clean processed video
    # ------------------------------------------------------
    run(build_ffmpeg(),
        "Create processed video",
        env)

    # ------------------------------------------------------
    # STEP 2 — SAM3 Tracking
    # ------------------------------------------------------
    run([
        "psifx","video","tracking","sam3","inference",
        "--video",str(PROCESSED_VIDEO),
        "--mask_dir",str(MASK_DIR),
        "--text_prompt",TEXT_PROMPT,
        "--chunk_size",str(CHUNK_SIZE),
        "--iou_threshold",str(IOU_THRESHOLD),
        "--device",DEVICE,
    ],"SAM3 tracking",env)

    # ------------------------------------------------------
    # STEP 3 — Tracking visualization
    # ------------------------------------------------------
    run([
        "psifx","video","tracking","visualization",
        "--video",str(PROCESSED_VIDEO),
        "--masks",str(MASK_DIR),
        "--visualization",
        str(VIS_DIR/"tracking.mp4"),
        "--labels","--color"
    ],"Tracking visualization",env)

    # ------------------------------------------------------
    # STEP 4 — Pose inference (FIXED)
    # ------------------------------------------------------
    run([
        "psifx","video","pose","mediapipe","multi-inference",
        "--video",str(PROCESSED_VIDEO),
        "--masks",str(MASK_DIR),
        "--poses_dir",str(POSES_DIR),
        "--mask_threshold",POSE_MASK_THRESHOLD,
        "--model_complexity",POSE_MODEL_COMPLEXITY,
        "--smooth",
        "--device",DEVICE
    ],"Pose inference",env)

    # ------------------------------------------------------
    # STEP 5 — Pose visualization
    # ------------------------------------------------------
    run([
        "psifx","video","pose","mediapipe","visualization",
        "--video",str(PROCESSED_VIDEO),
        "--poses",str(POSES_DIR),
        "--visualization",
        str(VIS_DIR/"pose.mp4"),
        "--confidence_threshold","0.0"
    ],"Pose visualization",env)

    print("\n DONE")


if __name__ == "__main__":
    main()
