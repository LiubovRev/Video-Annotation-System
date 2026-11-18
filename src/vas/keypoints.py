"""Keypoint layouts.

psifx's MediaPipe tools write the 33 MediaPipe *Pose* landmarks in
``pose_keypoints_2d`` as a flat ``[x, y, visibility] * 33`` list
(pixel coordinates). Earlier versions of this repository named those
points with the OpenPose BODY_25 layout, which silently mislabelled
almost every joint (e.g. BODY_25 "neck" = MediaPipe "left_eye_inner").
:func:`remap_legacy_columns` repairs tables produced that way.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

MEDIAPIPE_POSE = [
    "nose",
    "l_eye_inner", "l_eye", "l_eye_outer",
    "r_eye_inner", "r_eye", "r_eye_outer",
    "l_ear", "r_ear",
    "mouth_l", "mouth_r",
    "l_shoulder", "r_shoulder",
    "l_elbow", "r_elbow",
    "l_wrist", "r_wrist",
    "l_pinky", "r_pinky",
    "l_index", "r_index",
    "l_thumb", "r_thumb",
    "l_hip", "r_hip",
    "l_knee", "r_knee",
    "l_ankle", "r_ankle",
    "l_heel", "r_heel",
    "l_foot_index", "r_foot_index",
]

BODY25 = [
    "nose", "neck", "r_shoulder", "r_elbow", "r_wrist",
    "l_shoulder", "l_elbow", "l_wrist", "mid_hip",
    "r_hip", "r_knee", "r_ankle", "l_hip", "l_knee", "l_ankle",
    "r_eye", "l_eye", "r_ear", "l_ear",
    "l_big_toe", "l_small_toe", "l_heel",
    "r_big_toe", "r_small_toe", "r_heel",
]

LAYOUTS = {"mediapipe": MEDIAPIPE_POSE, "body25": BODY25}

# Joints every downstream step relies on.
CORE_JOINTS = [
    "nose", "neck", "mid_hip",
    "l_eye", "r_eye",
    "l_shoulder", "r_shoulder", "l_elbow", "r_elbow", "l_wrist", "r_wrist",
    "l_hip", "r_hip", "l_knee", "r_knee", "l_ankle", "r_ankle",
]


def joint_names(layout: str) -> list[str]:
    try:
        return LAYOUTS[layout]
    except KeyError as e:
        raise ValueError(f"Unknown keypoint layout '{layout}'") from e


def coord_columns(joints: list[str]) -> list[str]:
    return [f"{j}_{a}" for j in joints for a in ("x", "y")]


def add_virtual_joints(df: pd.DataFrame) -> pd.DataFrame:
    """Add ``neck`` (mid-shoulders) and ``mid_hip`` (mid-hips) when missing."""
    for name, (a, b) in {"neck": ("l_shoulder", "r_shoulder"), "mid_hip": ("l_hip", "r_hip")}.items():
        if f"{name}_x" in df.columns:
            continue
        if f"{a}_x" in df.columns and f"{b}_x" in df.columns:
            for ax in ("x", "y"):
                df[f"{name}_{ax}"] = (df[f"{a}_{ax}"] + df[f"{b}_{ax}"]) / 2.0
    return df


def ensure_core_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Guarantee all core joint columns exist (NaN when unavailable)."""
    df = add_virtual_joints(df)
    for col in coord_columns(CORE_JOINTS):
        if col not in df.columns:
            df[col] = np.nan
    return df


def is_legacy_table(df: pd.DataFrame) -> bool:
    """Tables from the old extractor carry a BODY_25 'background' joint."""
    return "background_x" in df.columns


def remap_legacy_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Rename BODY_25-named columns that actually hold MediaPipe landmarks.

    The old extractor kept indices 0–25 only, so MediaPipe landmarks 26–32
    (right knee, ankles, heels, foot indices) are lost and become NaN.
    Re-running pose extraction from the psifx archives is preferable.
    """
    legacy_names = BODY25 + ["background"]
    out = df.drop(columns=coord_columns(legacy_names), errors="ignore").copy()
    for idx, mp_name in enumerate(MEDIAPIPE_POSE):
        if idx < len(legacy_names):
            for ax in ("x", "y"):
                src = f"{legacy_names[idx]}_{ax}"
                out[f"{mp_name}_{ax}"] = df[src] if src in df.columns else np.nan
        else:
            out[f"{mp_name}_x"] = np.nan
            out[f"{mp_name}_y"] = np.nan
    logger.warning(
        "Remapped a legacy BODY_25-named table to MediaPipe names; landmarks 26-32 "
        "(r_knee, ankles, heels, feet) are unavailable. Re-extract from psifx archives if possible."
    )
    return add_virtual_joints(out)
