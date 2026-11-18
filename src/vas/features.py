"""Biomechanical feature engineering shared by training and prediction.

All operations are vectorised. Features never use the annotation label,
so exactly the same code runs at training and inference time.

Feature groups
--------------
* normalised coordinates — keypoints relative to mid-hip, scaled by torso
  length (neck ↔ mid-hip), i.e. invariant to camera distance and body size
* joint angles (9), inter-keypoint distances (8), left/right symmetry (6),
  body spread (2)
* temporal — per continuous track segment (same session + person, no gap
  > ``max_gap_s``): whole-body velocity in torso-lengths/s, limb velocities
  in the body frame, speeds, accelerations, rolling mean/std of speeds
* interpersonal — distance and direction to the nearest other person
* role one-hot (``role_Child``, ``role_Therapist`` …)
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from .keypoints import ensure_core_columns

logger = logging.getLogger(__name__)

ANGLES = {
    "r_elbow_angle": ("r_shoulder", "r_elbow", "r_wrist"),
    "l_elbow_angle": ("l_shoulder", "l_elbow", "l_wrist"),
    "r_shoulder_angle": ("neck", "r_shoulder", "r_elbow"),
    "l_shoulder_angle": ("neck", "l_shoulder", "l_elbow"),
    "r_knee_angle": ("r_hip", "r_knee", "r_ankle"),
    "l_knee_angle": ("l_hip", "l_knee", "l_ankle"),
    "r_hip_angle": ("mid_hip", "r_hip", "r_knee"),
    "l_hip_angle": ("mid_hip", "l_hip", "l_knee"),
    "trunk_angle": ("nose", "neck", "mid_hip"),
}
DISTANCES = {
    "eye_to_eye": ("l_eye", "r_eye"),
    "nose_to_neck": ("nose", "neck"),
    "r_wrist_to_hip": ("r_wrist", "mid_hip"),
    "l_wrist_to_hip": ("l_wrist", "mid_hip"),
    "r_wrist_to_nose": ("r_wrist", "nose"),
    "l_wrist_to_nose": ("l_wrist", "nose"),
    "wrist_to_wrist": ("l_wrist", "r_wrist"),
}
LOCAL_MOTION_JOINTS = ["nose", "l_wrist", "r_wrist", "l_ankle", "r_ankle", "neck"]
META_COLUMNS = {"session", "person_label", "track_id", "frame", "time_s", "video_time_s", "role",
                "annotation_label", "label_class", "n_overlapping_labels", "avg_pose_conf", "tier",
                "segment_id"}


def _xy(df: pd.DataFrame, joint: str, prefix: str = "") -> np.ndarray:
    return df[[f"{prefix}{joint}_x", f"{prefix}{joint}_y"]].to_numpy(dtype=float)


def joint_angle(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
    """Angle ABC in degrees for arrays of shape (n, 2)."""
    ba, bc = a - b, c - b
    cos = np.einsum("ij,ij->i", ba, bc) / (np.linalg.norm(ba, axis=1) * np.linalg.norm(bc, axis=1) + 1e-8)
    return np.degrees(np.arccos(np.clip(cos, -1.0, 1.0)))


def _joints(df: pd.DataFrame) -> list[str]:
    return sorted({c[:-2] for c in df.columns if c.endswith("_x") and f"{c[:-2]}_y" in df.columns
                   and not c.startswith(("n_", "com_", "other_"))})


def add_normalised_pose(df: pd.DataFrame) -> pd.DataFrame:
    """Add ``torso_px`` and ``n_<joint>_x/y`` (mid-hip origin, torso units)."""
    neck, hip = _xy(df, "neck"), _xy(df, "mid_hip")
    torso = np.linalg.norm(neck - hip, axis=1)
    torso[~np.isfinite(torso) | (torso < 1e-3)] = np.nan
    torso = pd.Series(torso, index=df.index)
    # Fill missing torso lengths from the same person's median, then global median.
    torso = torso.fillna(torso.groupby([df["session"], df["person_label"]]).transform("median"))
    torso = torso.fillna(torso.median()).fillna(1.0)
    df["torso_px"] = torso.to_numpy()

    hip_x, hip_y = df["mid_hip_x"].to_numpy(), df["mid_hip_y"].to_numpy()  # cache before any change
    new = {}
    for j in _joints(df):
        new[f"n_{j}_x"] = (df[f"{j}_x"].to_numpy() - hip_x) / df["torso_px"].to_numpy()
        new[f"n_{j}_y"] = (df[f"{j}_y"].to_numpy() - hip_y) / df["torso_px"].to_numpy()
    return pd.concat([df, pd.DataFrame(new, index=df.index)], axis=1)


def add_static_features(df: pd.DataFrame) -> pd.DataFrame:
    f = {}
    for name, (a, b, c) in ANGLES.items():
        f[name] = joint_angle(_xy(df, a, "n_"), _xy(df, b, "n_"), _xy(df, c, "n_"))
    for name, (a, b) in DISTANCES.items():
        f[name] = np.linalg.norm(_xy(df, a, "n_") - _xy(df, b, "n_"), axis=1)
    nose = _xy(df, "nose", "n_")
    f["nose_to_ankles"] = (np.linalg.norm(nose - _xy(df, "l_ankle", "n_"), axis=1)
                           + np.linalg.norm(nose - _xy(df, "r_ankle", "n_"), axis=1)) / 2
    f["shoulder_y_diff"] = df["n_l_shoulder_y"] - df["n_r_shoulder_y"]
    f["hip_y_diff"] = df["n_l_hip_y"] - df["n_r_hip_y"]
    f["elbow_angle_diff"] = f["l_elbow_angle"] - f["r_elbow_angle"]
    f["knee_angle_diff"] = f["l_knee_angle"] - f["r_knee_angle"]
    f["shoulder_angle_diff"] = f["l_shoulder_angle"] - f["r_shoulder_angle"]
    f["wrist_to_hip_diff"] = f["l_wrist_to_hip"] - f["r_wrist_to_hip"]
    ext = df[["n_l_wrist_x", "n_r_wrist_x", "n_l_ankle_x", "n_r_ankle_x"]]
    f["body_spread_x"] = ext.max(axis=1) - ext.min(axis=1)
    ext = df[["n_nose_y", "n_l_ankle_y", "n_r_ankle_y", "n_l_wrist_y", "n_r_wrist_y"]]
    f["body_spread_y"] = ext.max(axis=1) - ext.min(axis=1)
    # Centre of mass (raw pixels) for whole-body and interpersonal motion.
    f["com_x"] = (df["mid_hip_x"] + df["neck_x"]) / 2
    f["com_y"] = (df["mid_hip_y"] + df["neck_y"]) / 2
    return pd.concat([df, pd.DataFrame(f, index=df.index)], axis=1)


def assign_segments(df: pd.DataFrame, max_gap_s: float) -> pd.Series:
    """Continuous track segments: same session+person and no gap > max_gap_s.
    ``df`` must be sorted by session, person_label, time_s."""
    same = (df["session"].eq(df["session"].shift()) & df["person_label"].eq(df["person_label"].shift()))
    gap = df["time_s"].diff()
    new = ~same | (gap > max_gap_s) | gap.isna()
    return new.cumsum().rename("segment_id")


def add_temporal_features(df: pd.DataFrame, max_gap_s: float = 0.5, rolling_window_s: float = 1.0) -> pd.DataFrame:
    df = df.sort_values(["session", "person_label", "time_s"]).reset_index(drop=True)
    df["segment_id"] = assign_segments(df, max_gap_s)
    g = df.groupby("segment_id", sort=False)
    dt = g["time_s"].diff()
    f = {}

    def vel(col: pd.Series) -> pd.Series:
        return col.groupby(df["segment_id"]).diff() / dt

    # whole-body motion in torso lengths per second
    f["com_vx"] = vel(df["com_x"]) / df["torso_px"]
    f["com_vy"] = vel(df["com_y"]) / df["torso_px"]
    f["com_speed"] = np.hypot(f["com_vx"], f["com_vy"])
    speeds = ["com_speed"]
    # limb motion in the body frame
    for j in LOCAL_MOTION_JOINTS:
        vx, vy = vel(df[f"n_{j}_x"]), vel(df[f"n_{j}_y"])
        f[f"{j}_vx"], f[f"{j}_vy"] = vx, vy
        f[f"{j}_speed"] = np.hypot(vx, vy)
        speeds.append(f"{j}_speed")
    for s in speeds:
        f[s.replace("_speed", "_acc")] = vel(f[s])

    feats = pd.DataFrame(f, index=df.index).replace([np.inf, -np.inf], np.nan)
    med_dt = float(np.nanmedian(dt)) if dt.notna().any() else 1 / 15
    win = max(3, int(round(rolling_window_s / max(med_dt, 1e-3))))
    roll = feats[speeds].groupby(df["segment_id"]).rolling(win, center=True, min_periods=2)
    means, stds = roll.mean().reset_index(level=0, drop=True), roll.std().reset_index(level=0, drop=True)
    for s in speeds:
        feats[f"{s}_rmean"] = means[s]
        feats[f"{s}_rstd"] = stds[s]
    return pd.concat([df, feats], axis=1)


def add_interpersonal_features(df: pd.DataFrame) -> pd.DataFrame:
    """Distance/direction (torso units) to the nearest other person at the same time."""
    key = ["session", "time_s"]
    left = df[key + ["person_label", "com_x", "com_y"]].reset_index()
    pairs = left.merge(left, on=key, suffixes=("", "_o"))
    pairs = pairs[pairs["person_label"] != pairs["person_label_o"]]
    pairs["dx"] = pairs["com_x_o"] - pairs["com_x"]
    pairs["dy"] = pairs["com_y_o"] - pairs["com_y"]
    pairs["d"] = np.hypot(pairs["dx"], pairs["dy"])
    pairs = pairs.dropna(subset=["d"])
    nearest = pairs.loc[pairs.groupby("index")["d"].idxmin(), ["index", "dx", "dy", "d"]].set_index("index")
    t = df["torso_px"]
    df["other_dist"] = nearest["d"].reindex(df.index) / t
    df["other_dx"] = nearest["dx"].reindex(df.index) / t
    df["other_dy"] = nearest["dy"].reindex(df.index) / t
    dt = df.groupby("segment_id")["time_s"].diff()
    df["other_dist_vel"] = (df.groupby("segment_id")["other_dist"].diff() / dt).replace([np.inf, -np.inf], np.nan)
    return df


def add_role_features(df: pd.DataFrame, roles: dict | None) -> pd.DataFrame:
    roles = roles or {}
    if "role" not in df.columns:
        df["role"] = df["person_label"].map(lambda p: roles.get(p, p))
    for r in sorted(set(roles.values()) | set(df["role"].dropna().unique())):
        df[f"role_{r}"] = (df["role"] == r).astype(np.int8)
    return df


def build_features(frames: pd.DataFrame, cfg: dict | None = None) -> tuple[pd.DataFrame, list[str]]:
    """Return ``(table, feature_columns)``. The table is sorted by
    session, person, time and keeps all input columns."""
    cfg = cfg or {}
    fc = cfg.get("features", {})
    df = frames.copy()
    if "session" not in df.columns:
        df["session"] = "session"
    df = ensure_core_columns(df)
    df = add_normalised_pose(df)
    df = add_static_features(df)
    df = add_temporal_features(df, fc.get("max_gap_s", 0.5), fc.get("rolling_window_s", 1.0))
    if fc.get("interpersonal", True):
        df = add_interpersonal_features(df)
    df = add_role_features(df, cfg.get("roles"))
    feature_cols = feature_columns(df)
    logger.info("Built %d features for %d frames", len(feature_cols), len(df))
    return df, feature_cols


def feature_columns(df: pd.DataFrame) -> list[str]:
    joints = {c[2:-2] for c in df.columns if c.startswith("n_") and c.endswith("_x")}
    raw_coords = {f"{j}_{a}" for j in joints for a in ("x", "y")} | {"com_x", "com_y"}
    exclude = META_COLUMNS | raw_coords | {"torso_px"}
    return [c for c in df.columns if c not in exclude and pd.api.types.is_numeric_dtype(df[c])]
