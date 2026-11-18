"""Synthetic sessions in the exact on-disk formats the pipeline consumes.

Used by the test-suite and for a GPU-free demo (``scripts/make_demo_data.py``):
psifx-style ``PosesDir/<track>.tar.gz`` archives with MediaPipe landmarks,
a WAKEE annotation export and a ``project_config.yaml`` per session.
"""

from __future__ import annotations

import io
import json
import tarfile
from pathlib import Path

import numpy as np
import yaml

from .config import DEFAULT_CONFIG_PATH, deep_merge
from .keypoints import MEDIAPIPE_POSE

# Template skeleton in torso units (x right, y down), origin at mid-hip.
_TEMPLATE = {
    "nose": (0, -1.30), "l_eye_inner": (0.04, -1.38), "l_eye": (0.08, -1.38), "l_eye_outer": (0.12, -1.38),
    "r_eye_inner": (-0.04, -1.38), "r_eye": (-0.08, -1.38), "r_eye_outer": (-0.12, -1.38),
    "l_ear": (0.18, -1.33), "r_ear": (-0.18, -1.33), "mouth_l": (0.05, -1.20), "mouth_r": (-0.05, -1.20),
    "l_shoulder": (0.25, -1.0), "r_shoulder": (-0.25, -1.0), "l_elbow": (0.35, -0.55), "r_elbow": (-0.35, -0.55),
    "l_wrist": (0.38, -0.10), "r_wrist": (-0.38, -0.10), "l_pinky": (0.40, -0.02), "r_pinky": (-0.40, -0.02),
    "l_index": (0.38, 0.0), "r_index": (-0.38, 0.0), "l_thumb": (0.35, -0.04), "r_thumb": (-0.35, -0.04),
    "l_hip": (0.15, 0.0), "r_hip": (-0.15, 0.0), "l_knee": (0.18, 0.5), "r_knee": (-0.18, 0.5),
    "l_ankle": (0.18, 1.0), "r_ankle": (-0.18, 1.0), "l_heel": (0.18, 1.05), "r_heel": (-0.18, 1.05),
    "l_foot_index": (0.25, 1.05), "r_foot_index": (-0.25, 1.05),
}
_HAND = ["wrist", "pinky", "index", "thumb"]

# behaviour → (role, code)
CODES = {
    "Child": {"still": "CST", "reach": "CCR", "hold": "CHO"},
    "Therapist": {"still": "TST", "reach": "TRE", "hold": "THO"},
}


def _pose(behaviour: str, t: float, rng: np.random.Generator) -> dict[str, np.ndarray]:
    p = {k: np.array(v, dtype=float) for k, v in _TEMPLATE.items()}
    if behaviour == "reach":  # left arm raised and waving
        wave = 0.15 * np.sin(2 * np.pi * 1.2 * t)
        p["l_elbow"] = np.array([0.50, -1.25])
        for h in _HAND:
            p[f"l_{h}"] = np.array([0.60 + wave, -1.70]) + (p[f"l_{h}"] - np.array(_TEMPLATE["l_wrist"]))
    elif behaviour == "hold":  # both hands together in front of the chest
        for side, sx in (("l", 1), ("r", -1)):
            p[f"{side}_elbow"] = np.array([0.30 * sx, -0.65])
            for h in _HAND:
                p[f"{side}_{h}"] = np.array([0.06 * sx, -0.75]) + 0.02 * rng.standard_normal(2)
    sway = 0.03 * np.sin(2 * np.pi * 0.2 * t)
    for k in p:
        p[k] = p[k] + np.array([sway, 0]) + 0.015 * rng.standard_normal(2)
    return p


def _schedule(duration: float, rng: np.random.Generator) -> list[tuple[float, float, str]]:
    out, t = [], 0.0
    behaviours = ["still", "reach", "hold"]
    i = int(rng.integers(3))
    while t < duration:
        d = float(rng.uniform(4, 10))
        out.append((t, min(t + d, duration), behaviours[i % 3]))
        t += d
        i += int(rng.integers(1, 3))
    return out


def _hms(s: float) -> str:
    h, rem = divmod(s, 3600)
    m, sec = divmod(rem, 60)
    return f"{int(h):02d}:{int(m):02d}:{sec:06.3f}"


def wakee_line(tier: str, start: float, end: float, code: str) -> str:
    """One line of a WAKEE annotation export."""
    d = end - start
    return f"{tier}\t\t{_hms(start)}\t{start:.3f}\t{_hms(end)}\t{end:.3f}\t{_hms(d)}\t{d:.3f}\t{code}"


def _write_archive(path: Path, frames: list[list[float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(path, "w:gz") as tar:
        def add(name: str, obj) -> None:
            data = json.dumps(obj).encode()
            info = tarfile.TarInfo(name)
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))

        add("edges.json", {"pose_keypoints_2d": [[11, 12], [11, 13]]})
        for i, kp in enumerate(frames):
            n = 33 * 3
            add(f"{i: 015d}.json", {"pose_keypoints_2d": kp, "face_keypoints_2d": [0.0] * 3,
                                    "hand_left_keypoints_2d": [0.0] * 63, "hand_right_keypoints_2d": [0.0] * 63}
                if kp else {"pose_keypoints_2d": [0.0] * n})


def make_session(raw_root: Path, ann_root: Path, name: str, duration_s: float = 90.0, fps: float = 15.0,
                 start_trim_s: float = 2.0, seed: int = 0) -> Path:
    rng = np.random.default_rng(seed)
    session = Path(raw_root) / name
    persons = {"1": ("Therapist", np.array([300.0, 400.0]), 110.0),
               "2": ("Child", np.array([700.0, 450.0]), 80.0)}
    ann_lines = []
    for tid, (role, origin, scale) in persons.items():
        sched = _schedule(duration_s, rng)
        n = int(duration_s * fps)
        frames = []
        for i in range(n):
            t = i / fps
            beh = next(b for s, e, b in sched if s <= t < e or e >= duration_s)
            if rng.random() < 0.02:  # occasional dropped detection
                frames.append([])
                continue
            pose = _pose(beh, t, rng)
            drift = np.array([20 * np.sin(2 * np.pi * t / 60), 0.0])
            kp = []
            for j in MEDIAPIPE_POSE:
                x, y = origin + drift + scale * pose[j]
                vis = 0.95 if rng.random() > 0.03 else 0.2
                kp += [float(x), float(y), vis]
            frames.append(kp)
        _write_archive(session / "PosesDir" / f"{tid}.tar.gz", frames)
        tier = "c14_CP" if role == "Child" else "t14_TP"
        for s, e, beh in sched:
            code = CODES[role][beh]
            s0, e0 = s + start_trim_s, e + start_trim_s
            ann_lines.append(wakee_line(tier, s0, e0, code))
    # an overlapping vocalisation tier that must be ignored
    ann_lines.append(wakee_line("c14_V", 10.0, 20.0, "CS"))

    Path(ann_root).mkdir(parents=True, exist_ok=True)
    stem = name.split("[")[0]
    (Path(ann_root) / f"{stem}(00)_WAKEE_synthetic.txt").write_text("\n".join(ann_lines) + "\n")
    (session / "project_config.yaml").write_text(yaml.safe_dump({
        "video_processing": {"start_trim_sec": start_trim_s},
        "pose_extraction": {"fps": fps, "id_to_label": {1: "Therapist1", 2: "Patient1"}},
    }))
    return session


def make_demo_project(root: Path, n_sessions: int = 3, duration_s: float = 90.0, seed: int = 0,
                      config_overrides: dict | None = None) -> Path:
    """Create ``root/{configs,data}`` with synthetic sessions. Returns the config path."""
    root = Path(root)
    for i in range(n_sessions):
        make_session(root / "data" / "raw", root / "data" / "annotations",
                     f"1-1-2024_#{i + 1}_SYNTHETIC_[00]", duration_s=duration_s, seed=seed + i)
    cfg = yaml.safe_load(DEFAULT_CONFIG_PATH.read_text())
    cfg = deep_merge(cfg, {
        "directories": {"root": ".."},
        "flags": {"skip_video_processing": True},
        "model_training": {"n_iter_search": 4, "min_samples_per_class": 20},
        "plotting": {"dpi": 60},
    })
    cfg = deep_merge(cfg, config_overrides or {})
    cfg_path = root / "configs" / "config.yaml"
    cfg_path.parent.mkdir(parents=True, exist_ok=True)
    cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False))
    return cfg_path
