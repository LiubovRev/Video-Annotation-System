import numpy as np
import pandas as pd

from vas.keypoints import MEDIAPIPE_POSE
from vas.pose_extraction import extract_poses, read_pose_archive, run_pose_extraction
from vas.synthetic import _write_archive
from vas.utils import read_table, write_table


def _frame(scale=1.0, vis=0.9):
    kp = []
    for i, _ in enumerate(MEDIAPIPE_POSE):
        kp += [100.0 + i * scale, 200.0 + i * scale, vis]
    return kp


def _cfg(id_to_label):
    return {"pose_extraction": {"keypoint_format": "mediapipe", "confidence_threshold": 0.5,
                                "id_to_label": id_to_label}}


def test_read_archive_skips_edges_and_empty_frames(tmp_path):
    _write_archive(tmp_path / "1.tar.gz", [_frame(), [], _frame(2.0)])
    frames, kps = read_pose_archive(tmp_path / "1.tar.gz", 33)
    assert frames.tolist() == [0, 2]
    assert kps.shape == (2, 33, 3)
    assert kps[1, 11, 0] == 100.0 + 11 * 2.0


def test_extract_maps_track_ids_and_time(tmp_path):
    _write_archive(tmp_path / "1.tar.gz", [_frame()] * 3)
    _write_archive(tmp_path / "2.tar.gz", [_frame()] * 3)
    df = extract_poses(tmp_path, _cfg({1: "Therapist1", "id_2": "Patient1"}), fps=15, session="s")
    assert set(df["person_label"]) == {"Therapist1", "Patient1"}
    assert np.allclose(sorted(df["time_s"].unique()), [0, 1 / 15, 2 / 15])
    row = df.iloc[0]
    assert row["l_shoulder_x"] == 111.0 and row["r_hip_x"] == 124.0
    assert row["neck_x"] == (111.0 + 112.0) / 2


def test_low_visibility_points_become_nan(tmp_path):
    kp = _frame()
    kp[11 * 3 + 2] = 0.1  # l_shoulder visibility
    _write_archive(tmp_path / "1.tar.gz", [kp])
    df = extract_poses(tmp_path, _cfg({1: "A"}), fps=15, session="s")
    assert np.isnan(df.loc[0, "l_shoulder_x"]) and not np.isnan(df.loc[0, "r_shoulder_x"])


def test_unknown_track_gets_placeholder_label(tmp_path):
    _write_archive(tmp_path / "7.tar.gz", [_frame()])
    df = extract_poses(tmp_path, _cfg({1: "A"}), fps=15, session="s")
    assert df["person_label"].iloc[0] == "Track7"


def test_legacy_table_is_used_and_remapped(demo_cfg, demo_root):
    session = demo_root / "data" / "raw" / "legacy_session"
    poses = session / "PosesDir"
    poses.mkdir(parents=True)
    legacy = pd.DataFrame({"person_label": ["Patient1"], "time_s": [0.0], "avg_pose_conf": [0.9],
                           "background_x": [5.0], "background_y": [6.0], "r_ankle_x": [1.0], "r_ankle_y": [2.0]})
    write_table(legacy, poses / "processed_data.csv")
    out = run_pose_extraction(session, demo_cfg)
    df = read_table(out)
    assert df.loc[0, "l_shoulder_x"] == 1.0 and df.loc[0, "l_knee_x"] == 5.0
