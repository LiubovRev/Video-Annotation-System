import numpy as np
import pandas as pd

from vas.keypoints import BODY25, MEDIAPIPE_POSE, add_virtual_joints, is_legacy_table, remap_legacy_columns


def test_mediapipe_layout():
    assert len(MEDIAPIPE_POSE) == 33
    assert MEDIAPIPE_POSE[11] == "l_shoulder" and MEDIAPIPE_POSE[24] == "r_hip"


def test_virtual_joints():
    df = pd.DataFrame({"l_shoulder_x": [0.0], "l_shoulder_y": [0.0], "r_shoulder_x": [2.0], "r_shoulder_y": [4.0],
                       "l_hip_x": [1.0], "l_hip_y": [10.0], "r_hip_x": [3.0], "r_hip_y": [10.0]})
    df = add_virtual_joints(df)
    assert df.loc[0, "neck_x"] == 1.0 and df.loc[0, "neck_y"] == 2.0
    assert df.loc[0, "mid_hip_x"] == 2.0


def test_remap_legacy_body25_named_table():
    names = BODY25 + ["background"]
    # value == MediaPipe index stored under the BODY_25 name of that index
    df = pd.DataFrame({f"{n}_{a}": [float(i)] for i, n in enumerate(names) for a in ("x", "y")})
    df["person_label"] = "Patient1"
    assert is_legacy_table(df)
    out = remap_legacy_columns(df)
    assert out.loc[0, "l_shoulder_x"] == 11.0   # was called "r_ankle"
    assert out.loc[0, "l_hip_x"] == 23.0        # was called "r_small_toe"
    assert out.loc[0, "l_knee_x"] == 25.0       # was called "background"
    assert np.isnan(out.loc[0, "r_knee_x"])      # index 26 was never extracted
    assert out.loc[0, "neck_x"] == (11.0 + 12.0) / 2
    assert "background_x" not in out.columns
