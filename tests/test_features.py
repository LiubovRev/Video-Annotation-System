import numpy as np
import pandas as pd
import pytest

from vas.features import build_features, joint_angle
from vas.keypoints import MEDIAPIPE_POSE
from vas.synthetic import _TEMPLATE


def _skeleton_frames(n=30, scale=100.0, offset=(500.0, 300.0), person="Patient1", session="s", t0=0.0):
    rows = []
    for i in range(n):
        r = {"session": session, "person_label": person, "time_s": t0 + i / 15}
        for j in MEDIAPIPE_POSE:
            x, y = _TEMPLATE[j]
            r[f"{j}_x"] = offset[0] + scale * x + 3 * np.sin(i / 3)
            r[f"{j}_y"] = offset[1] + scale * y
        rows.append(r)
    return pd.DataFrame(rows)


def test_joint_angle():
    a, b, c = np.array([[1.0, 0.0]]), np.array([[0.0, 0.0]]), np.array([[0.0, 1.0]])
    assert joint_angle(a, b, c)[0] == pytest.approx(90.0)


def test_normalisation_regression():
    """mid_hip must not be zeroed before other joints are normalised."""
    df, _ = build_features(_skeleton_frames(5), {"features": {"interpersonal": False}})
    assert np.allclose(df["n_mid_hip_x"], 0) and np.allclose(df["n_mid_hip_y"], 0)
    assert np.allclose(df["n_neck_y"], -1.0, atol=1e-6)        # neck is one torso length above the hip
    assert np.allclose(df["n_r_hip_x"], -0.15, atol=1e-6)      # joints after mid_hip in column order


def test_static_features_scale_and_translation_invariant():
    cfg = {"features": {"interpersonal": False}}
    a, cols = build_features(_skeleton_frames(scale=100, offset=(500, 300)), cfg)
    b, _ = build_features(_skeleton_frames(scale=60, offset=(100, 50)), cfg)
    static = [c for c in cols if c.endswith("_angle") or c.startswith("n_") or c in ("eye_to_eye", "body_spread_y")]
    assert np.allclose(a[static].to_numpy(), b[static].to_numpy(), atol=1e-6)


def test_features_do_not_depend_on_labels():
    df = pd.concat([_skeleton_frames(20), _skeleton_frames(20, person="Therapist1", offset=(200, 300))])
    df["annotation_label"] = np.where(np.arange(len(df)) % 3 == 0, "CST", "CCR")
    a, cols = build_features(df, {})
    df["annotation_label"] = df["annotation_label"].sample(frac=1, random_state=0).to_numpy()
    b, _ = build_features(df, {})
    pd.testing.assert_frame_equal(a[cols], b[cols])


def test_temporal_features_reset_on_gaps_and_persons():
    df = pd.concat([_skeleton_frames(10), _skeleton_frames(10, t0=100.0),
                    _skeleton_frames(10, person="Therapist1", offset=(100, 100))])
    out, _ = build_features(df, {"features": {"max_gap_s": 0.5}})
    assert out["segment_id"].nunique() == 3
    first_rows = out.groupby("segment_id").head(1)
    assert first_rows["com_vx"].isna().all()


def test_interpersonal_distance():
    df = pd.concat([_skeleton_frames(5, offset=(500, 300)),
                    _skeleton_frames(5, person="Therapist1", offset=(800, 300))])
    out, cols = build_features(df, {})
    assert "other_dist" in cols
    assert np.allclose(out["other_dist"], 300 / 100.0, atol=0.2)
