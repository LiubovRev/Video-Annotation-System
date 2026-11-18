"""End-to-end run on synthetic sessions (no GPU, ffmpeg or psifx needed)."""

import json

import joblib
import numpy as np
import pandas as pd
import pytest

from vas.cli import main
from vas.train import cv_groups, session_split, time_split
from vas.utils import read_table


@pytest.fixture(scope="module")
def ran(demo_root, demo_cfg_path):
    assert main(["-c", str(demo_cfg_path), "--log-level", "WARNING", "run"]) == 0
    return demo_root


def test_outputs_exist(ran):
    sessions = sorted((ran / "data" / "raw").glob("1-1-2024_*"))
    assert len(sessions) == 3
    for s in sessions:
        out = ran / "outputs" / s.name
        for f in ["processed_data.parquet", "labeled_features.parquet", "predictions.parquet",
                  "predicted_segments.csv", "processing_log.log", "pose_clustering/summary.json"]:
            assert (out / f).exists(), f


def test_bundle_contents(ran):
    bundle = joblib.load(ran / "models" / "model_bundle.joblib")
    assert {"model", "classes", "feature_names", "features_config", "metrics"} <= set(bundle)
    assert set(bundle["classes"]) <= {"CST", "CCR", "CHO", "TST", "TRE", "THO"}
    assert "CS" not in bundle["classes"]
    metrics = json.loads((ran / "outputs" / "training" / "model_metrics.json").read_text())
    split = metrics["split"]
    assert split["strategy"] == "session"
    assert not set(split["train_sessions"]) & set(split["test_sessions"])
    assert metrics["test_metrics"][metrics["best_model"]]["f1_macro"] > 0.5


def test_predictions_are_label_strings(ran):
    s = next((ran / "outputs").glob("1-1-2024_#1*"))
    pred = read_table(s / "predictions.parquet")
    bundle = joblib.load(ran / "models" / "model_bundle.joblib")
    assert set(pred["predicted_label"]) <= set(bundle["classes"])
    assert pred["predicted_proba"].between(0, 1).all()
    seg = pd.read_csv(s / "predicted_segments.csv")
    assert (seg["end_s"] >= seg["start_s"]).all()
    assert seg["start_s"].min() >= 2.0  # shifted to original-video time (start_trim_sec=2)


def test_rerun_uses_cache(ran, demo_cfg_path):
    before = (ran / "models" / "model_bundle.joblib").stat().st_mtime
    assert main(["-c", str(demo_cfg_path), "--log-level", "WARNING", "run"]) == 0
    assert (ran / "models" / "model_bundle.joblib").stat().st_mtime == before


def test_predict_cli_on_file(ran, demo_cfg_path, tmp_path):
    src = next((ran / "outputs").glob("1-1-2024_#2*")) / "processed_data.parquet"
    assert main(["-c", str(demo_cfg_path), "--log-level", "WARNING", "predict",
                 "--input", str(src), "--output-dir", str(tmp_path)]) == 0
    assert any(tmp_path.glob("predictions_*.parquet"))


def _toy(n_sessions=4, n=100):
    return pd.DataFrame({
        "session": np.repeat([f"s{i}" for i in range(n_sessions)], n),
        "time_s": np.tile(np.arange(n) / 10.0, n_sessions),
        "annotation_label": np.tile(np.where(np.arange(n) % 2, "A", "B"), n_sessions),
    })


def test_session_split_disjoint():
    df = _toy()
    tr, te = session_split(df, 0.25, seed=0)
    assert not set(df.session.iloc[tr]) & set(df.session.iloc[te])


def test_time_split_purges_boundary():
    df = _toy(1, 100)
    tr, te = time_split(df, 0.2, purge_gap_s=1.0)
    assert df.time_s.iloc[tr].max() < df.time_s.iloc[te].min() - 1.0


def test_cv_groups_time_blocks_for_single_session():
    df = _toy(1, 90)
    g = cv_groups(df, 3)
    assert len(np.unique(g)) == 3
    assert (np.diff(g) >= 0).all()  # contiguous blocks
