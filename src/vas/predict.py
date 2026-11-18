"""Step 6 — predict behaviour codes for pose tables with a saved bundle.

Outputs per session (``outputs/<session>/``):

* ``predictions.parquet`` — one row per frame with ``predicted_label``,
  ``predicted_proba`` and per-class probabilities
* ``predicted_segments.csv`` — contiguous runs of the same prediction per
  person (start/end in original-video seconds), ready for review in an
  annotation tool
* ``prediction_report.json`` + ``confusion_matrix.png`` when the session
  also has ground-truth labels
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report

from .config import model_bundle_path, project_config, resolve_paths
from .features import build_features
from .train import compute_metrics, plot_confusion
from .utils import read_table, write_json, write_table

logger = logging.getLogger(__name__)


def load_bundle(path: Path) -> dict:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Model bundle not found: {path}. Train a model first.")
    obj = joblib.load(path)
    if isinstance(obj, dict) and "model" in obj:
        return obj
    # Legacy: bare estimator + feature_names.json next to it.
    feat_file = path.parent / "feature_names.json"
    if not feat_file.exists():
        raise ValueError(f"{path} is a legacy model without a bundle and no feature_names.json was found.")
    logger.warning("Loaded a legacy model (bare estimator). Its features were built by the old, "
                   "buggy feature code — retraining is strongly recommended.")
    classes = [str(c) for c in getattr(obj, "classes_", [])]
    return {"model": obj, "model_name": type(obj).__name__, "classes": classes,
            "feature_names": json.loads(feat_file.read_text()), "features_config": {}, "roles": {}}


def smooth_probabilities(df: pd.DataFrame, proba: np.ndarray, window_s: float) -> np.ndarray:
    """Centred rolling mean of class probabilities within each track segment."""
    if window_s <= 0:
        return proba
    dt = df.groupby("segment_id")["time_s"].diff()
    med = float(np.nanmedian(dt)) if dt.notna().any() else 1 / 15
    win = max(1, int(round(window_s / max(med, 1e-3))))
    p = pd.DataFrame(proba, index=df.index)
    return (p.groupby(df["segment_id"]).rolling(win, center=True, min_periods=1).mean()
            .reset_index(level=0, drop=True).sort_index().to_numpy())


def to_segments(df: pd.DataFrame, min_segment_s: float = 0.0) -> pd.DataFrame:
    """Collapse frame predictions into contiguous segments per person."""
    tcol = "video_time_s" if "video_time_s" in df.columns else "time_s"
    change = (df["predicted_label"] != df["predicted_label"].shift()) | (df["segment_id"] != df["segment_id"].shift())
    run = change.cumsum()
    seg = df.groupby(run).agg(
        session=("session", "first"), person_label=("person_label", "first"),
        label=("predicted_label", "first"), start_s=(tcol, "first"), end_s=(tcol, "last"),
        n_frames=(tcol, "size"), mean_proba=("predicted_proba", "mean"),
    ).reset_index(drop=True)
    seg["duration_s"] = seg["end_s"] - seg["start_s"]
    if min_segment_s > 0:
        seg = seg[seg["duration_s"] >= min_segment_s]
    return seg.reset_index(drop=True)


def predict_frame_table(frames: pd.DataFrame, bundle: dict, cfg: dict) -> pd.DataFrame:
    cfg_feat = dict(cfg)
    cfg_feat["features"] = bundle.get("features_config") or cfg.get("features", {})
    cfg_feat["roles"] = {**(bundle.get("roles") or {}), **(cfg.get("roles") or {})}
    df, _ = build_features(frames, cfg_feat)

    names = bundle["feature_names"]
    missing = [c for c in names if c not in df.columns]
    if missing:
        logger.warning("%d model feature(s) missing in input → NaN: %s", len(missing), missing[:10])
        for c in missing:
            df[c] = np.nan
    X = df[names].to_numpy(dtype=np.float32)

    model = bundle["model"]
    proba = model.predict_proba(X)
    proba = smooth_probabilities(df, proba, float(cfg.get("predict", {}).get("smooth_window_s", 0.0)))
    classes = np.asarray(bundle["classes"]) if bundle.get("classes") else np.asarray(model.classes_)
    best = proba.argmax(axis=1)

    keep = [c for c in ["session", "person_label", "role", "track_id", "frame", "time_s", "video_time_s",
                        "segment_id", "annotation_label"] if c in df.columns]
    out = df[keep].copy()
    out["predicted_label"] = classes[best]
    out["predicted_proba"] = proba[np.arange(len(best)), best]
    for i, c in enumerate(classes):
        out[f"proba_{c}"] = proba[:, i]
    return out


def evaluate_against_truth(pred: pd.DataFrame, out_dir: Path, dpi: int) -> dict | None:
    if "annotation_label" not in pred.columns:
        return None
    labeled = pred[pred["annotation_label"].notna()]
    if labeled.empty:
        return None
    y_true, y_pred = labeled["annotation_label"].astype(str), labeled["predicted_label"].astype(str)
    labels = sorted(set(y_true) | set(y_pred))
    report = {"metrics": compute_metrics(y_true, y_pred),
              "classification_report": classification_report(y_true, y_pred, labels=labels,
                                                             zero_division=0, output_dict=True)}
    write_json(report, out_dir / "prediction_report.json")
    plot_confusion(y_true, y_pred, labels, out_dir / "confusion_matrix.png", dpi, "Predictions vs annotations")
    logger.info("Agreement with annotations: f1_macro=%.4f, balanced_acc=%.4f",
                report["metrics"]["f1_macro"], report["metrics"]["balanced_accuracy"])
    return report


def predict_session(project_dir: Path, cfg: dict, bundle: dict | None = None, force: bool = False) -> Path | None:
    project_dir = Path(project_dir)
    cfg = project_config(cfg, project_dir)
    out_dir = resolve_paths(cfg).outputs / project_dir.name
    out_path = out_dir / "predictions.parquet"
    if out_path.exists() and not force:
        logger.info("Predictions exist — %s", out_path)
        return out_path

    frames_path = out_dir / "processed_data.parquet"
    if not frames_path.exists():
        logger.error("Pose table missing: %s", frames_path)
        return None
    frames = read_table(frames_path)
    frames["video_time_s"] = frames["time_s"] + float(cfg["video_processing"].get("start_trim_sec") or 0)

    # Attach ground truth (if aligned) for evaluation only — never used as input.
    labeled_path = out_dir / "labeled_features.parquet"
    if labeled_path.exists():
        gt = read_table(labeled_path)[["person_label", "time_s", "annotation_label"]]
        frames = frames.merge(gt, on=["person_label", "time_s"], how="left")

    bundle = bundle or load_bundle(model_bundle_path(cfg))
    pred = predict_frame_table(frames, bundle, cfg)
    write_table(pred, out_path)
    segments = to_segments(pred, float(cfg.get("predict", {}).get("min_segment_s", 0.0)))
    write_table(segments, out_dir / "predicted_segments.csv")
    logger.info("Predicted %d frames → %d segments. Distribution: %s", len(pred), len(segments),
                pred["predicted_label"].value_counts().head(8).to_dict())
    evaluate_against_truth(pred, out_dir, cfg.get("plotting", {}).get("dpi", 150))
    return out_path


def predict_file(table_path: Path, cfg: dict, output_dir: Path, bundle_path: Path | None = None) -> Path:
    """Predict for an arbitrary pose table (csv/parquet)."""
    bundle = load_bundle(bundle_path or model_bundle_path(cfg))
    frames = read_table(table_path)
    if "session" not in frames.columns:
        frames["session"] = Path(table_path).stem
    pred = predict_frame_table(frames, bundle, cfg)
    out = Path(output_dir) / f"predictions_{Path(table_path).stem}.parquet"
    write_table(pred, out)
    write_table(to_segments(pred), out.with_name(f"segments_{Path(table_path).stem}.csv"))
    evaluate_against_truth(pred, Path(output_dir), cfg.get("plotting", {}).get("dpi", 150))
    return out
