#!/usr/bin/env python3
# coding: utf-8
"""
predict.py
-----------
Run inference using a trained model on new pose keypoint data.
Feature engineering is identical to training: normalize → derived → temporal.

Outputs saved to output_dir:
  - predictions_<stem>.csv
  - prediction_report_<stem>.txt (if true labels present)
  - confusion_matrix_<stem>.png (if true labels present)
"""

import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import yaml
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from models.train import (
    normalize_keypoints,
    add_derived_pose_features,
    add_temporal_features,
)


# -----------------------------------------------------------------------------
# Core prediction function
# -----------------------------------------------------------------------------
def predict_annotations(data_path, output_dir, cfg=None):
    """
    Predict annotation labels for a CSV of pose keypoints.

    Returns a DataFrame with an added 'predicted_annotation_label' column.
    """
    cfg = cfg or {}
    mc = cfg.get("model", {})
    output_dir_cfg = cfg.get("directories", {}).get("output_base_dir", ".")
    fps = mc.get("fps", 15)
    model_file = mc.get("file", "model_xgboost.joblib")
    feat_file = mc.get("features_file", "feature_names.json")
    label_map = {int(k): v for k, v in (cfg.get("label_map") or {}).items()}
    dpi = cfg.get("plotting", {}).get("dpi", 300)

    output_dir = Path(output_dir)
    model_base_dir = Path(output_dir_cfg)
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- Load model ---
    model_path = model_base_dir / model_file
    if not model_path.exists():
        candidates = sorted(model_base_dir.glob("model_*.joblib"))
        if not candidates:
            raise FileNotFoundError(f"No model found in {model_base_dir}")
        model_path = candidates[-1]
    model = joblib.load(model_path)
    print(f"Loaded model: {model_path.name}")

    # --- Load feature names ---
    feat_path = model_base_dir / feat_file
    if not feat_path.exists():
        raise FileNotFoundError(f"Feature file not found: {feat_path}")
    with open(feat_path) as f:
        feature_names = json.load(f)
    print(f"Using {len(feature_names)} features.")

    # --- Load data ---
    data_path = Path(data_path)
    df = pd.read_csv(data_path)
    print(f"Loaded data: {len(df):,} rows")

    # --- Feature engineering ---
    keypoints = [c for c in df.columns if "_x" in c or "_y" in c]
    df = normalize_keypoints(df, keypoints)
    df = add_derived_pose_features(df)
    group_col = "person_label" if "person_label" in df.columns else None
    if not group_col:
        df["_group"] = "all"
        group_col = "_group"
    df, _ = add_temporal_features(df, group_col=group_col, fps=fps)
    if "_group" in df.columns:
        df = df.drop(columns=["_group"])
    if "person_label" in df.columns:
        df = pd.get_dummies(df, columns=["person_label"], prefix="person")

    # --- Align features ---
    missing = [f for f in feature_names if f not in df.columns]
    for col in missing:
        df[col] = 0.0
    df = df[[f for f in feature_names if f in df.columns]]

    X = df[feature_names].copy()
    nan_count = X.isnull().sum().sum()
    if nan_count:
        X = pd.DataFrame(SimpleImputer(strategy="median").fit_transform(X), columns=feature_names)
    X.replace([np.inf, -np.inf], 0.0, inplace=True)

    # --- Predict ---
    # Reload raw data to attach predictions (df above was reduced to feature columns only)
    data = pd.read_csv(data_path).iloc[:len(X)].copy().reset_index(drop=True)
    preds = model.predict(X)
    data["predicted_annotation_label"] = preds
    if label_map:
        data["predicted_annotation_label_str"] = data["predicted_annotation_label"].map(label_map)

    stem = data_path.stem
    pred_file = output_dir / f"predictions_{stem}.csv"
    data.to_csv(pred_file, index=False)
    print(f"Predictions saved: {pred_file.name}\nDistribution:\n{data['predicted_annotation_label'].value_counts()}")

    # --- Optional report if true labels exist ---
    if "annotation_label" in data.columns:
        y_true, y_pred = data["annotation_label"], preds
        labels = sorted(y_true.unique())
        acc = accuracy_score(y_true, y_pred)
        report_file = output_dir / f"prediction_report_{stem}.txt"
        with open(report_file, "w") as f:
            f.write(f"Accuracy: {acc:.4f}\n\n")
            f.write(classification_report(y_true, y_pred, digits=4))
        print(f"Accuracy: {acc:.4f} — report saved: {report_file.name}")

        # Confusion matrix plot
        cm = confusion_matrix(y_true, y_pred, labels=labels)
        fig, ax = plt.subplots(figsize=(12, 10))
        sns.heatmap(cm, annot=True, fmt="d", cmap="viridis",
                    xticklabels=labels, yticklabels=labels, ax=ax)
        ax.set_title("Confusion Matrix", fontsize=18)
        ax.set_xlabel("Predicted", fontsize=14)
        ax.set_ylabel("True", fontsize=14)
        cm_file = output_dir / f"confusion_matrix_{stem}.png"
        plt.savefig(cm_file, dpi=dpi, bbox_inches="tight", facecolor="white")
        plt.close()
        print(f"Confusion matrix saved: {cm_file.name}")

    return data


# -----------------------------------------------------------------------------
# Batch prediction
# -----------------------------------------------------------------------------
def run_batch_predictions(cfg: dict) -> dict:
    raw_root = Path(cfg["directories"]["raw_video_root"])
    output_base = Path(cfg["directories"]["output_base_dir"])
    pose_dir = cfg.get("pose_extraction", {}).get("directory", "PosesDir")

    results = {}
    for project_dir in sorted(d for d in raw_root.iterdir() if d.is_dir()):
        project_name = project_dir.name
        data_path = project_dir / pose_dir / "processed_data.csv"
        output_dir = output_base / project_name

        if not data_path.exists():
            print(f"Skipping {project_name} — no processed_data.csv")
            results[project_name] = "skipped"
            continue

        pred_file = output_dir / f"predictions_processed_data.csv"
        if pred_file.exists():
            print(f"Skipping {project_name} — predictions already exist")
            results[project_name] = "cached"
            continue

        print(f"\nPredicting: {project_name}")
        try:
            results[project_name] = predict_annotations(data_path, output_dir, cfg)
        except Exception as e:
            print(f"Error predicting {project_name}: {e}")
            results[project_name] = f"error: {e}"

    return results


# -----------------------------------------------------------------------------
# Standalone entry point
# -----------------------------------------------------------------------------
def main():
    import argparse

    parser = argparse.ArgumentParser(description="Predict annotations from pose keypoints.")
    parser.add_argument("--config", type=str, default=None, help="Path to config.yaml")
    args = parser.parse_args()

    # Load configuration
    config_file = Path(args.config or Path(__file__).parent.parent.parent / "config.yaml")
    if not config_file.exists():
        raise FileNotFoundError(f"Config file not found: {config_file}")
    with open(config_file) as f:
        cfg = yaml.safe_load(f)

    # Single-project mode
    data_path = cfg.get("predict", {}).get("data_path")
    output_base = Path(cfg["directories"]["output_base_dir"])

    if data_path:
        project_name = Path(data_path).parent.parent.name
        output_dir = output_base / project_name
        result = predict_annotations(data_path, output_dir, cfg)
        print(f"\nDone. {len(result):,} rows predicted.")
    else:
        # Batch mode
        results = run_batch_predictions(cfg)
        print("\nBATCH PREDICTION SUMMARY")
        for name, status in results.items():
            label = "OK" if isinstance(status, pd.DataFrame) else str(status)
            print(f"  {label:15s}  {name}")


if __name__ == "__main__":
    main()
