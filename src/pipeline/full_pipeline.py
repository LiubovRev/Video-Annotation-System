#!/usr/bin/env python3
# coding: utf-8
"""
full_pipeline.py
----------------
End-to-end orchestrator for the Video Annotation System.

Pipeline stages (per project):
  Step 1 — Video processing
  Step 2 — Pose extraction
  Step 3 — Pose clustering  [optional]
  Step 4 — Annotation alignment
  Step 5 — Model training

Configuration is loaded from config.yaml at the project root.
"""

import os
import sys
from pathlib import Path
import yaml
import pandas as pd
import numpy as np
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt

# -------------------------
# Project root and imports
# -------------------------
PROJECT_ROOT = Path(__file__).parent.parent  # src/
sys.path.insert(0, str(PROJECT_ROOT))

# Import pipeline components
try:
    from video_processing.processing import run_video_processing
    from pose.extractor import run_pose_extraction
    from pose.clustering import run_pose_clustering
    from annotations.generator import run_annotation_alignment
    from models.train import train_model
    from models.predict import predict_annotations, run_batch_predictions
except ModuleNotFoundError as e:
    raise ModuleNotFoundError(
        f"Failed to import modules: {e}. "
        "Check that PYTHONPATH includes the project root and all src submodules exist."
    )

# -------------------------
# Load configuration
# -------------------------
# -------------------------
# Load configuration (fixed paths)
# -------------------------
CONFIG_FILE = PROJECT_ROOT.parent / "src" / "config" / "config.yaml"  # project root/config/
if not CONFIG_FILE.exists():
    raise FileNotFoundError(f"Config file not found: {CONFIG_FILE}")

with open(CONFIG_FILE, "r") as f:
    cfg = yaml.safe_load(f)

# Correct raw_root to point relative to project root
directories_cfg = cfg.get("directories", {})
PROJECT_ROOT_PARENT = PROJECT_ROOT.parent  # video_annotation_system/
raw_root       = PROJECT_ROOT_PARENT / (directories_cfg.get("raw_video_root") or "data/raw")
base_data      = PROJECT_ROOT_PARENT / (directories_cfg.get("base_data_dir") or "data/base")
output_base    = PROJECT_ROOT_PARENT / (directories_cfg.get("output_base_dir") or "outputs")
output_base.mkdir(parents=True, exist_ok=True)

print(f" Loaded config from {CONFIG_FILE}")
print(f"Raw videos: {raw_root}")
print(f"Output base: {output_base}")

# Pipeline flags
flags_cfg = cfg.get("flags", {})
skip_video_processing      = flags_cfg.get("skip_video_processing", False)
skip_pose_extraction       = flags_cfg.get("skip_pose_extraction", False)
skip_pose_clustering       = flags_cfg.get("skip_pose_clustering", True)
skip_annotation_processing = flags_cfg.get("skip_annotation_processing", False)
skip_prediction            = flags_cfg.get("skip_prediction", False)
force_recombine            = flags_cfg.get("force_recombine", False)


# -------------------------
# Discover projects
# -------------------------
project_dirs = [d for d in raw_root.iterdir() if d.is_dir()]
print(f"\nFound {len(project_dirs)} project directories.\n{'=' * 70}")

# -------------------------
# Per-project processing
# -------------------------
for project_dir in project_dirs:
    project_name = project_dir.name
    output_dir = output_base / project_name
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n Processing project: {project_name}")

    # Step 1: Video processing
    if not skip_video_processing:
        if not run_video_processing(project_dir, cfg):
            print("  ✗ Video processing failed, skipping project.")
            continue
    else:
        print("   Skipping video processing step.")

    # Step 2: Pose extraction
    if not skip_pose_extraction:
        if not run_pose_extraction(project_dir, cfg):
            print("   Pose extraction failed, skipping project.")
            continue
    else:
        print("   Skipping pose extraction step.")

    # Merge processed parquet files
    processed_files = list(project_dir.glob("processed_data__*.parquet"))
    if not processed_files:
        print("   No processed parquet files found — skipping project.")
        continue

    combined_csv = output_dir / "processed_data.csv"
    if not combined_csv.exists() or force_recombine:
        df = pd.concat([pd.read_parquet(pf) for pf in processed_files], ignore_index=True)
        # Neutral identity mapping
        id_map = cfg["pose_extraction"].get("id_to_label", {})
        df["person_label"] = df["person_label"].replace(id_map)
        df.to_csv(combined_csv, index=False)
        print(f"   Merged {len(processed_files)} parquet files into {combined_csv.name}")
    else:
        df = pd.read_csv(combined_csv)
        print(f"   Loaded existing CSV: {combined_csv.name} ({len(df)} rows)")

    # Step 3: Pose clustering
    if skip_pose_clustering:
        print("   Skipping pose clustering step.")
    else:
        cluster_path = output_dir / f"pose_clusters_{project_name}.parquet"
        if cluster_path.exists() and not force_recombine:
            print(f"   Existing cluster file found: {cluster_path.name}")
        else:
            print("   Running pose clustering...")
            run_pose_clustering(df=df, output_path=cluster_path, cfg=cfg)

    # Step 4: Annotation alignment
    if skip_annotation_processing:
        print("  Skipping annotation alignment (flag set).")
        continue

    labeled_df = run_annotation_alignment(project_name, cfg, output_dir)
    if labeled_df is None:
        print(f"  Annotation alignment failed, skipping project.")
        continue

    # -------------------------
    # Step 5: Prediction (NEW)
    # -------------------------
    if skip_prediction:
        print("  Skipping prediction step (flag set).")
    else:
        try:
            pred_file = output_dir / "predictions_processed_data.csv"
            if pred_file.exists() and not force_recombine:
                print(f"  Predictions already exist: {pred_file.name}")
            else:
                print("  → Running prediction...")
                predict_annotations(combined_csv, output_dir, cfg)
        except Exception as e:
            print(f"  Prediction failed for {project_name}: {e}")

# -------------------------
# Combine labeled features across projects
# -------------------------
print("\n" + "="*70)
print("Combining labeled data from all projects for model training...")
print("="*70)

combined_path = output_base / "combined_labeled_features.csv"
if combined_path.exists() and not force_recombine:
    print(f" Found existing combined dataset: {combined_path.name}")
else:
    labeled_files = list(output_base.glob("*/labeled_features.csv"))
    if not labeled_files:
        print(" No labeled feature files found — cannot combine.")
        exit()

    print(f" Found {len(labeled_files)} labeled feature files.")
    all_dfs = []
    for f in labeled_files:
        try:
            df_part = pd.read_csv(f)
            df_part["project_name"] = f.parent.name
            all_dfs.append(df_part)
            print(f"  ✓ Loaded: {f.parent.name} ({len(df_part):,} rows)")
        except Exception as e:
            print(f"   Skipping {f.name}: {e}")

    if not all_dfs:
        print(" No valid dataframes to combine")
        exit()

    combined_df = pd.concat(all_dfs, ignore_index=True)
    print(f" Combined dataset shape: {combined_df.shape}")
    combined_df.to_csv(combined_path, index=False)
    print(f" Saved: {combined_path.name}")

# -------------------------
# Step 6: Model Training
# -------------------------
print("\n" + "="*70)
print("MODEL TRAINING")
print("="*70)

model_file = output_base / Path(cfg["model"]["file"]).name
if model_file.exists():
    print(f" Model already exists: {model_file.name} — skipping training.")
    model_package = None
else:
    try:
        model_package = train_model(combined_df, output_base, cfg)
        print(f"\n Training complete.")
        print(f" Best Model: {model_package['model_name']}")
        print(f" Final F1 Score: {model_package['f1_score']:.4f}")
    except Exception as e:
        print(f" Model training failed: {e}")
        import traceback
        traceback.print_exc()
        model_package = None

# -------------------------
# Final Summary
# -------------------------
print("\n" + "="*70)
print("PIPELINE COMPLETE - SUMMARY")
print("="*70)
print(f" All projects processed")
print(f" Results saved in: {output_base}")
if model_package:
    print(f" Best Model: {model_package['model_name']}")
    print(f" F1 Score:   {model_package['f1_score']:.4f}")
print("="*70)
