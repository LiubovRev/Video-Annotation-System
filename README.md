# Clinical Video ML Pipeline for Behavioral Analysis

A reproducible machine learning pipeline for **batch analysis of clinical
therapy-session videos**. Developed during applied ML work with clinical
video data at Lausanne University Hospital (CHUV).

The pipeline ingests raw therapy session videos, extracts multi-person
pose keypoints, engineers biomechanically-meaningful features, aligns
expert annotations to frames, and trains gradient-boosted classifiers
to recognize behavioral classes from skeletal kinematics.

---

## Scope and Status

This is a **research-grade pipeline with production engineering practices**.
It is intentionally framed as a **batch analysis tool** for the research
team, not a deployed inference service.

**What it is:**
- An end-to-end *batch* pipeline: raw video → labeled keypoints → trained
  classifiers → predictions and evaluation artifacts.
- Config-driven, with per-stage skip flags, reproducible feature
  engineering, time-aware validation, and explicit overfitting checks.
- Built to be re-run on new recording sessions by the research team.

**What it is not:**
- Not a deployed real-time inference service.
- No REST / gRPC interface, no container image, no CI.
- Model versioning, drift monitoring, and integration tests are out of
  scope for the current use case.

To take this to a production inference setting, the natural next steps
would be: wrapping `predict.py` as a FastAPI service, containerizing
with Docker, adding a model registry (e.g. MLflow), and monitoring
for input drift in pose-keypoint distributions across recording
conditions.

---

## What This Pipeline Does

```
Raw video + expert annotations (timestamped behavioral labels)  
        │  
        ▼  
Step 1 — Video preprocessing       (ffmpeg trim/crop, SAM tracking,  
                                    MediaPipe multi-person pose)  
        │  
        ▼  
Step 2 — Pose extraction           (parse psifx JSON, build keypoint  
                                    DataFrame, per-person assignment)  
        │  
        ▼  
Step 3 — Pose clustering           (optional; movement-based phases)  
        │  
        ▼  
Step 4 — Annotation alignment      (map timestamped labels to frames,  
                                    auto-trim, generate labeled dataset)  
        │  
        ▼  
Step 5 — Model training            (feature engineering + 3-model  
                                    benchmark with time-aware split)  
        │  
        ▼  
Step 6 — Prediction                (inference + confusion matrix)  
```

Each stage can be enabled or skipped via `src/config/config.yaml`.

---

## Engineering Practices

The repository is structured around a few deliberate decisions:

- **Config-driven execution.** All paths, hyperparameters, and per-stage
  skip flags live in `src/config/config.yaml`. Per-project overrides via
  an optional `project_config.yaml` inside each project directory.
- **Modular pipeline stages.** Each stage exposes a `run_*()` function
  (`run_video_processing`, `run_pose_extraction`,
  `run_annotation_alignment`, `train_model`, `predict_annotations`) that
  the orchestrator calls. Stages can also be run standalone via CLI.
- **Idempotent training.** The orchestrator skips training when a saved
  model artifact already exists; rerunning is cheap.
- **Time-aware train/test split.** When `time_s` is available, the
  training script splits on the 80th time-percentile rather than
  randomly, to avoid temporal leakage between adjacent frames of the
  same session. Falls back to stratified split when class coverage
  isn't preserved.
- **Explicit overfitting detection.** Train-vs-test F1 gap is logged per
  model and flagged at configurable thresholds (`good` / `moderate` /
  `severe`) — surfaced both in console output and in `model_metrics.json`.
- **Reproducibility artifacts.** Each training run saves
  `feature_names.json`, `model_metrics.json`, `model_comparison.png`,
  and `feature_importance.png` alongside the model file.

---

## Feature Engineering

The core ML signal comes from biomechanical features derived from raw
pose keypoints. This is the part most worth reading:

- **Normalization.** Keypoints are translated relative to mid-hip and
  scaled by torso length (neck-to-mid-hip), making features invariant
  to camera distance and subject size.
- **Joint angles.** 9 anatomical angles: elbow (L/R), shoulder (L/R),
  knee (L/R), hip (L/R), trunk.
- **Distances.** Inter-keypoint distances expressing posture and
  reach: eye-to-eye, nose-to-neck, wrist-to-hip (L/R), wrist-to-nose
  (L/R), nose-to-ankles, hip-to-ankle.
- **Symmetry features.** Left-vs-right diffs on shoulders, hips,
  elbow angles, knee angles, wrist-to-hip, shoulder angles.
- **Center of mass and body spread.** COM as mid-hip / neck midpoint;
  body spread in x and y as bounding ranges across wrists and ankles.
- **Temporal derivatives.** Per-keypoint velocity and acceleration on a
  curated subset (COM, nose, wrists, ankles, neck, mid-hip), grouped
  by annotation segment so derivatives don't cross segment boundaries.

After feature engineering, three feature-selection passes are applied:
zero-variance removal, low-correlation-with-target removal, and
redundancy removal (pairwise correlation > 0.95).

---

## Models

Three gradient-boosted classifiers are benchmarked on the same
train/test split:

| Model                  | Notable settings                                   |
|------------------------|----------------------------------------------------|
| LightGBM               | Early stopping on a held-out 15% slice of train    |
| XGBoost                | `RandomizedSearchCV` over 9 hyperparameters, 3-fold|
| HistGradientBoosting   | Early stopping with `n_iter_no_change=20`          |

All three use balanced or regularized settings appropriate for the
class imbalance present in clinical behavioral data. The best model
by weighted F1 on the held-out split is saved as the final artifact.

---

## Repository Layout

```
Video-Annotation-System/
├── src/
│   ├── pipeline/
│   │   └── full_pipeline.py         # Orchestrator
│   ├── video_processing/
│   │   └── processing.py            # Step 1: ffmpeg + psifx (SAM + MediaPipe)
│   ├── pose/
│   │   ├── extractor.py             # Step 2: JSON → keypoint DataFrame
│   │   ├── clustering.py            # Step 3 (optional)
│   │   └── convert_csv_to_parquet.py
│   ├── annotations/
│   │   └── generator.py             # Step 4: align labels to frames
│   ├── models/
│   │   ├── train.py                 # Step 5: features + training + eval
│   │   └── predict.py               # Step 6: inference
│   └── config/
│       └── config.yaml              # Central configuration
├── tests/
│   └── test_config.py
├── data/                            # Project-organized inputs (gitignored)
├── outputs/                         # Generated artifacts (gitignored)
├── requirements.txt
└── README.md
```

---

## Setup

External dependencies (must be installed and on `PATH`):

- `ffmpeg`
- [`psifx`](https://github.com/idiap/psifx) — installed locally as a
  development package; provides SAM-based tracking and MediaPipe
  multi-person pose inference.

A Hugging Face token is required for some tracking models. The pipeline
reads `HF_TOKEN` from the environment and forwards it to psifx as
`HUGGINGFACE_HUB_TOKEN`.

```bash
# Create environment (uv recommended; venv also works)
uv venv .venv
source .venv/bin/activate

# Install psifx (local package — adjust path)
uv pip install -e /path/to/psifx

# Install project dependencies
uv pip install -r requirements.txt

# HF token (optional, only for some tracking models)
export HF_TOKEN=your_token_here
```

---

## Usage

### Full pipeline

```bash
python src/pipeline/full_pipeline.py
```

The orchestrator iterates over project directories under
`data/raw/<project_name>/`, runs each enabled stage, and writes
artifacts to `outputs/<project_name>/`.

### Individual stages

Each stage can also be run standalone:

```bash
# Step 1: video preprocessing for one project
python src/video_processing/processing.py --project_dir data/raw/<project_name>

# Step 5: model training from a labeled CSV
python src/models/train.py \
    --input_csv outputs/combined_labeled_features.csv \
    --output_dir outputs/ \
    --config src/config/config.yaml

# Step 6: prediction
python src/models/predict.py --config src/config/config.yaml
```

### Tests

```bash
pytest tests/
```

---

## Outputs

Per project (`outputs/<project_name>/`):

```
processed_data.csv               # Merged keypoint DataFrame
labeled_features.csv             # After annotation alignment
predictions_processed_data.csv   # If prediction stage is enabled
```

Global (`outputs/`):

```
combined_labeled_features.csv    # Concatenation across all projects
model_<best_name>.joblib         # Best classifier by test F1
feature_names.json               # Exact feature order used at training
model_metrics.json               # Train/test metrics + overfitting flags
model_comparison.png             # Three-model comparison plot
feature_importance.png           # Top-20 importances per tree model
```

---

## Logging

Each project produces a `processing_log.log` capturing the full
configuration, executed commands, subprocess stdout/stderr, per-step
runtime, and final pipeline state. This is the primary mechanism for
debugging and for reproducing a past run.

---

## Limitations

- **Multi-person identity is heuristic.** Person IDs from SAM tracking
  are mapped via `pose_extraction.id_to_label`; durable identity across
  long videos is not guaranteed and is reviewed manually.
- **Annotation schema is project-specific.** The `label_to_class`
  mapping in `train.py` encodes a specific therapy-session taxonomy and
  is not transferable as-is to other clinical settings.
- **CUDA-only by default.** SAM tracking and pose inference are
  configured for GPU; CPU fallback is possible via psifx but slow.
- **Not validated on external datasets.** Results to date are
  in-distribution for the CHUV recording setup.
