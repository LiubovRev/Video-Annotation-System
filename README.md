# Video Annotation System

[![Python Version](https://img.shields.io/badge/python-3.10+-blue)](https://www.python.org/)

## Overview
This project provides a lightweight annotation tool designed to support machine learning workflows for clinical video analysis.

Developed during work at CHUV (Lausanne).

## Purpose
The system was used to annotate video data for downstream computer vision tasks such as pose-based behavior classification.

## ML Workflow
Video preprocessing → Object tracking → Pose estimation → Optional pose clustering → Annotation alignment → Machine learning model training → Prediction on new data 

## Why This Project Matters

This system was built to transform unstructured clinical therapy videos into structured datasets suitable for machine learning experimentation.

It bridges raw video processing, pose-based feature engineering, and supervised model training in a reproducible pipeline.

## My contribution included:
- designing and implementing the end-to-end processing pipeline
- integrating pose estimation
- developing feature extraction from skeleton keypoints
- enabling dataset generation for downstream ML model training

## Key Engineering Features

- Config-driven pipeline execution
- Modular step control (skip flags)
- Structured logging with full parameter capture
- Deterministic feature extraction
- Automatic annotation-frame alignment
- Model persistence and evaluation artifacts
  
## Status
Applied ML prototype designed for scalable clinical video processing workflows.

---

## Authentication (HF_TOKEN)

Some tracking models require a Hugging Face access token.

### Runtime Behavior

When running the pipeline:

- If `HF_TOKEN` is already set in your environment, it is used automatically.
- If not, the system securely prompts you to enter it (input is hidden).
- The token is automatically injected into the environment variable:

### Set Token Permanently (Recommended)

```bash
export HF_TOKEN=your_token_here
````

Add this line to:

```
~/.bashrc
```

or

```
~/.zshrc
```

---

## Logging System

The pipeline includes structured logging.

Each processed project produces:

```
processing_log.log
```

### Logged Information

* Full configuration parameters
* Crop settings
* Trim timestamps
* Model parameters
* Executed command lines
* Subprocess STDOUT and STDERR
* Execution time per step
* Pipeline success or failure state

### Example Log Snippet

```
PIPELINE STARTED
CONFIGURATION PARAMETERS
DEVICE=cuda
TEXT_PROMPT=person
CHUNK_SIZE=300
IOU_THRESHOLD=0.15
MAX_OBJECTS=3
START_TRIM_SEC=260
END_TRIM_SEC=900
CROP=(40,40) -> (1240,700)

START STEP: SAM3 tracking
END STEP: SAM3 tracking (142.38 sec)

PIPELINE FINISHED SUCCESSFULLY
```

This ensures reproducibility and simplifies debugging.

---

## Pipeline Architecture

```
Raw Video + Annotations
        │
        ▼
Step 1: Video Processing
        │
        ▼
Step 2: Pose Extraction
        │
        ├── (Optional) Step 3: Pose Clustering
        │
        ▼
Step 4: Annotation Alignment
        │
        ▼
Step 5: Model Training
        │
        ▼
Step 6: Prediction
```

---

## Project Structure

```
video_annotation_system/
│
├── src/
│   ├── video_processing/
│   ├── pose/
│   ├── annotations/
│   ├── models/
│   ├── configs/
│   │   └── config.yaml
│   └── pipeline/
│
├── data/
│   ├── raw/
│   ├── processed/
│   └── annotations/
│
├── models/
├── outputs/
├── requirements.txt
└── README.md
```

---

## Setup

### 1. Create and Activate Virtual Environment (Using uv)

From the project root:

```bash
uv venv .venv
```

Activate on macOS / Linux:

```bash
source .venv/bin/activate
```

Activate on Windows (PowerShell):

```powershell
.venv\Scripts\Activate.ps1
```

### 2. Install Dependencies

Install `psifx` first (local package):

```bash
uv pip install -e /path/to/psifx
```

Then install project requirements:

```bash
uv pip install -r requirements.txt
```

---

## Usage

### Run Full Pipeline

```bash
python src/pipeline/full_pipeline.py
```

### Run Individual Steps

```bash
# Step 1: Video processing
python src/video_processing/processing.py

# Step 2: Pose extraction
python src/pose/extractor.py

# Step 3: Pose clustering (optional)
python src/pose/clustering.py

# Step 4: Annotation alignment
python src/annotations/generator.py

# Step 5: Model training
python src/models/train.py

# Step 6: Prediction
python src/models/predict.py
```

Each script reads configuration from:

```
configs/config.yaml
```

---

## Pipeline Stages

### Step 1 — Video Processing

**Inputs**

* Raw video (e.g., `camera_a.mkv`)
* Annotation timestamps

**Operations**

* Automatic trim computation from annotations
* Object tracking (SAM-based)
* Pose estimation (MediaPipe)
* Overlay visualization

**Outputs**

```
processed_video.mp4
MaskDir/
PosesDir/
Visualizations/
processing_log.log
```

Controlled by:

```
flags.skip_video_processing
```

---

### Step 2 — Pose Extraction

Extracts:

* Frame-level keypoints
* Normalized coordinates
* Movement features
* Temporal derivatives

Output:

```
processed_data.csv
```

Controlled by:

```
flags.skip_pose_extraction
```

---

### Step 3 — Pose Clustering (Optional)

Computes:

* Movement speed
* Distance metrics
* Interaction patterns
* Cluster assignments

Controlled by:

```
flags.skip_pose_clustering
```

---

### Step 4 — Annotation Alignment

* Maps annotation timestamps to frames
* Automatically trims pose data
* Generates labeled dataset

Output:

```
labeled_features.csv
```

---

### Step 5 — Model Training

Supported models:

* LightGBM
* XGBoost
* HistGradientBoosting

Outputs:

```
model.joblib
feature_names.json
model_metrics.json
feature_importance.png
```

Training is skipped automatically if the model file already exists.

---

### Step 6 — Prediction

#### Single Project

Set in `config.yaml`:

```yaml
predict:
  data_path: path/to/project
```

#### Batch Mode

Leave `data_path` empty.

Outputs:

```
predictions.csv
confusion_matrix.png
```

---

## Configuration System

All behavior is controlled via:

```
configs/config.yaml
```

This file includes:

* Paths
* Skip flags
* Model hyperparameters
* Prediction settings
* Auto-trim behavior

Optional per-project configuration:

```
data/raw/<project>/project_config.yaml
```

This file stores computed trim timestamps and metadata.

---

## Reproducibility Features

* Structured logging
* Explicit parameter logging at startup
* Environment variable capture
* Deterministic feature generation
* Config-driven execution
* Step timing metrics

---

## Example Output

```
data/processed/project_01/
├── processed_video.mp4
├── MaskDir/
├── PosesDir/
├── processed_data.csv
├── labeled_features.csv
└── processing_log.log

outputs/
├── model_xgboost.joblib
├── model_metrics.json
├── feature_importance.png
└── model_training_summary.txt
```

---

## Automated Tests

```bash
pytest tests/
```

---

## Requirements

* Python 3.10+
* torch
* mediapipe
* scikit-learn
* lightgbm
* xgboost
* opencv-python
* pyyaml
* matplotlib
* seaborn

See `requirements.txt` for the full dependency list.

---

## Recent Updates

* Interactive `HF_TOKEN` prompt
* Secure token handling
* Structured logging system
* Execution time tracking per step
* Full configuration logging at startup
* Subprocess STDOUT and STDERR capture
* Improved reproducibility

