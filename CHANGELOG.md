# Changelog

## 0.2.0 — 2026-10-28

Complete overhaul. **Results produced with 0.1.0 should not be reused** —
several bugs changed the features and inflated the reported scores.

### Fixed — correctness of results
- **Keypoint names.** psifx's MediaPipe tool writes the 33 MediaPipe Pose
  landmarks, but they were named with the OpenPose BODY_25 layout, so almost
  every joint was wrong (e.g. "neck" was the left inner eye, "mid_hip" the
  right ear, "r_ankle" the left shoulder) and landmarks 26–32 were dropped.
  Extraction now uses the MediaPipe layout and derives neck / mid-hip;
  legacy tables are detected and re-mapped.
- **Low-confidence filtering** was computed but never applied; keypoints
  below the visibility threshold are now `NaN`.
- **Normalisation.** `mid_hip_x/y` were overwritten while looping over the
  columns, so every joint after `mid_hip` was normalised relative to 0.
- **Label leakage.** Velocities/accelerations were computed after grouping
  frames *by annotation label* (only possible when the label is known, and
  mixing persons and sessions). They are now computed per continuous track
  segment (session + person, split on time gaps) — identically at training
  and prediction time.
- **Time-aware split** used a time mask from the unsorted table against a
  re-sorted feature matrix, i.e. it was effectively random. Replaced by
  leave-sessions-out (default with ≥ 3 sessions) or a per-session time split
  with a purge gap. XGBoost CV and early stopping use the same grouping.
- **Feature selection** (low-correlation filter never ran; redundancy and
  imputation were fitted on all data incl. test) — now fitted on the training
  split only; the meaningless "correlation with an integer-encoded class"
  filter was removed.
- **Label decoding.** The LabelEncoder was not saved and `label_map` in the
  config was unrelated to it, so predicted class names could be wrong. The
  model bundle now stores class names.
- **Annotation alignment** never matched: tracks were labelled `Person1…`
  (or NaN because `id_to_label` keys `1,2` never matched `id_1`), annotations
  `Child/Therapist`. Explicit `roles` map + prefix map (longest prefix wins);
  non-posture codes (e.g. vocalisations) are removed *before* overlap
  resolution so they no longer hide posture codes; trim offsets come from the
  same per-session settings that trimmed the video.

### Fixed — pipeline crashes
- `combined_df` undefined when the combined CSV already existed.
- Config keys read by the code (`output_base_dir`, `base_data_dir`) did not
  exist in `config.yaml`.
- `run_pose_clustering` called with the wrong signature; its plotting
  methods did not exist.
- Extraction output (`PosesDir/processed_data.csv`) and the orchestrator's
  input (`processed_data__*.parquet`) did not match; hard-coded
  `/home/liubov/...` path in the CSV→parquet script.
- Standalone entry points looked for `config.yaml` in the wrong folder.
- Prediction ran *before* training, with the previous model.

### Added
- Installable package `vas` with a CLI (`vas run|video|extract|cluster|align|train|predict|show-config`).
- Per-session `project_config.yaml` deep-merged over the global config
  (track-id mapping, trim, crop, annotation file).
- New features: limb velocities in the body frame, rolling speed statistics,
  interpersonal distance/direction, role one-hot.
- Macro-F1 / balanced accuracy, per-class report, confusion matrix; model
  selection by macro-F1 by default.
- Prediction outputs: per-class probabilities, optional temporal smoothing,
  `predicted_segments.csv` in original-video time, evaluation against
  annotations when available.
- Synthetic data generator in the real psifx/WAKEE formats; 43 tests;
  GitHub Actions CI (ruff + pytest on Python 3.10 and 3.12); pre-commit with
  nbstripout.
- Logging via `logging` with a log file per session and for training.
- fps can be probed with ffprobe (`pose_extraction.fps: null`).

### Changed
- Pose archives are streamed from the `.tar.gz` files (no extraction to
  disk); feature engineering is vectorised (no per-row `apply`).
- Intermediate tables are Parquet; results live in `outputs/<session>/`.
- `configs/config.yaml` replaces `src/config/config.yaml`.
- Dependencies declared in `pyproject.toml`; unused `torch`, `torchvision`,
  `optuna`, `shap` removed; `pyarrow` added.

### Removed from the repository
- Data, model artefacts, generated plots, the poster PDF and `__pycache__`
  (now git-ignored). Notebook outputs were stripped.
