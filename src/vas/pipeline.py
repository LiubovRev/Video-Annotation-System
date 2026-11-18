"""End-to-end orchestration across all recording sessions.

Per session: video processing → pose extraction → [clustering] →
annotation alignment. Then: combine labeled sessions → train → predict
for every session with the freshly trained (or existing) model.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from .annotations import run_annotation_alignment
from .clustering import run_pose_clustering
from .config import discover_projects, model_bundle_path, resolve_paths
from .pose_extraction import run_pose_extraction
from .predict import load_bundle, predict_session
from .train import train_model
from .utils import log_to_file, read_table, write_table
from .video_processing import run_video_processing

logger = logging.getLogger(__name__)


@dataclass
class SessionResult:
    name: str
    status: dict = field(default_factory=dict)

    def mark(self, step: str, ok: bool | str) -> None:
        self.status[step] = ok if isinstance(ok, str) else ("ok" if ok else "failed")


def process_session(project_dir: Path, cfg: dict) -> SessionResult:
    flags = cfg["flags"]
    force = flags.get("force_recompute", False)
    res = SessionResult(project_dir.name)
    t0 = time.time()

    if flags.get("skip_video_processing"):
        res.mark("video", "skipped")
    else:
        ok = run_video_processing(project_dir, cfg, force=force)
        res.mark("video", ok)
        if not ok:
            return res

    if flags.get("skip_pose_extraction"):
        res.mark("poses", "skipped")
    else:
        res.mark("poses", run_pose_extraction(project_dir, cfg, force=force) is not None)
        if res.status["poses"] != "ok":
            return res

    if flags.get("skip_pose_clustering", True):
        res.mark("clustering", "skipped")
    else:
        try:
            res.mark("clustering", run_pose_clustering(project_dir, cfg, force=force) is not None)
        except Exception as e:  # exploratory step: never block the pipeline
            logger.exception("Clustering failed: %s", e)
            res.mark("clustering", False)

    if flags.get("skip_annotation_processing"):
        res.mark("alignment", "skipped")
    else:
        out = run_annotation_alignment(project_dir, cfg, force=force)
        res.mark("alignment", "ok" if out is not None else "no labels")
    logger.info("Session %s finished in %.1fs", project_dir.name, time.time() - t0)
    return res


def combine_labeled(cfg: dict, sessions: list[str]) -> pd.DataFrame:
    outputs = resolve_paths(cfg).outputs
    parts = []
    for s in sessions:
        p = outputs / s / "labeled_features.parquet"
        if p.exists():
            df = read_table(p)
            df["session"] = s
            parts.append(df)
    if not parts:
        return pd.DataFrame()
    combined = pd.concat(parts, ignore_index=True)
    write_table(combined, outputs / "combined_labeled_features.parquet")
    logger.info("Combined %d labeled sessions: %d frames", len(parts), len(combined))
    return combined


def run_pipeline(cfg: dict, sessions: list[str] | None = None) -> dict:
    paths = resolve_paths(cfg)
    paths.outputs.mkdir(parents=True, exist_ok=True)
    projects = discover_projects(cfg)
    if sessions:
        projects = [p for p in projects if p.name in set(sessions)]
    logger.info("Found %d session(s) in %s", len(projects), paths.raw)

    results = []
    for project in projects:
        logger.info("=" * 70)
        logger.info("Session: %s", project.name)
        with log_to_file(paths.outputs / project.name / "processing_log.log"):
            try:
                results.append(process_session(project, cfg))
            except Exception as e:
                logger.exception("Unhandled error in %s: %s", project.name, e)
                r = SessionResult(project.name)
                r.mark("error", str(e))
                results.append(r)

    flags = cfg["flags"]
    summary: dict = {"sessions": {r.name: r.status for r in results}}
    bundle_path = model_bundle_path(cfg)
    trained_now = False

    if flags.get("skip_training"):
        summary["training"] = "skipped"
    elif bundle_path.exists() and not flags.get("force_retrain"):
        logger.info("Model bundle exists (%s) — skipping training (set flags.force_retrain).", bundle_path)
        summary["training"] = "cached"
    else:
        combined = combine_labeled(cfg, [p.name for p in projects])
        if combined.empty:
            logger.warning("No labeled sessions — cannot train.")
            summary["training"] = "no labeled data"
        else:
            with log_to_file(paths.outputs / "training" / "training_log.log"):
                try:
                    t = train_model(combined, cfg)
                    summary["training"] = {"model": t["model_name"], t["metric"]: round(t["score"], 4)}
                    trained_now = True
                except Exception as e:
                    logger.exception("Training failed: %s", e)
                    summary["training"] = f"failed: {e}"

    if flags.get("skip_prediction"):
        summary["prediction"] = "skipped"
    elif not bundle_path.exists():
        summary["prediction"] = "no model"
    else:
        bundle = load_bundle(bundle_path)
        force = flags.get("force_recompute", False) or trained_now
        for project in projects:
            if not (paths.outputs / project.name / "processed_data.parquet").exists():
                continue
            try:
                ok = predict_session(project, cfg, bundle=bundle, force=force) is not None
            except Exception as e:
                logger.exception("Prediction failed for %s: %s", project.name, e)
                ok = False
            summary["sessions"].setdefault(project.name, {})["prediction"] = "ok" if ok else "failed"

    logger.info("=" * 70)
    logger.info("PIPELINE SUMMARY")
    for name, st in summary["sessions"].items():
        logger.info("  %-40s %s", name, ", ".join(f"{k}={v}" for k, v in st.items()))
    logger.info("  training: %s", summary.get("training"))
    return summary
