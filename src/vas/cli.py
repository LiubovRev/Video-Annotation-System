"""Command-line interface: ``vas <command> [options]`` (or ``python -m vas``)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .config import ConfigError, deep_merge, discover_projects, load_config, resolve_paths
from .utils import read_table, setup_logging


def _projects(cfg: dict, names: list[str] | None, project_dir: str | None) -> list[Path]:
    if project_dir:
        return [Path(project_dir).resolve()]
    projects = discover_projects(cfg)
    if names:
        wanted = set(names)
        projects = [p for p in projects if p.name in wanted]
        missing = wanted - {p.name for p in projects}
        if missing:
            raise ConfigError(f"Unknown session(s): {sorted(missing)}")
    return projects


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="vas", description="Clinical video annotation pipeline")
    p.add_argument("-c", "--config", help="config YAML (default: configs/config.yaml)")
    p.add_argument("--log-level", default=None, help="DEBUG, INFO, WARNING …")
    sub = p.add_subparsers(dest="command", required=True)

    def session_args(sp, force=True):
        sp.add_argument("-s", "--session", action="append", help="session folder name (repeatable)")
        sp.add_argument("--project-dir", help="path to a single session folder")
        if force:
            sp.add_argument("-f", "--force", action="store_true", help="recompute even if outputs exist")

    run = sub.add_parser("run", help="run the full pipeline")
    run.add_argument("-s", "--session", action="append", help="restrict to these sessions")
    run.add_argument("--force", action="store_true", help="set flags.force_recompute")
    run.add_argument("--retrain", action="store_true", help="set flags.force_retrain")

    session_args(sub.add_parser("video", help="step 1: ffmpeg + psifx tracking/pose"))
    session_args(sub.add_parser("extract", help="step 2: pose archives → frame table"))
    session_args(sub.add_parser("cluster", help="step 3: interaction analysis + pose clusters"))
    session_args(sub.add_parser("align", help="step 4: attach annotation codes to frames"))

    tr = sub.add_parser("train", help="step 5: train on labeled sessions")
    tr.add_argument("--input", help="labeled table (csv/parquet); default: all sessions' labeled_features")
    tr.add_argument("-s", "--session", action="append", help="restrict to these sessions")

    pr = sub.add_parser("predict", help="step 6: predict codes with the trained bundle")
    session_args(pr)
    pr.add_argument("--input", help="predict for an arbitrary pose table instead of sessions")
    pr.add_argument("--output-dir", help="where to write predictions for --input")
    pr.add_argument("--bundle", help="model bundle path (default from config)")

    sub.add_parser("show-config", help="print the resolved configuration")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        cfg = load_config(args.config)
    except ConfigError as e:
        print(f"Config error: {e}", file=sys.stderr)
        return 2
    log = setup_logging(args.log_level or cfg.get("logging", {}).get("level", "INFO"))

    try:
        if args.command == "show-config":
            out = dict(cfg)
            out["_resolved_paths"] = {k: str(v) for k, v in vars(resolve_paths(cfg)).items()}
            print(json.dumps(out, indent=2, default=str))
            return 0

        if args.command == "run":
            from .pipeline import run_pipeline

            overrides = {"flags": {}}
            if args.force:
                overrides["flags"]["force_recompute"] = True
            if args.retrain:
                overrides["flags"]["force_retrain"] = True
            summary = run_pipeline(deep_merge(cfg, overrides), args.session)
            failed = [s for s, st in summary["sessions"].items() if "failed" in st.values() or "error" in st]
            return 1 if failed else 0

        if args.command == "train":
            import pandas as pd

            from .pipeline import combine_labeled
            from .train import train_model

            if args.input:
                df = read_table(Path(args.input))
                if "session" not in df.columns:
                    df["session"] = df.get("project_name", Path(args.input).stem)
            else:
                names = args.session or [p.name for p in discover_projects(cfg)]
                df = combine_labeled(cfg, names)
            if isinstance(df, pd.DataFrame) and df.empty:
                log.error("No labeled data found. Run `vas align` first.")
                return 1
            res = train_model(df, cfg)
            log.info("Done: %s %s=%.4f", res["model_name"], res["metric"], res["score"])
            return 0

        if args.command == "predict" and args.input:
            from .predict import predict_file

            out_dir = Path(args.output_dir or resolve_paths(cfg).outputs / "predictions")
            out_dir.mkdir(parents=True, exist_ok=True)
            path = predict_file(Path(args.input), cfg, out_dir, Path(args.bundle) if args.bundle else None)
            log.info("Predictions written to %s", path)
            return 0

        steps = {
            "video": ("video_processing", "run_video_processing"),
            "extract": ("pose_extraction", "run_pose_extraction"),
            "cluster": ("clustering", "run_pose_clustering"),
            "align": ("annotations", "run_annotation_alignment"),
            "predict": ("predict", "predict_session"),
        }
        module, func = steps[args.command]
        import importlib

        fn = getattr(importlib.import_module(f"vas.{module}"), func)
        kwargs = {}
        if args.command == "predict" and args.bundle:
            from .predict import load_bundle

            kwargs["bundle"] = load_bundle(Path(args.bundle))
        failures = 0
        for project in _projects(cfg, args.session, args.project_dir):
            log.info("── %s: %s", args.command, project.name)
            result = fn(project, cfg, force=args.force, **kwargs)
            failures += result in (None, False)
        return 1 if failures else 0
    except (ConfigError, FileNotFoundError, ValueError) as e:
        log.error("%s", e)
        return 1


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
