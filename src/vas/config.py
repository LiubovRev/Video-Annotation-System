"""Configuration loading, per-session overrides and path resolution."""

from __future__ import annotations

import copy
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)

PACKAGE_ROOT = Path(__file__).resolve().parent
REPO_ROOT = PACKAGE_ROOT.parent.parent
DEFAULT_CONFIG_PATH = REPO_ROOT / "configs" / "config.yaml"
PROJECT_CONFIG_NAME = "project_config.yaml"

REQUIRED_SECTIONS = (
    "directories",
    "flags",
    "video_processing",
    "pose_extraction",
    "annotation_alignment",
    "features",
    "model_training",
    "model",
)

# Keys from the previous config layout → their new location.
_LEGACY_KEYS = {
    ("directories", "output_base_dir"): ("directories", "output_dir"),
    ("flags", "force_recombine"): ("flags", "force_recompute"),
}


class ConfigError(ValueError):
    """Raised when the configuration is missing or inconsistent."""


def deep_merge(base: dict, override: dict) -> dict:
    """Return a new dict with ``override`` recursively merged into ``base``."""
    out = copy.deepcopy(base)
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def _apply_legacy_keys(cfg: dict) -> dict:
    for (sec, old), (new_sec, new) in _LEGACY_KEYS.items():
        section = cfg.get(sec) or {}
        if old in section:
            logger.warning("Config key '%s.%s' is deprecated; use '%s.%s'.", sec, old, new_sec, new)
            cfg.setdefault(new_sec, {}).setdefault(new, section.pop(old))
    return cfg


def validate(cfg: dict) -> None:
    missing = [s for s in REQUIRED_SECTIONS if s not in cfg]
    if missing:
        raise ConfigError(f"Missing config section(s): {', '.join(missing)}")
    for name, value in (cfg.get("flags") or {}).items():
        if not isinstance(value, bool):
            raise ConfigError(f"flags.{name} must be true/false, got {value!r}")
    mt = cfg["model_training"]
    if not 0 < float(mt.get("test_size", 0.2)) < 1:
        raise ConfigError("model_training.test_size must be in (0, 1)")
    if mt.get("split_strategy", "auto") not in {"auto", "session", "time"}:
        raise ConfigError("model_training.split_strategy must be auto|session|time")
    fmt = cfg["pose_extraction"].get("keypoint_format", "mediapipe")
    if fmt not in {"mediapipe", "body25"}:
        raise ConfigError("pose_extraction.keypoint_format must be mediapipe|body25")


@dataclass(frozen=True)
class Paths:
    root: Path
    raw: Path
    annotations: Path
    outputs: Path
    models: Path


def load_config(path: str | Path | None = None, overrides: dict | None = None) -> dict:
    """Load the YAML config, apply legacy-key migration and validate it.

    The absolute path of the config file is stored under ``_config_path`` so
    relative directories can be resolved later.
    """
    path = Path(path) if path else DEFAULT_CONFIG_PATH
    if not path.exists():
        raise ConfigError(f"Config file not found: {path}")
    with open(path) as f:
        cfg = yaml.safe_load(f) or {}
    cfg = _apply_legacy_keys(cfg)
    if overrides:
        cfg = deep_merge(cfg, overrides)
    validate(cfg)
    cfg["_config_path"] = str(path.resolve())
    return cfg


def resolve_paths(cfg: dict) -> Paths:
    d = cfg.get("directories", {})
    cfg_dir = Path(cfg.get("_config_path", DEFAULT_CONFIG_PATH)).parent
    root = (cfg_dir / d.get("root", "..")).resolve()

    def _p(key: str, default: str) -> Path:
        p = Path(d.get(key) or default)
        return p if p.is_absolute() else root / p

    return Paths(
        root=root,
        raw=_p("raw_video_root", "data/raw"),
        annotations=_p("annotations_dir", "data/annotations"),
        outputs=_p("output_dir", "outputs"),
        models=_p("models_dir", "models"),
    )


def model_bundle_path(cfg: dict) -> Path:
    name = cfg.get("model", {}).get("bundle_file", "model_bundle.joblib")
    p = Path(name)
    return p if p.is_absolute() else resolve_paths(cfg).models / p


def project_config(cfg: dict, project_dir: Path) -> dict:
    """Return ``cfg`` with ``<project_dir>/project_config.yaml`` merged in."""
    override_path = Path(project_dir) / PROJECT_CONFIG_NAME
    if not override_path.exists():
        return cfg
    with open(override_path) as f:
        override = yaml.safe_load(f) or {}
    logger.info("Applying per-session overrides from %s", override_path)
    merged = deep_merge(cfg, override)
    validate(merged)
    return merged


def discover_projects(cfg: dict) -> list[Path]:
    raw = resolve_paths(cfg).raw
    if not raw.exists():
        raise ConfigError(f"Raw data directory not found: {raw}")
    return sorted(p for p in raw.iterdir() if p.is_dir() and not p.name.startswith("."))


def role_of(cfg: dict, person_label: Any) -> str | None:
    roles = cfg.get("roles") or {}
    return roles.get(person_label, person_label if person_label in set(roles.values()) else None)
