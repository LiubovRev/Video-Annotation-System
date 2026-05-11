"""
Minimal sanity tests for the project configuration file.
Run with: pytest tests/
"""

import yaml
from pathlib import Path

# Resolve config path relative to the project root (one level above this tests/ dir)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = PROJECT_ROOT / "src" / "config" / "config.yaml"


def test_config_exists():
    assert CONFIG_PATH.exists(), f"config.yaml not found at {CONFIG_PATH}"


def test_required_top_level_keys():
    cfg = yaml.safe_load(CONFIG_PATH.read_text())
    required_keys = [
        "directories",
        "flags",
        "video_processing",
        "pose_extraction",
        "annotation_alignment",
        "model",
        "model_training",
    ]
    for key in required_keys:
        assert key in cfg, f"Missing top-level key '{key}' in config.yaml"


def test_flags_are_booleans():
    cfg = yaml.safe_load(CONFIG_PATH.read_text())
    for flag_name, flag_value in cfg["flags"].items():
        assert isinstance(flag_value, bool), (
            f"flags.{flag_name} must be a boolean, got {type(flag_value).__name__}"
        )


def test_model_training_has_required_fields():
    cfg = yaml.safe_load(CONFIG_PATH.read_text())
    mt = cfg["model_training"]
    for field in ["min_samples_per_class", "test_size", "n_iter_search"]:
        assert field in mt, f"Missing model_training.{field}"
