from pathlib import Path

import pytest

from vas.config import load_config
from vas.synthetic import make_demo_project
from vas.utils import setup_logging

FAST_TRAINING = {
    "model_training": {"models": ["LightGBM", "HistGradientBoosting"], "n_iter_search": 2},
    "flags": {"skip_pose_clustering": False},
}


@pytest.fixture(scope="session", autouse=True)
def _logging():
    setup_logging("WARNING")


@pytest.fixture(scope="session")
def demo_root(tmp_path_factory) -> Path:
    root = tmp_path_factory.mktemp("demo")
    make_demo_project(root, n_sessions=3, duration_s=45, config_overrides=FAST_TRAINING)
    return root


@pytest.fixture(scope="session")
def demo_cfg_path(demo_root) -> Path:
    return demo_root / "configs" / "config.yaml"


@pytest.fixture()
def demo_cfg(demo_cfg_path) -> dict:
    return load_config(demo_cfg_path)
