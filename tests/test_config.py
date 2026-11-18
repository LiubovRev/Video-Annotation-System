import pytest
import yaml

from vas.config import (
    DEFAULT_CONFIG_PATH,
    REQUIRED_SECTIONS,
    ConfigError,
    deep_merge,
    load_config,
    project_config,
    resolve_paths,
)


def test_default_config_is_valid():
    cfg = load_config()
    for section in REQUIRED_SECTIONS:
        assert section in cfg
    assert all(isinstance(v, bool) for v in cfg["flags"].values())


def test_paths_resolve_relative_to_config_file(tmp_path):
    cfg = yaml.safe_load(DEFAULT_CONFIG_PATH.read_text())
    (tmp_path / "configs").mkdir()
    p = tmp_path / "configs" / "c.yaml"
    p.write_text(yaml.safe_dump(cfg))
    paths = resolve_paths(load_config(p))
    assert paths.root == tmp_path
    assert paths.raw == tmp_path / "data" / "raw"
    assert paths.models == tmp_path / "models"


def test_non_boolean_flag_rejected(tmp_path):
    cfg = yaml.safe_load(DEFAULT_CONFIG_PATH.read_text())
    cfg["flags"]["skip_training"] = "yes"
    p = tmp_path / "c.yaml"
    p.write_text(yaml.safe_dump(cfg))
    with pytest.raises(ConfigError):
        load_config(p)


def test_legacy_keys_are_migrated(tmp_path):
    cfg = yaml.safe_load(DEFAULT_CONFIG_PATH.read_text())
    cfg["directories"].pop("output_dir")
    cfg["directories"]["output_base_dir"] = "legacy_out"
    cfg["flags"].pop("force_recompute")
    cfg["flags"]["force_recombine"] = True
    p = tmp_path / "c.yaml"
    p.write_text(yaml.safe_dump(cfg))
    loaded = load_config(p)
    assert loaded["directories"]["output_dir"] == "legacy_out"
    assert loaded["flags"]["force_recompute"] is True


def test_deep_merge_does_not_mutate():
    base = {"a": {"b": 1, "c": 2}}
    out = deep_merge(base, {"a": {"b": 5}})
    assert out == {"a": {"b": 5, "c": 2}} and base == {"a": {"b": 1, "c": 2}}


def test_project_override(tmp_path):
    cfg = load_config()
    (tmp_path / "project_config.yaml").write_text("video_processing:\n  start_trim_sec: 7\n")
    merged = project_config(cfg, tmp_path)
    assert merged["video_processing"]["start_trim_sec"] == 7
    assert merged["video_processing"]["x_max"] == cfg["video_processing"]["x_max"]
