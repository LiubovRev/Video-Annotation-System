import pytest

from vas import video_processing as vp
from vas.config import load_config


def test_ffmpeg_command_trim_crop():
    cmd = vp.build_ffmpeg_cmd("in.mkv", "out.mp4", {"start_trim_sec": 5, "end_trim_sec": 60,
                                                    "x_min": 40, "y_min": 40, "x_max": 1240, "y_max": 700})
    assert cmd[cmd.index("-ss") + 1] == "5.0" and cmd[cmd.index("-to") + 1] == "60"
    assert "crop=1200:660:40:40" in cmd


@pytest.mark.parametrize("params", [{"x_min": 100, "x_max": 50, "y_max": 10},
                                    {"start_trim_sec": 10, "end_trim_sec": 5}])
def test_ffmpeg_invalid_params(params):
    with pytest.raises(ValueError):
        vp.build_ffmpeg_cmd("in", "out", params)


def test_run_video_processing_calls_steps(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(vp, "run_command", lambda cmd, name, env=None: calls.append(name))
    (tmp_path / "camera_a.mkv").write_bytes(b"")
    cfg = load_config()
    assert vp.run_video_processing(tmp_path, cfg)
    assert calls == ["FFmpeg trim + crop", "SAM3 tracking", "Tracking visualization",
                     "Pose inference", "Pose visualization"]


def test_run_video_processing_missing_video(tmp_path):
    assert vp.run_video_processing(tmp_path, load_config()) is False


def test_existing_archives_skip(tmp_path, monkeypatch):
    (tmp_path / "PosesDir").mkdir()
    (tmp_path / "PosesDir" / "1.tar.gz").write_bytes(b"")
    monkeypatch.setattr(vp, "run_command", lambda *a, **k: pytest.fail("should not run"))
    assert vp.run_video_processing(tmp_path, load_config())


def test_failed_step_returns_false(tmp_path, monkeypatch):
    def boom(cmd, name, env=None):
        raise vp.StepError("nope")
    monkeypatch.setattr(vp, "run_command", boom)
    (tmp_path / "camera_a.mkv").write_bytes(b"")
    assert vp.run_video_processing(tmp_path, load_config()) is False
