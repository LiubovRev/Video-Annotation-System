import pandas as pd

from vas.annotations import (
    align_session,
    find_annotation_file,
    map_annotations_to_frames,
    parse_annotations,
    prepare_annotations,
    role_for_label,
)
from vas.config import load_config
from vas.synthetic import wakee_line


def test_parse_wakee(tmp_path):
    p = tmp_path / "a.txt"
    p.write_text(wakee_line("c14_CP", 165.741, 1118.787, "CST") + "\n\ngarbage line\n")
    df = parse_annotations(p)
    assert df.iloc[0].to_dict() == {"tier": "c14_CP", "start_time": 165.741, "end_time": 1118.787, "label": "CST"}


def test_role_longest_prefix():
    m = {"C": "Child", "T": "Therapist", "T2": "Therapist2"}
    assert role_for_label("CST", m) == "Child"
    assert role_for_label("T2RE", m) == "Therapist2"
    assert role_for_label("GO", m) is None


def _frames(times, role="Child"):
    return pd.DataFrame({"video_time_s": times, "role": role})


def test_shortest_overlap_wins_and_roles_are_respected():
    ann = pd.DataFrame({"start_time": [0, 2], "end_time": [10, 4], "label": ["CST", "CCR"],
                        "role": ["Child", "Child"], "duration": [10, 2]})
    out = map_annotations_to_frames(pd.concat([_frames([1, 3, 5]), _frames([3], "Therapist")]), ann)
    labels = out["annotation_label"].tolist()
    assert labels[:3] == ["CST", "CCR", "CST"] and pd.isna(labels[3])
    assert out["n_overlapping_labels"].tolist() == [1, 2, 1, 0]


def test_first_strategy_keeps_file_order():
    ann = pd.DataFrame({"start_time": [0, 2], "end_time": [10, 4], "label": ["CST", "CCR"],
                        "role": ["Child", "Child"], "duration": [10, 2]})
    out = map_annotations_to_frames(_frames([3]), ann, strategy="first")
    assert out["annotation_label"].tolist() == ["CST"]


def test_non_posture_codes_are_excluded_before_matching():
    cfg = load_config()
    ann = pd.DataFrame({"tier": "x", "start_time": [0, 0], "end_time": [5, 5], "label": ["CS", "CST"]})
    out = prepare_annotations(ann, cfg)
    assert out["label"].tolist() == ["CST"]


def test_trim_shift(tmp_path):
    cfg = load_config()
    cfg["video_processing"]["start_trim_sec"] = 100
    p = tmp_path / "a.txt"
    p.write_text(wakee_line("c", 100, 101, "CST") + "\n")
    frames = pd.DataFrame({"person_label": ["Patient1"] * 3, "time_s": [0.5, 1.5, 0.5]})
    frames.loc[2, "person_label"] = "Therapist1"
    out = align_session(frames, p, cfg)
    assert len(out) == 1 and out.loc[0, "video_time_s"] == 100.5 and out.loc[0, "role"] == "Child"


def test_annotation_discovery_handles_brackets(demo_cfg, demo_root):
    session = demo_root / "data" / "raw" / "1-1-2024_#2_SYNTHETIC_[00]"
    found = find_annotation_file(session, demo_cfg)
    assert found is not None and found.name.startswith("1-1-2024_#2_SYNTHETIC_(00)")
