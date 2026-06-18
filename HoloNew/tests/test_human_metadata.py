import joblib
import numpy as np
import pytest

from HoloNew.src.test_socp.correspondence.human_metadata import (
    load_human_metadata,
    load_object_scale,
)


def _write_p(path, entries):
    """entries: list of (seq_name, betas, gender) -> OMOMO-style {idx: {...}} .p file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {i: {"seq_name": s, "betas": b, "gender": g} for i, (s, b, g) in enumerate(entries)}
    joblib.dump(data, str(path))


def _write_p_scale(path, entries):
    """entries: list of (seq_name, obj_scale_array) -> OMOMO-style .p file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {i: {"seq_name": s, "obj_scale": sc} for i, (s, sc) in enumerate(entries)}
    joblib.dump(data, str(path))


def test_loads_object_scale_mean_over_frames(tmp_path):
    sc = np.array([0.34, 0.36], dtype=np.float32)   # near-constant per-frame scale
    _write_p_scale(tmp_path / "data" / "train_diffusion_manip_seq_joints24.p",
                   [("sub3_largebox_003", sc)])
    out = load_object_scale(tmp_path, "sub3_largebox_003")
    assert isinstance(out, float)
    assert out == pytest.approx(0.35, abs=1e-6)


def test_object_scale_missing_returns_none(tmp_path):
    # sequence present for betas but without an obj_scale key -> None (native size)
    _write_p(tmp_path / "data" / "train_diffusion_manip_seq_joints24.p",
             [("seq", np.zeros(16, np.float32), "male")])
    assert load_object_scale(tmp_path, "seq") is None
    assert load_object_scale(tmp_path, "absent") is None


def test_loads_betas_and_gender_from_train(tmp_path):
    betas = np.arange(16, dtype=np.float32)
    _write_p(tmp_path / "data" / "train_diffusion_manip_seq_joints24.p",
             [("sub3_largebox_003", betas[None], "male")])

    out_betas, gender = load_human_metadata(tmp_path, "sub3_largebox_003")
    assert gender == "male"
    assert out_betas.shape == (16,)              # (1, 16) flattened to (16,)
    np.testing.assert_allclose(out_betas, betas)


def test_falls_back_to_test_file(tmp_path):
    betas = np.ones(16, dtype=np.float32)
    _write_p(tmp_path / "data" / "train_diffusion_manip_seq_joints24.p",
             [("other_seq_000", np.zeros(16, np.float32), "female")])
    _write_p(tmp_path / "data" / "test_diffusion_manip_seq_joints24.p",
             [("sub3_largebox_003", betas, "neutral")])

    out_betas, gender = load_human_metadata(tmp_path, "sub3_largebox_003")
    assert gender == "neutral"
    np.testing.assert_allclose(out_betas, betas)


def test_train_takes_precedence_over_test(tmp_path):
    _write_p(tmp_path / "data" / "train_diffusion_manip_seq_joints24.p",
             [("seq", np.full(16, 2.0, np.float32), "male")])
    _write_p(tmp_path / "data" / "test_diffusion_manip_seq_joints24.p",
             [("seq", np.full(16, 9.0, np.float32), "female")])

    out_betas, gender = load_human_metadata(tmp_path, "seq")
    assert gender == "male"
    np.testing.assert_allclose(out_betas, np.full(16, 2.0, np.float32))


def test_missing_sequence_returns_neutral_default(tmp_path):
    _write_p(tmp_path / "data" / "train_diffusion_manip_seq_joints24.p",
             [("some_other_seq", np.zeros(16, np.float32), "male")])

    out_betas, gender = load_human_metadata(tmp_path, "not_present")
    assert out_betas is None
    assert gender == "neutral"


def test_missing_files_returns_neutral_default(tmp_path):
    out_betas, gender = load_human_metadata(tmp_path, "anything")
    assert out_betas is None
    assert gender == "neutral"
