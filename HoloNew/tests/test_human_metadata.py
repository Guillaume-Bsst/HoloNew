import joblib
import numpy as np

from HoloNew.src.gmr_socp_v2.correspondence.human_metadata import load_human_metadata


def _write_p(path, entries):
    """entries: list of (seq_name, betas, gender) -> OMOMO-style {idx: {...}} .p file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {i: {"seq_name": s, "betas": b, "gender": g} for i, (s, b, g) in enumerate(entries)}
    joblib.dump(data, str(path))


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
