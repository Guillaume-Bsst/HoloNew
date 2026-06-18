from pathlib import Path

import pytest

from HoloNew.src.data_loaders import facade


def _touch(p: Path):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(b"")


def test_resolve_omomo_by_name(tmp_path, monkeypatch):
    omomo = tmp_path / "OMOMO"
    omomo_new = tmp_path / "OMOMO_new"
    smplh = tmp_path / "smplh"
    _touch(omomo_new / "sub3_largebox_003.pt")
    _touch(omomo / "data" / "train_diffusion_manip_seq_joints24.p")
    _touch(omomo / "data" / "captured_objects" / "largebox_cleaned_simplified.obj")
    smplh.mkdir()
    monkeypatch.setenv("WBT_OMOMO_NEW_DIR", str(omomo_new))
    monkeypatch.setenv("WBT_OMOMO_DIR", str(omomo))
    monkeypatch.setenv("WBT_SMPLH_DIR", str(smplh))

    model, motion, obj, smpl_model_dir = facade.resolve_paths_by_name("omomo", "sub3_largebox_003")
    assert motion == omomo_new / "sub3_largebox_003.pt"
    assert model == omomo / "data" / "train_diffusion_manip_seq_joints24.p"
    assert obj == omomo / "data" / "captured_objects" / "largebox_cleaned_simplified.obj"
    assert smpl_model_dir == smplh


def test_resolve_hodome_by_name(tmp_path, monkeypatch):
    root = tmp_path / "HODome"
    models = tmp_path / "models"
    _touch(root / "smplx" / "subject01_baseball.npz")
    _touch(root / "object" / "subject01_baseball.npz")
    (models / "smplx").mkdir(parents=True)
    monkeypatch.setenv("WBT_HODOME_DIR", str(root))
    monkeypatch.setenv("WBT_SMPLX_DIR", str(models))

    model, motion, obj, smpl_model_dir = facade.resolve_paths_by_name("hodome", "subject01_baseball")
    assert motion == root / "smplx" / "subject01_baseball.npz"
    assert obj == root / "object" / "subject01_baseball.npz"
    assert model == models / "smplx"


def test_resolve_missing_root_raises(monkeypatch):
    monkeypatch.delenv("WBT_HODOME_DIR", raising=False)
    with pytest.raises(ValueError, match="WBT_HODOME_DIR"):
        facade.resolve_paths_by_name("hodome", "subject01_baseball")


def test_normalize_uses_motion_name(tmp_path, monkeypatch):
    from HoloNew.examples.robot_retarget import RetargetingConfig
    omomo = tmp_path / "OMOMO"; omomo_new = tmp_path / "OMOMO_new"; smplh = tmp_path / "smplh"
    _touch(omomo_new / "sub3_largebox_003.pt")
    _touch(omomo / "data" / "train_diffusion_manip_seq_joints24.p")
    smplh.mkdir()
    monkeypatch.setenv("WBT_OMOMO_NEW_DIR", str(omomo_new))
    monkeypatch.setenv("WBT_OMOMO_DIR", str(omomo))
    monkeypatch.setenv("WBT_SMPLH_DIR", str(smplh))

    cfg = RetargetingConfig(dataset="omomo", task_type="robot_only", motion_name="sub3_largebox_003")
    facade.normalize_dataset_cfg(cfg)
    assert cfg.data_format == "smplh"
    assert cfg.task_name == "sub3_largebox_003"
    assert cfg.data_path == omomo_new
