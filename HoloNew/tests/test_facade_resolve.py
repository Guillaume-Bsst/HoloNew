from pathlib import Path

import pytest

from HoloNew.src import paths
from HoloNew.src.data_loaders import facade


def _touch(p: Path):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(b"")


def _write_paths(tmp_path, monkeypatch, **kv):
    y = tmp_path / "path.yaml"
    y.write_text("".join(f"{k}: {v}\n" for k, v in kv.items()))
    monkeypatch.setattr(paths, "PATHS_YAML", y)


def test_resolve_omomo_by_name(tmp_path, monkeypatch):
    omomo = tmp_path / "OMOMO"
    omomo_new = tmp_path / "OMOMO_new"
    smplh = tmp_path / "smplh"
    _touch(omomo_new / "sub3_largebox_003.pt")
    _touch(omomo / "data" / "train_diffusion_manip_seq_joints24.p")
    _touch(omomo / "data" / "captured_objects" / "largebox_cleaned_simplified.obj")
    smplh.mkdir()
    _write_paths(tmp_path, monkeypatch, omomo=omomo, omomo_new=omomo_new, smplh_models=smplh)

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
    _write_paths(tmp_path, monkeypatch, hodome=root, smplx_models=models)

    model, motion, obj, smpl_model_dir = facade.resolve_paths_by_name("hodome", "subject01_baseball")
    assert motion == root / "smplx" / "subject01_baseball.npz"
    assert obj == root / "object" / "subject01_baseball.npz"
    assert model == models / "smplx"


def test_resolve_missing_key_raises(tmp_path, monkeypatch):
    _write_paths(tmp_path, monkeypatch, smplx_models=tmp_path / "models")  # no 'hodome'
    with pytest.raises(ValueError, match="hodome"):
        facade.resolve_paths_by_name("hodome", "subject01_baseball")


def test_normalize_uses_motion_name(tmp_path, monkeypatch):
    from HoloNew.examples.robot_retarget import RetargetingConfig
    omomo = tmp_path / "OMOMO"; omomo_new = tmp_path / "OMOMO_new"; smplh = tmp_path / "smplh"
    _touch(omomo_new / "sub3_largebox_003.pt")
    _touch(omomo / "data" / "train_diffusion_manip_seq_joints24.p")
    smplh.mkdir()
    _write_paths(tmp_path, monkeypatch, omomo=omomo, omomo_new=omomo_new, smplh_models=smplh)

    cfg = RetargetingConfig(dataset="omomo", task_type="robot_only", motion_name="sub3_largebox_003")
    facade.normalize_dataset_cfg(cfg)
    assert cfg.data_format == "smplh"
    assert cfg.task_name == "sub3_largebox_003"
    assert cfg.data_path == omomo_new


def test_normalize_dataset_is_case_insensitive(tmp_path, monkeypatch):
    # --dataset OMOMO must resolve like --dataset omomo (registry keys are lowercase).
    from HoloNew.examples.robot_retarget import RetargetingConfig
    omomo = tmp_path / "OMOMO"; omomo_new = tmp_path / "OMOMO_new"; smplh = tmp_path / "smplh"
    _touch(omomo_new / "sub3_largebox_003.pt")
    _touch(omomo / "data" / "train_diffusion_manip_seq_joints24.p")
    smplh.mkdir()
    _write_paths(tmp_path, monkeypatch, omomo=omomo, omomo_new=omomo_new, smplh_models=smplh)

    cfg = RetargetingConfig(dataset="OMOMO", task_type="robot_only", motion_name="sub3_largebox_003")
    facade.normalize_dataset_cfg(cfg)
    assert cfg.dataset == "omomo"
    assert cfg.data_format == "smplh"
    assert cfg.task_name == "sub3_largebox_003"
    assert cfg.data_path == omomo_new


def test_smplx_dataset_forces_robot_only(tmp_path):
    # smplx datasets have no .pt-based object channel in the solve, so object_interaction
    # (the default task_type) is downgraded to robot_only (object stays a viewer overlay).
    from HoloNew.examples.robot_retarget import RetargetingConfig
    motion = tmp_path / "clip.npz"; motion.write_bytes(b"")
    model = tmp_path / "model"; model.mkdir()
    cfg = RetargetingConfig(dataset="sfu", task_type="object_interaction",
                            model_path=model, motion_path=motion)
    facade.normalize_dataset_cfg(cfg)
    assert cfg.data_format == "smplx"
    assert cfg.task_type == "robot_only"


def test_smplh_dataset_keeps_object_interaction(tmp_path):
    # smplh (OMOMO) carries the object in the .pt, so object_interaction is kept.
    from HoloNew.examples.robot_retarget import RetargetingConfig
    motion = tmp_path / "seq.pt"; motion.write_bytes(b"")
    model = tmp_path / "data" / "x.p"; model.parent.mkdir(parents=True); model.write_bytes(b"")
    cfg = RetargetingConfig(dataset="omomo", task_type="object_interaction",
                            model_path=model, motion_path=motion)
    facade.normalize_dataset_cfg(cfg)
    assert cfg.data_format == "smplh"
    assert cfg.task_type == "object_interaction"
