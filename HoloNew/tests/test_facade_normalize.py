from pathlib import Path

import numpy as np

from HoloNew.examples.robot_retarget import RetargetingConfig
from HoloNew.src.data_loaders import facade


def test_normalize_omomo_maps_to_legacy_fields():
    cfg = RetargetingConfig(
        dataset="omomo", task_type="robot_only",
        model_path=Path("/data/OMOMO/data/train_diffusion_manip_seq_joints24.p"),
        motion_path=Path("/data/OMOMO_new/sub3_largebox_003.pt"),
    )
    facade.normalize_dataset_cfg(cfg)
    assert cfg.data_format == "smplh"
    assert cfg.data_path == Path("/data/OMOMO_new")
    assert cfg.task_name == "sub3_largebox_003"
    # OMOMO root (holds data/*.p) is two levels up from the pickle.
    assert cfg.omomo_dir == Path("/data/OMOMO")


def test_normalize_hodome_runs_prep_and_maps(tmp_path, monkeypatch):
    raw = tmp_path / "subject01_baseball.npz"
    raw.write_bytes(b"")  # content irrelevant — prep is monkeypatched
    model = tmp_path / "smplx"

    def fake_prep(npz_path, model_dir):
        assert Path(npz_path) == raw and Path(model_dir) == model
        T = 5
        return {
            "global_joint_positions": np.zeros((T, 22, 3), np.float32),
            "global_joint_orientations": np.tile([1, 0, 0, 0], (T, 22, 1)).astype(np.float32),
            "height": np.float32(1.7),
            "betas": np.zeros((1, 10), np.float32),
            "gender": "neutral",
        }

    monkeypatch.setattr(facade, "prep_hodome_processed", fake_prep)
    monkeypatch.setattr(facade, "_HODOME_CACHE_DIR", tmp_path / "cache")

    cfg = RetargetingConfig(dataset="hodome", task_type="robot_only",
                            model_path=model, motion_path=raw)
    facade.normalize_dataset_cfg(cfg)

    assert cfg.data_format == "smplx"
    assert cfg.task_name == "subject01_baseball"
    processed = cfg.data_path / "subject01_baseball.npz"
    assert processed.exists()
    d = np.load(processed)
    assert d["global_joint_positions"].shape == (5, 22, 3)


def test_normalize_noop_without_dataset():
    cfg = RetargetingConfig()  # dataset is None
    before = (cfg.data_path, cfg.task_name, cfg.data_format)
    facade.normalize_dataset_cfg(cfg)
    assert (cfg.data_path, cfg.task_name, cfg.data_format) == before
