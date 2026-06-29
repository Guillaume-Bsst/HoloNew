from pathlib import Path

from src.prepare.load.datasets.hoim3 import _resolve_assets


def _smplx_dir(tmp_path: Path) -> Path:
    # conventional depth: .../models/<release>/models/smplx  (so parents[2] == .../models)
    d = tmp_path / "models" / "release" / "models" / "smplx"
    d.mkdir(parents=True)
    (d / "SMPLX_NEUTRAL.npz").write_bytes(b"")
    return d


def test_resolve_assets_convention(tmp_path):
    smplx_dir = _smplx_dir(tmp_path)
    models_root = smplx_dir.parents[2]
    (models_root / "smplh" / "neutral").mkdir(parents=True)
    (models_root / "smplh" / "neutral" / "model.npz").write_bytes(b"")
    (models_root / "model_transfer").mkdir(parents=True)
    (models_root / "model_transfer" / "smpl2smplx_deftrafo_setup.pkl").write_bytes(b"")
    _, sh, dt = _resolve_assets(smplx_dir, "neutral")
    assert sh == models_root / "smplh" / "neutral" / "model.npz"
    assert dt == models_root / "model_transfer" / "smpl2smplx_deftrafo_setup.pkl"


def test_resolve_assets_overrides_win(tmp_path):
    smplx_dir = _smplx_dir(tmp_path)
    smplh = tmp_path / "custom_smplh"
    (smplh / "neutral").mkdir(parents=True)
    (smplh / "neutral" / "model.npz").write_bytes(b"")
    deftrafo = tmp_path / "custom" / "dt.pkl"
    deftrafo.parent.mkdir(parents=True)
    deftrafo.write_bytes(b"")
    _, sh, dt = _resolve_assets(smplx_dir, "neutral", smplh_dir=smplh, deftrafo_pkl=deftrafo)
    assert sh == smplh / "neutral" / "model.npz"
    assert dt == deftrafo
