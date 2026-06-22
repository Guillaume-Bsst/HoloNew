import numpy as np
import joblib
import trimesh
import pytest
from pathlib import Path
from HoloNew.src.data_loaders.omomo import (
    resolve_omomo_object_mesh, _omomo_obj_name,
)


def test_obj_name_second_token():
    assert _omomo_obj_name("sub3_largebox_003") == "largebox"
    assert _omomo_obj_name("solo") == "solo"


def test_bundled_found_regardless_of_cwd(tmp_path, monkeypatch):
    # largebox is bundled in the package; resolver must find it from any cwd.
    monkeypatch.chdir(tmp_path)
    p = resolve_omomo_object_mesh("sub3_largebox_003")
    assert p is not None
    assert p.name == "largebox.obj"
    assert p.is_absolute()


def test_captured_fallback_recenters_and_scales(tmp_path):
    # Fake OMOMO release: captured unit mesh (off-origin) + a .p carrying obj_scale.
    omomo = tmp_path / "OMOMO"
    cap = omomo / "data" / "captured_objects"
    cap.mkdir(parents=True)
    box = trimesh.creation.box(extents=(1.0, 1.0, 1.0))
    box.apply_translation([5.0, 0.0, 0.0])  # off-origin
    box.export(cap / "widget_cleaned_simplified.obj")
    (omomo / "data").mkdir(exist_ok=True)
    joblib.dump({0: {"seq_name": "sub9_widget_001", "obj_scale": np.array([2.0, 2.0])}},
                omomo / "data" / "train_diffusion_manip_seq_joints24.p")

    out = resolve_omomo_object_mesh("sub9_widget_001", omomo_dir=omomo,
                                    cache_dir=tmp_path / "cache")
    assert out is not None and out.exists()
    m = trimesh.load(str(out), force="mesh", process=False)
    v = np.asarray(m.vertices)
    # Recentred on its own centroid (~origin) and scaled ×2 -> extents 2.0.
    assert np.allclose(v.mean(0), 0.0, atol=1e-6)
    assert np.allclose(v.max(0) - v.min(0), 2.0, atol=1e-6)


def test_captured_missing_scale_raises(tmp_path):
    omomo = tmp_path / "OMOMO"
    cap = omomo / "data" / "captured_objects"
    cap.mkdir(parents=True)
    trimesh.creation.box(extents=(1.0, 1.0, 1.0)).export(
        cap / "widget_cleaned_simplified.obj")
    (omomo / "data").mkdir(exist_ok=True)
    joblib.dump({0: {"seq_name": "other"}},  # no obj_scale for our seq
                omomo / "data" / "train_diffusion_manip_seq_joints24.p")
    with pytest.raises(ValueError):
        resolve_omomo_object_mesh("sub9_widget_001", omomo_dir=omomo,
                                  cache_dir=tmp_path / "cache")


def test_no_omomo_dir_no_bundled_returns_none():
    assert resolve_omomo_object_mesh("sub9_widget_001", omomo_dir=None) is None
