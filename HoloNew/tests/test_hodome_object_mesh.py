import tarfile
from pathlib import Path

import pytest

from HoloNew.src.data_loaders.hodome import extract_hodome_object_mesh


def _make_tar(scaned_dir: Path, token: str):
    scaned_dir.mkdir(parents=True, exist_ok=True)
    objdir = scaned_dir / "_src" / token
    objdir.mkdir(parents=True)
    (objdir / f"{token}.obj").write_text("o test\nv 0 0 0\n")
    with tarfile.open(scaned_dir / f"{token}.tar", "w") as t:
        t.add(objdir / f"{token}.obj", arcname=f"{token}/{token}.obj")


def test_extract_hodome_object_mesh(tmp_path):
    scaned = tmp_path / "scaned_object"
    _make_tar(scaned, "baseball")
    cache = tmp_path / "cache"

    out = extract_hodome_object_mesh("baseball", scaned, cache_dir=cache)
    assert out.name == "baseball.obj"
    assert out.exists()
    # second call uses the cache (idempotent), same path
    out2 = extract_hodome_object_mesh("baseball", scaned, cache_dir=cache)
    assert out2 == out


def test_extract_missing_tar_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        extract_hodome_object_mesh("nope", tmp_path / "scaned_object", cache_dir=tmp_path / "c")


_REPO = Path(__file__).resolve().parents[5]
_SCANED = _REPO / "data/00_raw_datasets/HODome/scaned_object"
_OBJ_NPZ = _REPO / "data/00_raw_datasets/HODome/object/subject01_baseball.npz"


@pytest.mark.skipif(not ((_SCANED / "baseball.tar").exists() and _OBJ_NPZ.exists()),
                    reason="HODome object data not present")
def test_hodome_object_pipeline_real(tmp_path):
    import numpy as np
    import trimesh
    from HoloNew.src.data_loaders.hodome import hodome_object_poses

    mesh_path = extract_hodome_object_mesh("baseball", _SCANED, cache_dir=tmp_path)
    mesh = trimesh.load(str(mesh_path), force="mesh", process=False)
    assert np.asarray(mesh.vertices).ndim == 2 and mesh.vertices.shape[1] == 3
    poses = hodome_object_poses(_OBJ_NPZ)
    assert poses.shape[1] == 7
    assert np.allclose(np.linalg.norm(poses[:, :4], axis=1), 1.0, atol=1e-4)  # unit quats
