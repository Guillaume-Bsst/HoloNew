"""SDF band shells for the viewer's 'SDF Floor' / 'SDF Object' overlays. Shared by the
OMOMO (smplh) and HODome (smplx) paths so both datasets get the visualization."""
import numpy as np
import trimesh

from HoloNew.examples.view_stages import _sdf_floor_band, _sdf_object_band


def test_floor_band_three_z_layers():
    pts, cols = _sdf_floor_band((1.0, -2.0), 0.2)
    zs = np.unique(np.round(pts[:, 2], 6))
    np.testing.assert_allclose(zs, [-0.2, 0.0, 0.2], atol=1e-6)   # stacked sheets in [-L, L]
    assert pts.shape[0] == cols.shape[0] and cols.shape[1] == 3
    assert pts.shape[1] == 3
    # the grid is centred on the requested xy
    assert abs(pts[:, 0].mean() - 1.0) < 0.5 and abs(pts[:, 1].mean() + 2.0) < 0.5


def test_object_band_wraps_the_mesh(tmp_path):
    # A small box: the band shell (|signed dist| < L) must be non-empty and sit near it.
    box = trimesh.creation.box(extents=(0.2, 0.2, 0.2))
    mesh_file = tmp_path / "box.obj"
    box.export(mesh_file)
    pts, cols = _sdf_object_band(mesh_file, 0.1, 0.02, cache_dir=tmp_path)
    assert pts.shape[0] > 0                       # the |dist|<L shell is non-empty
    assert pts.shape[1] == 3 and cols.shape[1] == 3 and pts.shape[0] == cols.shape[0]
    # The band hugs the box surface (±0.1) within ~L of it.
    assert np.all(pts.max(0) < 0.25) and np.all(pts.min(0) > -0.25)


# --- integration: the real HODome object (baseball) gets non-empty SDF bands ---
import pytest  # noqa: E402
from pathlib import Path  # noqa: E402

from HoloNew.src.paths import get_path  # noqa: E402

_HAVE_BASEBALL = (get_path("hodome") / "scaned_object" / "baseball.tar").exists()


@pytest.mark.skipif(not _HAVE_BASEBALL, reason="HODome baseball mesh not present")
def test_hodome_object_band_nonempty(tmp_path):
    from HoloNew.src.data_loaders.hodome import extract_hodome_object_mesh
    mesh = extract_hodome_object_mesh("baseball", get_path("hodome") / "scaned_object",
                                      cache_dir=tmp_path / "meshes")
    pts, cols = _sdf_object_band(mesh, 0.2, 0.01, cache_dir=tmp_path / "contact")
    assert pts.shape[0] > 0 and pts.shape == cols.shape   # HODome object now has a band
    fpts, fcols = _sdf_floor_band((0.0, 0.0), 0.2)
    assert fpts.shape[0] > 0 and fpts.shape == fcols.shape
