"""BakeSource — la cuisson headless du vue-modèle sur les données HODome réelles (sans serveur
viser). Déterminisme (build ×2 identique), formes/dtypes des VizFrame (nuages monde float32,
solved=None). Le seam solve=True est testé dans test_viz_solved_frame.py (Phase B). Même garde de
skip / spec que test_viewer_bake.py."""
import shutil
from pathlib import Path

import numpy as np
import pytest

from src.prepare.config import PrepareConfig
from src.prepare.contracts import RobotSpec, SceneSpec
from src.viz.model import VizContext, VizFrame
from src.viz.sources import BakeSource, Source
from datapaths import HODOME as _HODOME, SMPLX_MODELS as _SMPLX
_CORR = Path(__file__).resolve().parent.parent / "cache" / "correspondence" / "corr_neutral.npz"
_URDF = Path(__file__).resolve().parent.parent / "models" / "g1" / "g1_29dof.urdf"


def _pick() -> Path | None:
    sm, ob = _HODOME / "smplx", _HODOME / "object"
    if not (sm.is_dir() and ob.is_dir() and _SMPLX.is_dir() and _CORR.exists()):
        return None
    shared = {p.stem for p in sm.glob("*.npz")} & {p.stem for p in ob.glob("*.npz")}
    return sm / f"{sorted(shared)[0]}.npz" if shared else None


_SEQ = _pick()
_SKIP = pytest.mark.skipif(_SEQ is None, reason="HODome data / SMPL-X model / corr_neutral.npz absent")


def _robot() -> RobotSpec:
    return RobotSpec(name="g1", urdf_path=_URDF, link_names=("pelvis",), dof=29, height=1.3)


@pytest.fixture(scope="module")
def cache(tmp_path_factory):
    c = tmp_path_factory.mktemp("viz_bake_cache")
    (c / "correspondence").mkdir(parents=True, exist_ok=True)
    shutil.copy(_CORR, c / "correspondence" / "corr_neutral.npz")
    return c


def _spec(cache) -> SceneSpec:
    return SceneSpec(dataset="hodome", motion_path=_SEQ, robot=_robot(),
                     smpl_model_dir=_SMPLX, cache_dir=cache)


@_SKIP
def test_source_protocol_and_context(cache):
    src = BakeSource(_spec(cache), PrepareConfig(), solve=False, frame_step=8, max_frames=3)
    assert isinstance(src, Source)
    assert isinstance(src.context, VizContext)
    assert src.context.has_solve is False
    assert len(src.context.channel_names) == src.context.n_objects + 1
    assert src.context.channel_names[0] == "ground"
    assert 1 <= src.n_frames <= 3


@_SKIP
def test_vizframe_shapes_dtypes_solved_none(cache):
    src = BakeSource(_spec(cache), PrepareConfig(), solve=False, frame_step=8, max_frames=3)
    ctx = src.context
    for i in range(src.n_frames):
        fr = src.get(i)
        assert isinstance(fr, VizFrame)
        assert fr.solved is None
        assert fr.human_cloud_world.dtype == np.float32 and fr.human_cloud_world.shape[1] == 3
        assert fr.smpl_verts_world is not None and fr.smpl_verts_world.dtype == np.float32
        assert len(fr.object_clouds_world) == ctx.n_objects
        for oc in fr.object_clouds_world:
            assert oc.dtype == np.float32 and oc.shape[1] == 3
        # champ canal-first (C, N), aligné avec les canaux du contexte
        assert fr.human_field.distance.shape[0] == len(ctx.channel_names)
        assert np.isfinite(fr.human_cloud_world).all()


@_SKIP
def test_determinism_build_twice_identical(cache):
    a = BakeSource(_spec(cache), PrepareConfig(), solve=False, frame_step=8, max_frames=3)
    b = BakeSource(_spec(cache), PrepareConfig(), solve=False, frame_step=8, max_frames=3)
    assert a.n_frames == b.n_frames
    for i in range(a.n_frames):
        fa, fb = a.get(i), b.get(i)
        assert np.array_equal(fa.human_cloud_world, fb.human_cloud_world)
        assert np.array_equal(fa.smpl_verts_world, fb.smpl_verts_world)
        assert np.array_equal(fa.human_field.distance, fb.human_field.distance)
        for oa, ob in zip(fa.object_clouds_world, fb.object_clouds_world):
            assert np.array_equal(oa, ob)


# test_solve_true_not_implemented supprimé en phase B : BakeSource(solve=True) est désormais
# implémenté. Le comportement est vérifié dans tests/test_viz_solved_frame.py.
