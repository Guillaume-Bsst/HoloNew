"""viewer bake test — the HEADLESS data path of the FrameTrace viewer (no viser server).

Runs ``prepare`` then bakes a few ``trace_frame`` on REAL HODome data (the same skip-guard / spec as
test_runner_prepare) and asserts each ``FrameTrace`` is well-formed: posed human cloud, the 14 style
link targets, the (C, N) human field, finite arrays, object clouds present. Also asserts the viewer
module imports and exposes its entry points (``Viewer`` / ``view_trace`` / ``main``). It NEVER starts a
viser server — that is the runnable ``__main__`` path, out of the headless test's scope.
"""
import shutil
from pathlib import Path

import numpy as np
import pytest

from src.prepare.contracts import RobotSpec, SceneSpec
from src.prepare.config import PrepareConfig
from src.prepare.runner import prepare
from src.targets.contracts import FrameTrace
from src.targets.pipeline import trace_frame

_DATA = Path("/home/vboxuser/Documents/wbt_rl/data/00_raw_datasets")
_HODOME = _DATA / "HODome"
_SMPLX = _DATA / "models" / "models_smplx_v1_1" / "models" / "smplx"
_CORR = Path(__file__).resolve().parent.parent / "cache" / "correspondence" / "corr_neutral.npz"


def _pick() -> Path | None:
    """A HODome sequence with BOTH smplx params and an object mesh (parametric + 1 object)."""
    sm, ob = _HODOME / "smplx", _HODOME / "object"
    if not (sm.is_dir() and ob.is_dir() and _SMPLX.is_dir() and _CORR.exists()):
        return None
    shared = {p.stem for p in sm.glob("*.npz")} & {p.stem for p in ob.glob("*.npz")}
    return sm / f"{sorted(shared)[0]}.npz" if shared else None


_SEQ = _pick()
_SKIP = pytest.mark.skipif(_SEQ is None, reason="HODome data / SMPL-X model / corr_neutral.npz absent")


def _robot() -> RobotSpec:
    return RobotSpec(name="g1", urdf_path=Path("g1.urdf"), link_names=("pelvis",), dof=29, height=1.3)


@pytest.fixture(scope="module")
def baked(tmp_path_factory):
    """prepare(spec) then bake a few trace_frame against a tmp cache seeded with corr_neutral.npz."""
    if _SEQ is None:
        pytest.skip("HODome data / SMPL-X model / corr_neutral.npz absent")
    cache = tmp_path_factory.mktemp("viewer_cache")
    (cache / "correspondence").mkdir(parents=True, exist_ok=True)
    shutil.copy(_CORR, cache / "correspondence" / "corr_neutral.npz")
    spec = SceneSpec(dataset="hodome", motion_path=_SEQ, robot=_robot(),
                     smpl_model_dir=_SMPLX, cache_dir=cache)
    grounded, ctx = prepare(spec, PrepareConfig())
    frames = list(range(0, grounded.n_frames, max(1, grounded.n_frames // 4)))[:4]
    traces = [trace_frame(grounded, ctx, spec.robot, f) for f in frames]
    return spec, grounded, ctx, traces


# --------------------------------------------------------------------------- case 1: well-formed bake
@_SKIP
def test_baked_traces_are_well_formed(baked):
    _, grounded, ctx, traces = baked
    assert len(traces) >= 1
    N = ctx.human_cloud.n_points
    C = len(ctx.channels)
    for tr in traces:
        assert isinstance(tr, FrameTrace)

        # posed human cloud (P, 3), finite.
        assert tr.human_cloud_world.shape == (N, 3)
        assert np.isfinite(tr.human_cloud_world).all()

        # the 14 style link targets: position (14, 3), orientation (14, 4) wxyz (geometry only).
        style = tr.targets.style
        L = len(style.link_names)
        assert L == 14
        assert style.position.shape == (L, 3)
        assert style.orientation is not None and style.orientation.shape == (L, 4)
        assert np.isfinite(style.position).all() and np.isfinite(style.orientation).all()

        # human field (C, N) channel-first, channel names aligned with the context.
        hf = tr.human_field
        assert hf.distance.shape == (C, N)
        assert hf.direction.shape == (C, N, 3)
        assert hf.active.shape == (C, N)
        assert hf.channels == ctx.channel_names
        assert np.isfinite(hf.distance).all() and np.isfinite(hf.direction).all()

        # object clouds present (the picked scene has >= 1 object), each (P_i, 3) finite.
        assert grounded.n_objects >= 1
        assert len(tr.object_clouds_world) == grounded.n_objects
        for oc in tr.object_clouds_world:
            assert oc.ndim == 2 and oc.shape[1] == 3
            assert np.isfinite(oc).all()

        # pose bones (J_bones, 3), the skeleton layer's source.
        assert tr.pose.bone_pos.shape[1] == 3


# --------------------------------------------------------------------------- case 2: viewer API surface
def test_viewer_module_exposes_entry():
    """The viewer module imports (pure consumer) and exposes its entry points — NO server started."""
    from src.viz import viewer
    assert hasattr(viewer, "Viewer")
    assert callable(viewer.view_trace)
    assert callable(viewer.main)
    # Viewer is a pure consumer: it only pulls prepare + trace_frame, no compute hook.
    assert viewer.trace_frame is trace_frame
