"""build_solved_frame : la seam solve du viz. Data-gated (HODome / SMPL-X / corr / G1 URDF, comme
test_solve_runner). Vérifie déterminisme (build x2 identique), formes/dtypes, et que l'« atteint »
(style_achieved/contact_achieved) provient bien d'un appel Evaluator direct (pattern test_solve_runner).
Un smoke-test BakeSource(solve=True) verrouille la glu (dépend de Phase A : BakeSource).

Note : le smoke-test utilise max_frames=1 pour rester sous ~40 s (le budget global est la même
borne que test_solve_runner, documenté dans les global constraints de la spec Phase B)."""
import shutil
from pathlib import Path

import numpy as np
import pytest

from src.prepare.config import PrepareConfig
from src.prepare.contracts import RobotSpec, SceneSpec
from src.prepare.runner import prepare
from src.targets.evaluator import Evaluator
from src.targets.pipeline import process_frame
from src.solve.config import SolveConfig
from src.solve.runner import solve
from src.viz.model import SolvedFrame
from src.viz.sources import build_solved_frame
from datapaths import HODOME as _HODOME, SMPLX_MODELS as _SMPLX

_CORR = Path(__file__).resolve().parent.parent / "cache" / "correspondence" / "corr_neutral.npz"
_URDF = Path(__file__).resolve().parent.parent / "models" / "g1" / "g1_29dof.urdf"


def _pick():
    sm, ob = _HODOME / "smplx", _HODOME / "object"
    if not (sm.is_dir() and ob.is_dir() and _SMPLX.is_dir() and _CORR.exists() and _URDF.exists()):
        return None
    shared = {p.stem for p in sm.glob("*.npz")} & {p.stem for p in ob.glob("*.npz")}
    return sm / f"{sorted(shared)[0]}.npz" if shared else None


_SEQ = _pick()
_SKIP = pytest.mark.skipif(_SEQ is None, reason="HODome / SMPL-X / corr / G1 URDF absent")


def _prepare_solve(tmp_path):
    """prepare -> quelques FrameTargets -> solve -> (g, ctx, spec, traj, ev, T). T bas (suite rapide)."""
    (tmp_path / "correspondence").mkdir(parents=True)
    shutil.copy(_CORR, tmp_path / "correspondence" / "corr_neutral.npz")
    spec = SceneSpec(dataset="hodome", motion_path=_SEQ,
                     robot=RobotSpec(name="g1", urdf_path=_URDF, link_names=("pelvis",), dof=29,
                                     height=1.3),
                     smpl_model_dir=_SMPLX, cache_dir=tmp_path)
    g, ctx = prepare(spec, PrepareConfig())
    T = min(2, g.n_frames)
    frame_targets = [process_frame(g, ctx, spec.robot, f) for f in range(T)]
    traj = solve(g, ctx, frame_targets, SolveConfig(), robot_name="g1")
    ev = Evaluator(ctx, "g1")
    return g, ctx, spec, traj, ev, T


@_SKIP
def test_build_solved_frame_shapes_and_achieved(tmp_path):
    g, ctx, spec, traj, ev, T = _prepare_solve(tmp_path)
    L = len(ctx.robot.link_names)
    M = ctx.robot_cloud.n_points
    N = len(ctx.object_clouds)
    for f in range(T):
        sf = build_solved_frame(traj, ev, ctx, f)
        assert isinstance(sf, SolvedFrame)
        # formes / dtypes
        assert sf.q.shape == (ctx.robot.nq,) and sf.q.dtype == np.float64
        assert sf.object_poses.shape == (N, 7)
        assert sf.robot_points_world.shape == (M, 3)
        assert np.isfinite(sf.robot_points_world).all()
        assert sf.link_transforms.shape == (L, 4, 4)
        assert np.allclose(sf.link_transforms[:, 3, :], np.array([0.0, 0.0, 0.0, 1.0]))  # ligne homogène
        # diagnostics FrameInfo
        assert isinstance(sf.cost, float)
        assert isinstance(sf.cost_by_term, dict)
        assert isinstance(sf.n_iters, int) and isinstance(sf.status, str)
        # « atteint » == appel Evaluator direct (le pattern test_solve_runner)
        assert np.allclose(sf.style_achieved.position, ev.style(traj.qpos[f]).position)
        assert sf.contact_achieved.field.n_points == M


@_SKIP
def test_build_solved_frame_deterministic(tmp_path):
    g, ctx, spec, traj, ev, T = _prepare_solve(tmp_path)
    for f in range(T):
        a = build_solved_frame(traj, ev, ctx, f)
        b = build_solved_frame(traj, ev, ctx, f)
        assert np.array_equal(a.q, b.q)
        assert np.array_equal(a.object_poses, b.object_poses)
        assert np.array_equal(a.robot_points_world, b.robot_points_world)
        assert np.array_equal(a.link_transforms, b.link_transforms)
        assert np.array_equal(a.style_achieved.position, b.style_achieved.position)
        assert np.array_equal(a.contact_achieved.field.distance, b.contact_achieved.field.distance)


@_SKIP
def test_bake_source_solve_fills_solved(tmp_path):
    """Smoke : BakeSource(solve=True).get(f).solved est non-None + cohérent (dépend de Phase A).
    max_frames=1 pour rester dans le budget de temps (~40 s max, identique à test_solve_runner)."""
    from src.viz.sources import BakeSource
    (tmp_path / "correspondence").mkdir(parents=True)
    shutil.copy(_CORR, tmp_path / "correspondence" / "corr_neutral.npz")
    spec = SceneSpec(dataset="hodome", motion_path=_SEQ,
                     robot=RobotSpec(name="g1", urdf_path=_URDF, link_names=("pelvis",), dof=29,
                                     height=1.3),
                     smpl_model_dir=_SMPLX, cache_dir=tmp_path)
    # max_frames=1 : un seul frame résolu (budget ~40 s)
    src = BakeSource(spec, PrepareConfig(), solve=True, max_frames=1, frame_step=1000)
    assert src.context.has_solve is True
    vf = src.get(0)
    assert vf.solved is not None
    assert vf.solved.q.shape[0] > 0  # q présent (nq > 0)
    assert vf.solved.status != ""
