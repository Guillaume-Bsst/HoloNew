"""solve end-to-end (data-gated HODome) : prepare -> quelques FrameTargets -> solve. Vérifie une qpos
finie de bonne forme, des objets (T,N,7), un statut optimal par frame, la sanité des joints, et le
pelvis solveur ~ cible pelvis du style (le seed Holosoma + le suivi de style le tiennent proche)."""
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
from src.solve.contracts import SolveTrajectory
from src.solve.runner import solve
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
_SKIP = pytest.mark.skipif(_SEQ is None, reason="HODome data / SMPL-X / corr / G1 URDF absent")


@_SKIP
def test_solve_end_to_end_on_real_assets(tmp_path):
    (tmp_path / "correspondence").mkdir(parents=True)
    shutil.copy(_CORR, tmp_path / "correspondence" / "corr_neutral.npz")
    spec = SceneSpec(dataset="hodome", motion_path=_SEQ,
                     robot=RobotSpec(name="g1", urdf_path=_URDF, link_names=("pelvis",), dof=29,
                                     height=1.3),
                     smpl_model_dir=_SMPLX, cache_dir=tmp_path)
    g, ctx = prepare(spec, PrepareConfig())

    T = min(2, g.n_frames)                                   # max_frames bas (suite rapide)
    frame_targets = [process_frame(g, ctx, spec.robot, f) for f in range(T)]

    traj = solve(g, ctx, frame_targets, SolveConfig(), robot_name="g1")

    assert isinstance(traj, SolveTrajectory)
    assert traj.qpos.shape == (T, ctx.robot.nq)
    assert np.isfinite(traj.qpos).all()
    assert traj.object_poses.shape == (T, len(ctx.object_clouds), 7)
    assert np.isfinite(traj.object_poses).all()
    assert len(traj.info) == T

    # joints : finis et dans une bande de sanité (les limites dures vivent dans build_constraints ; le
    # protocol RobotModel n'expose pas les bornes -> on borne grossièrement + on exige un statut feasible).
    assert np.all(np.abs(traj.qpos[:, 7:]) < 2 * np.pi)
    for fi in traj.info:
        assert fi.status in ("optimal", "optimal_inaccurate")
        assert np.isfinite(fi.cost)

    # pelvis solveur ~ cible pelvis du style (seed = cible pelvis exacte + suivi de style).
    ev = Evaluator(ctx, "g1")
    pidx = tuple(frame_targets[0].style.link_names).index("pelvis")
    for f in range(T):
        pos_solver = ev.style(traj.qpos[f]).position[pidx]
        pos_target = frame_targets[f].style.position[pidx]
        assert np.linalg.norm(pos_solver - pos_target) < 0.15      # < 15 cm
