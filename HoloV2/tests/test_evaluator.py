"""Evaluator (intégration) sur des assets réellement préparés (data-gated, comme test_solve_field_eval).
Construit Evaluator(ctx, "g1") une fois ; à q neutre, StyleEval.position est fini et de bonnes formes,
et ContactEval.field retombe sur eval_fields direct sur le robot_cloud posé @q — sanity du seam."""
import shutil
from pathlib import Path

import numpy as np
import pytest

from src.prepare.config import PrepareConfig
from src.prepare.contracts import RobotSpec, SceneSpec
from src.prepare.runner import prepare
from src.targets.evaluator import Evaluator
from src.targets.contracts import StyleEval, ContactEval
from src.targets import eval_fields, pose_cloud
from src.targets.config import style_table
from src.targets.pipeline import process_frame

_DATA = Path.home() / "Documents" / "wbt_rl" / "data" / "00_raw_datasets"  # machine-agnostic (was hardcoded)
_HODOME = _DATA / "HODome"
_SMPLX = _DATA / "models" / "models_smplx_v1_1" / "models" / "smplx"
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
def test_evaluator_style_and_contacts_on_real_assets(tmp_path):
    (tmp_path / "correspondence").mkdir(parents=True)
    shutil.copy(_CORR, tmp_path / "correspondence" / "corr_neutral.npz")
    spec = SceneSpec(dataset="hodome", motion_path=_SEQ,
                     robot=RobotSpec(name="g1", urdf_path=_URDF, link_names=("pelvis",), dof=29,
                                     height=1.3),
                     smpl_model_dir=_SMPLX, cache_dir=tmp_path)
    g, ctx = prepare(spec, PrepareConfig())

    ev = Evaluator(ctx, spec.robot.name)
    q = ctx.robot.neutral()                                    # valid free-flyer config (nq,)

    se = ev.style(q)
    assert isinstance(se, StyleEval)
    L = len(style_table("g1"))
    assert se.position.shape == (L, 3) and se.jac_pos.shape == (L, 3, ctx.robot.nv)
    assert np.isfinite(se.position).all() and np.isfinite(se.jac_pos).all()

    ft = process_frame(g, ctx, spec.robot, f=0)
    ce = ev.contacts(q, ft.object_rot, ft.object_pos)
    assert isinstance(ce, ContactEval)
    M, C = ctx.correspondence.n_points, len(ctx.channels)
    assert ce.point_jac.shape == (M, 3, ctx.robot.nv)
    assert ce.probe_jac_obj.shape == (C, M, 3, 6)
    assert len(ce.env) == len(ctx.object_clouds)

    pts = pose_cloud(ctx.robot_cloud, *ctx.robot.link_transforms(q))   # (M, 3) world
    ref = eval_fields(pts, ctx.channels, ft.object_rot, ft.object_pos, ctx.margin)
    assert np.allclose(ce.field.distance, ref.distance)       # ContactEval.field == eval_fields direct
    assert np.allclose(ce.field.witness, ref.witness)
