"""Online field package on REAL prepared assets (data-gated): pose the robot_cloud at a config via
ctx.robot and re-evaluate against ctx.channels with the SAME pose_cloud/eval_fields that targets uses,
so solve gets (distance, witness) per channel per robot point — matching the reference field's (C, M)
layout. Skips when the HODome demo / SMPL-X / committed correspondence / G1 URDF are absent."""
import shutil
from pathlib import Path

import numpy as np
import pytest

from src.prepare.config import PrepareConfig
from src.prepare.contracts import RobotSpec, SceneSpec
from src.prepare.runner import prepare
from src.targets import MultiChannelField, eval_fields, pose_cloud
from src.targets.pipeline import process_frame

_DATA = Path("/home/vboxuser/Documents/wbt_rl/data/00_raw_datasets")
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
def test_online_package_evaluates_robot_cloud_against_channels(tmp_path):
    (tmp_path / "correspondence").mkdir(parents=True)
    shutil.copy(_CORR, tmp_path / "correspondence" / "corr_neutral.npz")
    spec = SceneSpec(dataset="hodome", motion_path=_SEQ,
                     robot=RobotSpec(name="g1", urdf_path=_URDF, link_names=("pelvis",), dof=29,
                                     height=1.3),
                     smpl_model_dir=_SMPLX, cache_dir=tmp_path)
    g, ctx = prepare(spec, PrepareConfig())

    M = ctx.correspondence.n_points
    C = len(ctx.channels)

    ft = process_frame(g, ctx, spec.robot, f=0)
    ref = ft.robot_interaction.field                       # reference (C, M) on the robot points
    assert ref.distance.shape == (C, M)

    # ONLINE: pose the robot_cloud at the rest config via ctx.robot, eval against the same channels.
    q = np.zeros(ctx.robot.dof)
    pts = pose_cloud(ctx.robot_cloud, *ctx.robot.link_transforms(q))    # (M, 3) world
    assert pts.shape == (M, 3)
    cur = eval_fields(pts, ctx.channels, ft.object_rot, ft.object_pos, ctx.margin)
    assert isinstance(cur, MultiChannelField)
    assert cur.distance.shape == (C, M)                    # same (channel, point) layout as reference
    assert cur.witness.shape == (C, M, 3)
    assert np.isfinite(cur.distance).all()                 # finite everywhere (active or clamped)
    assert np.isfinite(cur.witness).all()
