"""prepare/runner integration test: wire load -> calibration -> scene -> channels -> clouds ->
correspondence into ``(GroundedScene, InteractionContext)`` on REAL data, asserting the assembly
invariants, the disk-cache round-trip/determinism, and that prepare's output flows end-to-end into
the interaction ops (pose -> eval -> transport).

Skips cleanly when the HODome demo data / SMPL-X model / committed correspondence are absent (same
guard style as test_point_cloud_*). Uses a tmp cache dir seeded with the committed corr_neutral.npz,
so the repo cache is untouched and no robot URDF is needed (the correspondence is loaded, not built).
"""
import shutil
from pathlib import Path

import numpy as np
import pytest

from src.prepare.contracts import GroundedScene, InteractionContext, RobotSpec, SceneSpec
from src.prepare.config import PrepareConfig
from src.prepare.runner import prepare
from src.targets.pipeline import frame_pose
from src.targets.interaction import eval_fields, pose_cloud, transport

_DATA = Path("/home/vboxuser/Documents/wbt_rl/data/00_raw_datasets")
_HODOME = _DATA / "HODome"
_SMPLX = _DATA / "models" / "models_smplx_v1_1" / "models" / "smplx"
_CORR = Path(__file__).resolve().parent.parent / "cache" / "correspondence" / "corr_neutral.npz"
_URDF = Path(__file__).resolve().parent.parent / "models" / "g1" / "g1_29dof.urdf"


def _pick() -> Path | None:
    """A HODome sequence that has BOTH smplx params and an object mesh (parametric + 1 object)."""
    sm, ob = _HODOME / "smplx", _HODOME / "object"
    if not (sm.is_dir() and ob.is_dir() and _SMPLX.is_dir() and _CORR.exists() and _URDF.exists()):
        return None
    shared = {p.stem for p in sm.glob("*.npz")} & {p.stem for p in ob.glob("*.npz")}
    return sm / f"{sorted(shared)[0]}.npz" if shared else None


_SEQ = _pick()
_SKIP = pytest.mark.skipif(_SEQ is None, reason="HODome data / SMPL-X model / corr_neutral.npz absent")


def _robot() -> RobotSpec:
    return RobotSpec(name="g1", urdf_path=_URDF, link_names=("pelvis",), dof=29, height=1.3)


@pytest.fixture(scope="module")
def prepared(tmp_path_factory):
    """Run prepare TWICE against a tmp cache (cold build, then warm load) — shared by the cases."""
    if _SEQ is None:
        pytest.skip("HODome data / SMPL-X model / corr_neutral.npz absent")
    cache = tmp_path_factory.mktemp("prep_cache")
    (cache / "correspondence").mkdir(parents=True, exist_ok=True)
    shutil.copy(_CORR, cache / "correspondence" / "corr_neutral.npz")   # reuse the committed default
    spec = SceneSpec(dataset="hodome", motion_path=_SEQ, robot=_robot(),
                     smpl_model_dir=_SMPLX, cache_dir=cache)
    cfg = PrepareConfig()
    g1, ctx1 = prepare(spec, cfg)        # cold: builds + caches every asset
    g2, ctx2 = prepare(spec, cfg)        # warm: loads each cached asset back
    return spec, cfg, g1, ctx1, g2, ctx2


# --------------------------------------------------------------------------- case 1: assembly
@_SKIP
def test_prepare_returns_grounded_and_context_with_invariants(prepared):
    _, cfg, g, ctx, *_ = prepared
    assert isinstance(g, GroundedScene)
    assert isinstance(ctx, InteractionContext)

    # ground first, then object channels aligned 0..N-1 with the clouds and the scene's objects.
    assert ctx.channels[0].name == "ground"
    assert ctx.channels[0].object_idx is None
    assert g.n_objects >= 1                                   # the picked HODome scene has an object
    assert len(ctx.channels) == g.n_objects + 1
    assert len(ctx.object_clouds) == g.n_objects
    for i, ch in enumerate(ctx.channels[1:]):
        assert ch.object_idx == i

    # the binding the transport gather relies on.
    assert ctx.human_cloud.sampling_id == ctx.correspondence.smpl_sampling_id
    assert ctx.margin == cfg.sdf.margin

    # sane shapes: human cloud skinning, SDF grids/witness, correspondence points.
    N, K = ctx.human_cloud.n_points, ctx.human_cloud.n_influences
    assert N > 0 and K == cfg.cloud.k_influences
    assert ctx.human_cloud.parts.shape == (N, K)
    assert np.allclose(ctx.human_cloud.weights.sum(axis=1), 1.0, atol=1e-5)
    for ch in ctx.channels:
        assert ch.sdf.grid.ndim == 3
        assert ch.sdf.witness.shape == ch.sdf.grid.shape + (3,)
    assert ctx.correspondence.n_points > 0
    assert ctx.object_clouds[0].n_influences == 1            # objects are rigid K=1

    # the robot side carried for solve: same M points as the correspondence + a usable FK engine.
    assert ctx.robot_cloud.n_points == ctx.correspondence.n_points
    assert ctx.robot_cloud.n_influences == 1
    assert "pelvis" in ctx.robot.link_names and ctx.robot.dof == 29


# --------------------------------------------------------------------------- case 2: cache round-trip
@_SKIP
def test_prepare_cache_roundtrip_is_deterministic(prepared):
    _, _, _, ctx1, _, ctx2 = prepared
    # human cloud: cold build vs warm load == array-equal (the .npz round-trips exactly).
    assert np.array_equal(ctx1.human_cloud.parts, ctx2.human_cloud.parts)
    assert np.array_equal(ctx1.human_cloud.weights, ctx2.human_cloud.weights)
    assert np.array_equal(ctx1.human_cloud.offsets, ctx2.human_cloud.offsets)
    assert ctx1.human_cloud.sampling_id == ctx2.human_cloud.sampling_id

    # object cloud + object SDF grid (the cached object asset) load back identically.
    assert np.array_equal(ctx1.object_clouds[0].offsets, ctx2.object_clouds[0].offsets)
    assert np.array_equal(ctx1.channels[1].sdf.grid, ctx2.channels[1].sdf.grid)
    # ground plane SDF is rebuilt analytically both times -> deterministic, equal.
    assert np.array_equal(ctx1.channels[0].sdf.grid, ctx2.channels[0].sdf.grid)


# --------------------------------------------------------------------------- case 3: prepare -> interaction
@_SKIP
def test_prepare_output_flows_through_interaction_ops(prepared):
    _, _, g, ctx, *_ = prepared
    pose = frame_pose(g, 0)                                   # bone (R,t) + object (R,t), no style

    human_world = pose_cloud(ctx.human_cloud, pose.bone_rot, pose.bone_pos)   # (N, 3)
    assert human_world.shape == (ctx.human_cloud.n_points, 3)
    assert np.isfinite(human_world).all()

    human_field = eval_fields(human_world, ctx.channels, pose.object_rot, pose.object_pos, ctx.margin)
    C, N = len(ctx.channels), ctx.human_cloud.n_points
    assert human_field.distance.shape == (C, N)

    robot_field = transport(human_field, ctx.correspondence)
    M = ctx.correspondence.n_points
    assert robot_field.distance.shape == (C, M)
    assert robot_field.direction.shape == (C, M, 3)
    assert robot_field.active.shape == (C, M)
    assert np.isfinite(robot_field.distance).all()
    assert np.isfinite(robot_field.direction).all()
