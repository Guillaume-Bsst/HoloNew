"""contact_eval : champ courant + jacobiennes analytiques pour (q, poses objets). Gardé par le G1
URDF (jacobiennes cohérentes via le moteur pinocchio de Plan 1). ctx synthétique : contact_eval ne
lit que robot_cloud / channels / object_clouds / margin / robot. Les jacobiennes sont validées par
différences finies ; tangente objet world-aligned (δt, δθ): rotation = expm([δθ]_x) R (gauche, monde)."""
from pathlib import Path

import numpy as np
import pytest
from scipy.spatial.transform import Rotation as R

from src.prepare.contracts import (Channel, CorrespondenceTable, InteractionContext, PointCloud,
                                    RobotSpec)
from src.prepare.load.robot import build_robot_model
from src.prepare.sdf.build import build_plane_sdf
from src.targets.interaction import contact_eval, eval_fields, pose_cloud

_URDF = Path(__file__).resolve().parent.parent / "models" / "g1" / "g1_29dof.urdf"
_SKIP = pytest.mark.skipif(not _URDF.exists(), reason="G1 URDF absent")


def _robot():
    return build_robot_model(RobotSpec(name="g1", urdf_path=_URDF, link_names=(), dof=29, height=1.3))


def _ctx(robot):
    """Synthetic InteractionContext: a few robot control points on REAL links, a ground + one object
    plane channel, one object cloud. human_cloud / correspondence are unused dummies."""
    names = [n for n in ("left_elbow_link", "pelvis", "right_knee_link") if n in robot.link_names]
    rng = np.random.default_rng(3)
    parts = np.array([[robot.link_names.index(n)] for n in names], np.int64)        # (M, 1) FK order
    offsets = (0.05 * rng.standard_normal((len(names), 1, 3)))                       # (M, 1, 3) local
    robot_cloud = PointCloud(parts=parts, weights=np.ones((len(names), 1)), offsets=offsets)

    ground = Channel("ground", None, build_plane_sdf([-2, -2], [2, 2], spacing=0.1, margin=0.5, name="ground"))
    obj = Channel("obj0", 0, build_plane_sdf([-2, -2], [2, 2], spacing=0.1, margin=0.5, name="obj0"))
    obj_cloud = PointCloud(parts=np.zeros((2, 1), np.int64), weights=np.ones((2, 1)),
                           offsets=np.array([[[0.1, 0.0, 0.0]], [[0.0, 0.1, 0.0]]]))
    dummy = PointCloud(parts=np.zeros((1, 1), np.int64), weights=np.ones((1, 1)), offsets=np.zeros((1, 1, 3)))
    corr = CorrespondenceTable(smpl_idx=np.zeros(1, np.int64), link_idx=np.zeros(1, np.int64),
                               offset_local=np.zeros((1, 3)), link_names=robot.link_names)
    return InteractionContext(channels=(ground, obj), human_cloud=dummy, object_clouds=(obj_cloud,),
                              correspondence=corr, margin=0.5, robot_cloud=robot_cloud, robot=robot)


def _obj_pose():
    rot = R.from_rotvec([0.2, -0.5, 0.3]).as_matrix()[None]    # (1, 3, 3)
    pos = np.array([[0.4, -0.2, 0.6]])                         # (1, 3)
    return rot, pos


@_SKIP
def test_contact_eval_field_matches_eval_fields_direct():
    robot = _robot(); ctx = _ctx(robot)
    q = robot.integrate(robot.neutral(), 0.1 * np.random.default_rng(0).standard_normal(robot.nv))
    object_rot, object_pos = _obj_pose()

    ce = contact_eval(ctx, q, object_rot, object_pos)
    pts = pose_cloud(ctx.robot_cloud, *ctx.robot.link_transforms(q))    # (M, 3) world
    ref = eval_fields(pts, ctx.channels, object_rot, object_pos, ctx.margin)
    assert np.allclose(ce.field.distance, ref.distance)
    assert np.allclose(ce.field.witness, ref.witness)
    assert np.allclose(ce.field.direction, ref.direction)
    M, nv = ctx.robot_cloud.n_points, robot.nv
    assert ce.point_jac.shape == (M, 3, nv)
    assert ce.probe_jac_obj.shape == (len(ctx.channels), M, 3, 6)


@_SKIP
def test_point_jac_matches_finite_differences():
    robot = _robot(); ctx = _ctx(robot)
    rng = np.random.default_rng(1)
    q = robot.integrate(robot.neutral(), 0.1 * rng.standard_normal(robot.nv))
    object_rot, object_pos = _obj_pose()

    ce = contact_eval(ctx, q, object_rot, object_pos)
    nv, eps = robot.nv, 1e-6
    for k in range(nv):
        v = np.zeros(nv); v[k] = eps
        p_plus = pose_cloud(ctx.robot_cloud, *robot.link_transforms(robot.integrate(q, v)))
        p_minus = pose_cloud(ctx.robot_cloud, *robot.link_transforms(robot.integrate(q, -v)))
        fd = (p_plus - p_minus) / (2 * eps)                    # (M, 3) ∂(point monde)/∂v_k
        assert np.allclose(ce.point_jac[:, :, k], fd, atol=1e-4), k


@_SKIP
def test_probe_jac_obj_matches_finite_differences():
    robot = _robot(); ctx = _ctx(robot)
    q = robot.integrate(robot.neutral(), 0.1 * np.random.default_rng(2).standard_normal(robot.nv))
    object_rot, object_pos = _obj_pose()

    ce = contact_eval(ctx, q, object_rot, object_pos)
    points = pose_cloud(ctx.robot_cloud, *robot.link_transforms(q))   # (M, 3) world, held FIXED
    c = 1                                                            # channels[1] is the object channel
    j = ctx.channels[c].object_idx                                   # object index of that channel

    def probe_x(rot_j, pos_j):
        return (points - pos_j) @ rot_j                            # (M, 3) = R_jᵀ (p - t_j)

    eps = 1e-6
    for a in range(3):                                             # δt columns 0..2
        dt = np.zeros(3); dt[a] = eps
        fd = (probe_x(object_rot[j], object_pos[j] + dt) - probe_x(object_rot[j], object_pos[j] - dt)) / (2 * eps)
        assert np.allclose(ce.probe_jac_obj[c, :, :, a], fd, atol=1e-6), ("δt", a)
    for a in range(3):                                             # δθ columns 3..5 (world-aligned)
        w = np.zeros(3); w[a] = eps
        rp = R.from_rotvec(w).as_matrix() @ object_rot[j]
        rm = R.from_rotvec(-w).as_matrix() @ object_rot[j]
        fd = (probe_x(rp, object_pos[j]) - probe_x(rm, object_pos[j])) / (2 * eps)
        assert np.allclose(ce.probe_jac_obj[c, :, :, 3 + a], fd, atol=1e-6), ("δθ", a)
    assert np.allclose(ce.probe_jac_obj[0], 0.0)                   # ground channel rows = 0


@_SKIP
def test_env_cloud_jac_self_matches_finite_differences():
    robot = _robot(); ctx = _ctx(robot)
    q = robot.integrate(robot.neutral(), 0.1 * np.random.default_rng(4).standard_normal(robot.nv))
    object_rot, object_pos = _obj_pose()

    ce = contact_eval(ctx, q, object_rot, object_pos)
    assert len(ce.env) == len(ctx.object_clouds)
    env0 = ce.env[0]

    # field == eval_fields direct on the posed object cloud (self_idx=0)
    obj_world = pose_cloud(ctx.object_clouds[0], object_rot[0][None], object_pos[0][None])
    ref = eval_fields(obj_world, ctx.channels, object_rot, object_pos, ctx.margin, self_idx=0)
    assert np.allclose(env0.field.distance, ref.distance)
    assert np.allclose(env0.field.witness, ref.witness)

    eps = 1e-6
    for a in range(3):                                            # δt: ∂p/∂δt = I
        dt = np.zeros(3); dt[a] = eps
        pp = pose_cloud(ctx.object_clouds[0], object_rot[0][None], (object_pos[0] + dt)[None])
        pm = pose_cloud(ctx.object_clouds[0], object_rot[0][None], (object_pos[0] - dt)[None])
        fd = (pp - pm) / (2 * eps)
        assert np.allclose(env0.cloud_jac_self[:, :, a], fd, atol=1e-6), ("δt", a)
    for a in range(3):                                            # δθ: ∂p/∂δθ = -[p - t_i]_x
        w = np.zeros(3); w[a] = eps
        rp = R.from_rotvec(w).as_matrix() @ object_rot[0]
        rm = R.from_rotvec(-w).as_matrix() @ object_rot[0]
        pp = pose_cloud(ctx.object_clouds[0], rp[None], object_pos[0][None])
        pm = pose_cloud(ctx.object_clouds[0], rm[None], object_pos[0][None])
        fd = (pp - pm) / (2 * eps)
        assert np.allclose(env0.cloud_jac_self[:, :, 3 + a], fd, atol=1e-6), ("δθ", a)
