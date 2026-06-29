"""process_frame integration (synthetic, torch-free): the assembled FrameTargets carries the per-frame
object world transforms (from frame_pose) — the seam solve needs to re-pose the object channels and
seed the object-variable terms. Exercises the full pose->style->eval->transport->assemble flow on a
fake 22-bone body + analytic plane SDFs (no SMPL, no trimesh, no robot URDF)."""
from pathlib import Path

import numpy as np
import pytest

from src.prepare.contracts import (Calibration, Channel, CorrespondenceTable, GroundedScene,
                                    InteractionContext, PointCloud, RobotSpec)
from src.prepare.sdf.build import build_plane_sdf
from src.targets.pipeline import frame_pose, process_frame


class _Body22:
    """Minimal BodyModel with 22 SMPL-X bones (the G1 style table reads bone indices up to 21)."""
    faces = np.zeros((1, 3), np.int64)
    n_bones = 22
    stature = 1.7

    def bone_transforms(self, params, t):
        rot = np.tile(np.eye(3), (22, 1, 1))                 # (22, 3, 3)
        pos = np.zeros((22, 3)); pos[:, 2] = 0.9             # all bones at z=0.9
        pos[0, 2] = 0.9 + 0.01 * t                            # frame index leaks into pelvis z
        return rot, pos


def _ctx(n_obj, obj_off_z=0.0):
    margin = 0.1
    channels = [Channel("ground", None,
                        build_plane_sdf([-1.0, -1.0], [1.0, 1.0], spacing=0.1, margin=margin,
                                        name="ground"))]
    for i in range(n_obj):
        channels.append(Channel(f"obj{i}", i,
                                build_plane_sdf([-1.0, -1.0], [1.0, 1.0], spacing=0.1, margin=margin,
                                                name=f"obj{i}")))
    human = PointCloud(parts=np.zeros((5, 1), np.int64), weights=np.ones((5, 1), np.float32),
                       offsets=np.zeros((5, 1, 3), np.float32), sampling_id="s")
    # obj_off_z lifts the object cloud OFF its own (thin, z~0) plane SDF: with a real sample the self
    # channel would fall out of grid (inactive), so only the short-circuit yields distance 0 / active.
    obj_off = np.zeros((3, 1, 3), np.float32); obj_off[:, :, 2] = obj_off_z
    obj_clouds = tuple(PointCloud(parts=np.zeros((3, 1), np.int64), weights=np.ones((3, 1), np.float32),
                                  offsets=obj_off.copy()) for _ in range(n_obj))
    corr = CorrespondenceTable(smpl_idx=np.array([0, 1, 2, 3]), link_idx=np.zeros(4, np.int64),
                               offset_local=np.zeros((4, 3)), link_names=("root",),
                               smpl_sampling_id="s")
    robot_cloud = PointCloud(parts=np.zeros((4, 1), np.int64), weights=np.ones((4, 1), np.float32),
                             offsets=np.zeros((4, 1, 3), np.float32))
    # process_frame never touches ctx.robot / ctx.robot_cloud (those are solve-only) -> robot=None ok.
    return InteractionContext(channels=tuple(channels), human_cloud=human, object_clouds=obj_clouds,
                              correspondence=corr, margin=margin, robot_cloud=robot_cloud, robot=None)


def _grounded(n_obj, T=3):
    obj = np.tile([0.2, 0.3, 0.5, 1, 0, 0, 0], (T, 1)).astype(np.float32)   # identity-quat pose
    return GroundedScene(joint_pos=np.zeros((T, 1, 3), np.float32), joint_names=("a",),
                         object_poses=(obj,) * n_obj, object_mesh_paths=(Path("o.obj"),) * n_obj,
                         calibration=Calibration(0.0, 0.0, np.eye(4)), fps=30.0,
                         smpl_params=None, body=_Body22())


def test_frame_targets_carry_object_poses_from_frame_pose():
    from src.targets.config import TargetsConfig, SceneScaleConfig
    g, ctx = _grounded(n_obj=2), _ctx(n_obj=2)
    robot = RobotSpec(name="g1", urdf_path=Path("g1.urdf"), link_names=(), dof=29, height=1.3)
    identity = TargetsConfig(scene_scale=SceneScaleConfig(scale_xy=1.0, scale_z=1.0))   # no-op
    ft = process_frame(g, ctx, robot, f=1, cfg=identity)
    pose = frame_pose(g, f=1)
    assert ft.object_rot.shape == (2, 3, 3) and ft.object_pos.shape == (2, 3)
    assert np.allclose(ft.object_rot, pose.object_rot)
    assert np.allclose(ft.object_pos, pose.object_pos)
    assert np.allclose(ft.object_pos[0], [0.2, 0.3, 0.5])           # the grounded object position


def test_process_frame_requires_parametric_body():
    """The targets pipeline is bone-based (style + cloud posing both need the SMPL body), so a
    positions-only scene (``body is None``) is rejected with an explicit contract error — not a bare
    ``AttributeError`` on ``grounded.body.stature``."""
    g = _grounded(n_obj=1)
    g = GroundedScene(joint_pos=g.joint_pos, joint_names=g.joint_names, object_poses=g.object_poses,
                      object_mesh_paths=g.object_mesh_paths, calibration=g.calibration, fps=g.fps,
                      smpl_params=None, body=None)          # positions-only
    robot = RobotSpec(name="g1", urdf_path=Path("g1.urdf"), link_names=(), dof=29, height=1.3)
    with pytest.raises(ValueError, match="parametric body"):
        process_frame(g, _ctx(n_obj=1), robot, f=0)


def test_self_channel_short_circuited_in_env_targets():
    """``_build_frame`` must pass ``self_idx=i`` so object cloud ``i``'s SELF channel (object_idx == i)
    is the on-surface fill (distance 0, active), not an SDF sample. The clouds sit OFF their own SDF
    grid (``obj_off_z``), so a real sample would be out-of-grid/inactive — only the wired short-circuit
    gives distance 0."""
    g, ctx = _grounded(n_obj=2), _ctx(n_obj=2, obj_off_z=5.0)
    robot = RobotSpec(name="g1", urdf_path=Path("g1.urdf"), link_names=(), dof=29, height=1.3)
    ft = process_frame(g, ctx, robot, f=0)
    # channels are (ground, obj0, obj1); object 0's self channel is index 1, object 1's is index 2.
    f0 = ft.env_interaction.per_object[0]
    assert np.allclose(f0.distance[1], 0.0) and f0.active[1].all()      # obj0 vs obj0 = self
    f1 = ft.env_interaction.per_object[1]
    assert np.allclose(f1.distance[2], 0.0) and f1.active[2].all()      # obj1 vs obj1 = self
    # and a NON-self object channel is still a real sample: obj0's channel-2 (obj1) is off-grid -> inactive.
    assert not f0.active[2].any()


def test_object_pos_scaled_by_scene():
    from src.targets.config import StyleConfig, TargetsConfig, SceneScaleConfig
    g, ctx = _grounded(n_obj=1), _ctx(n_obj=1)
    robot = RobotSpec(name="g1", urdf_path=Path("g1.urdf"), link_names=(), dof=29, height=1.3)
    ratio = g.body.stature / StyleConfig().human_height_assumption          # 1.7 / 1.8
    identity = TargetsConfig(scene_scale=SceneScaleConfig(scale_xy=1.0, scale_z=1.0))
    ft_native = process_frame(g, ctx, robot, f=0, cfg=identity)
    ft_scaled = process_frame(g, ctx, robot, f=0)                            # défaut None,None -> ratio
    np.testing.assert_allclose(ft_native.object_pos[0], [0.2, 0.3, 0.5], atol=1e-9)
    # z scale uniforme par ratio car StyleConfig().ground_height == 0.0 (ancre sol = origine)
    np.testing.assert_allclose(ft_scaled.object_pos[0], np.array([0.2, 0.3, 0.5]) * ratio, atol=1e-9)
    np.testing.assert_array_equal(ft_scaled.object_rot, ft_native.object_rot)   # rotation inchangée


def test_ground_channel_scaled_in_robot_field():
    """Wiring : le pipeline scale le canal SOL des refs. Corps près du sol -> canal sol actif ;
    on vérifie distance(hauteur) *= s_z (le z du witness reste sur le plan, xy nuls ici)."""
    from src.targets.config import TargetsConfig, SceneScaleConfig

    class _BodyLow(_Body22):
        def bone_transforms(self, params, t):
            rot, pos = _Body22.bone_transforms(self, params, t)
            pos = pos.copy(); pos[:, 2] = 0.05            # dans la marge (0.1) du plan sol z~0
            return rot, pos

    g0 = _grounded(n_obj=1)
    g = GroundedScene(joint_pos=g0.joint_pos, joint_names=g0.joint_names, object_poses=g0.object_poses,
                      object_mesh_paths=g0.object_mesh_paths, calibration=g0.calibration, fps=g0.fps,
                      smpl_params=None, body=_BodyLow())
    ctx = _ctx(n_obj=1)
    robot = RobotSpec(name="g1", urdf_path=Path("g1.urdf"), link_names=(), dof=29, height=1.3)
    ft_id = process_frame(g, ctx, robot, f=0,
                          cfg=TargetsConfig(scene_scale=SceneScaleConfig(scale_xy=1.0, scale_z=1.0)))
    ft_sc = process_frame(g, ctx, robot, f=0,
                          cfg=TargetsConfig(scene_scale=SceneScaleConfig(scale_xy=2.0, scale_z=2.0)))
    a = np.asarray(ft_id.robot_interaction.field.active[0])           # canal sol = index 0
    did = np.asarray(ft_id.robot_interaction.field.distance[0])
    dsc = np.asarray(ft_sc.robot_interaction.field.distance[0])
    assert a.any()                                                    # sol actif (corps dans la marge)
    np.testing.assert_allclose(dsc[a], did[a] * 2.0, atol=1e-9)       # hauteur scalée par s_z
    # z du witness reste sur le plan (ground_height 0) -> invariant
    wid = np.asarray(ft_id.robot_interaction.field.witness[0])
    wsc = np.asarray(ft_sc.robot_interaction.field.witness[0])
    np.testing.assert_allclose(wsc[a][:, 2], wid[a][:, 2], atol=1e-9)


def test_frame_targets_rejects_object_pose_count_mismatch():
    from src.targets.contracts import (EnvironmentInteractionTargets, FrameTargets,
                                        MultiChannelField, RobotInteractionTargets, StyleTargets)

    def _mcf(c, p):
        return MultiChannelField(distance=np.zeros((c, p)), direction=np.zeros((c, p, 3)),
                                 witness=np.zeros((c, p, 3)), active=np.zeros((c, p), bool),
                                 channels=tuple(f"ch{i}" for i in range(c)))

    style = StyleTargets(link_names=(), position=np.zeros((0, 3)))
    env = EnvironmentInteractionTargets(per_object=(_mcf(1, 3), _mcf(1, 3)))   # N=2
    with pytest.raises(ValueError, match="per_object"):
        FrameTargets(style=style, robot_interaction=RobotInteractionTargets(field=_mcf(1, 4)),
                     env_interaction=env,
                     object_rot=np.zeros((1, 3, 3)), object_pos=np.zeros((1, 3)))   # N=1 mismatch
