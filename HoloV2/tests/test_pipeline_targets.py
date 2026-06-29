"""process_frame integration (synthetic, torch-free): the assembled FrameTargets carries the per-frame
object world transforms (from frame_pose) — the seam solve needs to re-pose the object channels and
seed the object-variable terms. Exercises the full pose->style->eval->transport->assemble flow on a
fake 22-bone body + analytic plane SDFs (no SMPL, no trimesh, no robot URDF)."""
from pathlib import Path

import numpy as np

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


def _ctx(n_obj):
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
    obj_clouds = tuple(PointCloud(parts=np.zeros((3, 1), np.int64), weights=np.ones((3, 1), np.float32),
                                  offsets=np.zeros((3, 1, 3), np.float32)) for _ in range(n_obj))
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
    g, ctx = _grounded(n_obj=2), _ctx(n_obj=2)
    robot = RobotSpec(name="g1", urdf_path=Path("g1.urdf"), link_names=(), dof=29, height=1.3)
    ft = process_frame(g, ctx, robot, f=1)
    pose = frame_pose(g, f=1)
    assert ft.object_rot.shape == (2, 3, 3) and ft.object_pos.shape == (2, 3)
    assert np.allclose(ft.object_rot, pose.object_rot)
    assert np.allclose(ft.object_pos, pose.object_pos)
    assert np.allclose(ft.object_pos[0], [0.2, 0.3, 0.5])           # the grounded object position
