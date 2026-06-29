"""Unit test for robot_point_cloud: the correspondence robot side as a K=1 PointCloud, with link
indices REMAPPED from the correspondence link order to the robot FK link order (by name), so that
pose_cloud under the robot's link_transforms reproduces link_pos + R @ offset_local."""
import numpy as np
import pytest
from scipy.spatial.transform import Rotation as R

from src.prepare.contracts import CorrespondenceTable
from src.prepare.point_cloud.correspondence import robot_point_cloud
from src.targets.interaction import pose_cloud


def _corr(link_idx, link_names, offsets):
    m = len(link_idx)
    return CorrespondenceTable(
        smpl_idx=np.arange(m), link_idx=np.asarray(link_idx, np.int64),
        offset_local=np.asarray(offsets, np.float64), link_names=tuple(link_names),
        smpl_sampling_id="test")


def test_remaps_link_order_by_name():
    # correspondence link order != robot FK order; the remap must follow NAMES, not raw indices.
    corr = _corr(link_idx=[0, 1, 0], link_names=("elbow", "wrist"),
                 offsets=[[0.1, 0, 0], [0, 0.2, 0], [0, 0, 0.3]])
    robot_link_names = ("pelvis", "wrist", "knee", "elbow")          # elbow->3, wrist->1
    cloud = robot_point_cloud(corr, robot_link_names)
    assert cloud.n_points == 3 and cloud.n_influences == 1
    assert cloud.parts[:, 0].tolist() == [3, 1, 3]                   # elbow, wrist, elbow in FK order
    assert np.allclose(cloud.weights, 1.0)
    assert np.allclose(cloud.offsets[:, 0, :], corr.offset_local)


def test_posing_reproduces_link_placement():
    corr = _corr(link_idx=[1, 0], link_names=("a", "b"),
                 offsets=[[0.1, 0.2, 0.3], [-0.1, 0.0, 0.05]])
    robot_link_names = ("b", "a")                                    # a->FK 1, b->FK 0
    cloud = robot_point_cloud(corr, robot_link_names)
    rb, tb = R.from_rotvec([0, 0, 0.5]).as_matrix(), np.array([1.0, 0.0, 0.0])
    ra, ta = R.from_rotvec([0.3, 0, 0]).as_matrix(), np.array([0.0, 2.0, 0.0])
    part_rot, part_pos = np.stack([rb, ra]), np.stack([tb, ta])      # FK order [b, a]
    out = pose_cloud(cloud, part_rot, part_pos)
    # link_idx[0]=1 -> link_names[1]="b" -> point 0 on "b"; link_idx[1]=0 -> point 1 on "a".
    assert np.allclose(out[0], rb @ corr.offset_local[0] + tb, atol=1e-6)   # point 0 on "b"
    assert np.allclose(out[1], ra @ corr.offset_local[1] + ta, atol=1e-6)   # point 1 on "a"


def test_missing_link_raises():
    corr = _corr(link_idx=[0], link_names=("ghost",), offsets=[[0, 0, 0]])
    with pytest.raises(ValueError, match="ghost"):
        robot_point_cloud(corr, ("pelvis", "knee"))
