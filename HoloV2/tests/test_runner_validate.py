"""Unit test for runner._validate's robot_cloud invariant (synthetic, no data): the robot cloud must
carry the SAME M points as the correspondence, else the transport gather and the online re-eval would
disagree on the point set."""
import numpy as np
import pytest

from src.prepare.contracts import (Calibration, Channel, CorrespondenceTable, GroundedScene,
                                    PointCloud)
from src.prepare.sdf.build import build_plane_sdf
from src.prepare.runner import _validate


def _ground_channel():
    sdf = build_plane_sdf([-0.5, -0.5], [0.5, 0.5], spacing=0.1, margin=0.1, name="ground")
    return Channel("ground", None, sdf)


def _grounded0():
    return GroundedScene(joint_pos=np.zeros((1, 1, 3), np.float32), joint_names=("a",),
                         object_poses=(), object_mesh_paths=(),
                         calibration=Calibration(0.0, 0.0, np.eye(4)), fps=30.0,
                         smpl_params=None, body=None)


def _cloud(n, sid=""):
    return PointCloud(parts=np.zeros((n, 1), np.int64), weights=np.ones((n, 1), np.float32),
                      offsets=np.zeros((n, 1, 3), np.float32), sampling_id=sid)


def _corr(m):
    return CorrespondenceTable(smpl_idx=np.arange(m), link_idx=np.zeros(m, np.int64),
                               offset_local=np.zeros((m, 3)), link_names=("root",),
                               smpl_sampling_id="s")


def test_validate_accepts_matching_robot_cloud():
    _validate(_grounded0(), (_ground_channel(),), _cloud(20, "s"), (), _corr(7), _cloud(7))  # no raise


def test_validate_rejects_robot_cloud_point_count_mismatch():
    with pytest.raises(ValueError, match="robot_cloud"):
        _validate(_grounded0(), (_ground_channel(),), _cloud(20, "s"), (), _corr(7), _cloud(6))
