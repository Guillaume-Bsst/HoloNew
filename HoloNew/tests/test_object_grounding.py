"""ground_object_pose: a single constant per-clip z-shift drops the object so its lowest
surface point over the clip rests on z=0 (HODome only). Non-HODome / no-surface = no-op."""
import numpy as np

from HoloNew.src.test_socp.movable import ground_object_pose


def _identity_poses(T, z):
    p = np.zeros((T, 7))
    p[:, 0] = 1.0          # qw (identity rotation)
    p[:, 6] = z            # translate z
    return p


def test_hodome_grounds_lowest_surface_to_zero():
    surface = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], float)
    poses = _identity_poses(3, 0.5)        # lowest surface world z = 0.5
    grounded, shift = ground_object_pose(poses, surface, "hodome")
    assert np.isclose(shift, -0.5)
    # lowest surface point over the clip now rests on z=0 (identity rot => world = local + t)
    zmin = min((surface + grounded[t, 4:7])[:, 2].min() for t in range(3))
    assert np.isclose(zmin, 0.0, atol=1e-9)
    # only z is shifted; rotation + xy untouched
    np.testing.assert_array_equal(grounded[:, :6], poses[:, :6])


def test_non_hodome_is_noop():
    surface = np.array([[0, 0, 0], [0, 0, 1]], float)
    poses = _identity_poses(2, 0.5)
    grounded, shift = ground_object_pose(poses, surface, "omomo")
    assert shift == 0.0
    np.testing.assert_array_equal(grounded, poses)


def test_no_surface_is_noop():
    poses = _identity_poses(2, 0.5)
    grounded, shift = ground_object_pose(poses, None, "hodome")
    assert shift == 0.0
    np.testing.assert_array_equal(grounded, poses)


def test_preserves_input_dtype():
    surface = np.array([[0, 0, 0], [0, 0, 1]], float)
    poses32 = np.zeros((2, 7), dtype=np.float32)
    poses32[:, 0] = 1.0
    poses32[:, 6] = 0.5
    # HODome path (shift applied) and no-op path both keep float32
    grounded_h, _ = ground_object_pose(poses32, surface, "hodome")
    grounded_n, _ = ground_object_pose(poses32, surface, "omomo")
    assert grounded_h.dtype == np.float32
    assert grounded_n.dtype == np.float32


import pytest  # noqa: E402
from pathlib import Path  # noqa: E402

from HoloNew.src.paths import get_path  # noqa: E402

_HODOME = get_path("hodome") / "smplx" / "subject01_baseball.npz"
_SMPLX = get_path("smplx_models") / "smplx"
_HAVE = (_HODOME.exists() and _SMPLX.is_dir()
         and (get_path("hodome") / "object" / "subject01_baseball.npz").exists())


@pytest.mark.skipif(not _HAVE, reason="HODome + SMPL-X assets not present")
def test_object_grounded_in_ground_stage_single_source():
    from HoloNew.examples.robot_retarget import (
        RetargetingConfig, convert_object_poses_to_mujoco_order)
    from HoloNew.src.test_socp.config import TestSocpRetargeterConfig
    from HoloNew.src.test_socp.test_socp import TestSocpRetargeter
    from HoloNew.src.data_loaders.facade import normalize_dataset_cfg
    from scipy.spatial.transform import Rotation as Rot

    cfg = RetargetingConfig(dataset="hodome", motion_name="subject01_baseball",
                            task_type="object_interaction",
                            retargeter=TestSocpRetargeterConfig(floor_contact_margin=0.01))
    normalize_dataset_cfg(cfg)
    rt = TestSocpRetargeter.from_config(cfg)

    # The grounded object lives in the ground stage AND is the single source object pose.
    obj = rt.gmr_stages["floor"]["object_pose"]
    assert obj is rt._obj_poses_raw
    assert rt._obj_ground_shift > 0.0          # a real downward correction was applied

    # The lowest object surface point over the clip rests on z ~ 0.
    osl = np.asarray(rt.object_surface_local, float)
    fsamp = np.unique(np.linspace(0, len(obj) - 1, min(len(obj), 60)).astype(int))
    zmin = min(float((osl @ Rot.from_quat(obj[f, [1, 2, 3, 0]]).as_matrix().T
                      + obj[f, 4:7])[:, 2].min()) for f in fsamp)
    assert abs(zmin) < 1e-6, f"object lowest surface z={zmin:.4f} not grounded"

    # If the MuJoCo drive was built, it derives from the same grounded source.
    if rt._obj_poses_mj is not None:
        np.testing.assert_allclose(
            rt._obj_poses_mj, convert_object_poses_to_mujoco_order(rt._obj_poses_raw))
