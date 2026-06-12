"""
Preprocessing helpers for holosoma motion retargeting.
"""

from __future__ import annotations

import pickle

import numpy as np

from HoloNew.src.utils import extract_object_first_moving_frame


def calculate_scale_factor(task_name, robot_height):
    """Calculate scale factor based on human height."""
    with open("demo_data/height_dict.pkl", "rb") as f:
        height_dict = pickle.load(f)
    sub_name = task_name.split("_")[0]
    human_height = height_dict[sub_name]
    return robot_height / human_height


def preprocess_motion_data(
    human_joints,
    retargeter,
    foot_names,
    scale=0.714,
    mat_height=0.1,
    object_poses=None,
):
    """
    Preprocess human joints and object poses for retargeting.

    Args:
        human_joints (np.ndarray): Human joint positions.
        object_poses (np.ndarray): Object poses.
        retargeter: Retargeting object with smplh_joint2idx attribute.
        scale (float): Scaling factor.
        normalize_height (bool): Whether to normalize human joint heights.

    Returns:
        tuple: (human_joints_scaled, object_poses_scaled, object_moving_frame_idx).
    """
    # Normalize human joint heights
    toe_indices = [
        retargeter.demo_joints.index(foot_names[0]),
        retargeter.demo_joints.index(foot_names[1]),
    ]
    z_min = human_joints[:, toe_indices, 2].min()
    if z_min >= mat_height:
        # On a mat.
        z_min -= mat_height
    human_joints[:, :, 2] -= z_min

    # Scale human joints
    human_joints = human_joints * scale

    if object_poses is not None:
        object_poses[:, -3:-1] = object_poses[:, -3:-1] * scale
        object_z0 = object_poses[0, -1]
        dz_scale = (object_poses[:, -1] - object_z0) * scale
        object_poses[:, -1] = object_z0 + dz_scale

        object_moving_frame_idx = extract_object_first_moving_frame(object_poses)

        return human_joints, object_poses, object_moving_frame_idx

    return human_joints
