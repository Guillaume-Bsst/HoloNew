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
        object_poses[:] = scale_object_poses_to_center(object_poses, scale)

        object_moving_frame_idx = extract_object_first_moving_frame(object_poses)

        return human_joints, object_poses, object_moving_frame_idx

    return human_joints


def scale_object_poses_to_center(object_poses, scale):
    """Pull object poses toward the world centre, mirroring preprocess_motion_data.

    object_poses layout: [..., x, y, z] (last three entries). XY is scaled toward
    the origin by ``scale``; Z keeps its frame-0 height and scales only the deviation
    from it. Returns a new array; the input is left unchanged.
    """
    out = object_poses.copy()
    out[:, -3:-1] = out[:, -3:-1] * scale
    object_z0 = out[0, -1]
    out[:, -1] = object_z0 + (out[:, -1] - object_z0) * scale
    return out


def compute_holosoma_stages(raw_joints, scale, toe_indices, mapped_indices, mat_height=0.1):
    """Holosoma preprocessing as per-stage skeleton arrays (no mutation).

    Mirrors preprocess_motion_data: ground (drop lowest toe z to 0, less mat_height if
    on a mat) then scale (multiply all joints), then select the mapped joints. Returns
    {Original (T,52,3), Grounded (T,52,3), Scaled (T,52,3), Mapped (T,len(mapped),3)}.
    """
    raw = np.asarray(raw_joints, dtype=float)
    original = raw.copy()
    grounded = ground_to_floor(raw, toe_indices, mat_height)
    scaled = grounded * float(scale)
    mapped = scaled[:, list(mapped_indices)]
    return {"Original": original, "Grounded": grounded, "Scaled": scaled, "Mapped": mapped}


def ground_to_floor(joints, toe_indices, mat_height=0.1):
    """Drop the skeleton uniformly so its lowest toe rests on z=0 (or mat_height above the
    floor when it is already standing on a mat). Returns a new array; input unchanged.

    Shared by compute_holosoma_stages and the GMR stage viewer so both expose the same
    'Grounded' stage (the raw input re-grounded onto the floor)."""
    out = np.asarray(joints, dtype=float).copy()
    z_min = float(out[:, toe_indices, 2].min())
    if z_min >= mat_height:
        z_min -= mat_height
    out[:, :, 2] -= z_min
    return out
