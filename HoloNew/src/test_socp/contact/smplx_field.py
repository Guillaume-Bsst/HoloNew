"""Online SMPL-X -> object-SDF probe for the TEST-SOCP retargeter.

Causal, frame by frame: at frame t it places the SMPL-X surface point cloud at
that frame's Grounded pose and queries the object SDF, using only frame-t inputs
(per-joint quaternions, grounded pelvis, object pose) plus motion-independent
caches built once at construction. No look-ahead, so it works in an online /
streaming setting (the retargeter advances one frame at a time, with no view of
future frames).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from HoloNew.src.holosoma.interaction_mesh import transform_points_world_to_local


@dataclass
class SmplxGroundProbe:
    """Query the object SDF at the Grounded-pose SMPL-X surface samples, per frame.

    human_body / cache: the SMPL-X body (subject betas) and its rest-pose surface
        samples, built once (motion-independent).
    object_sdf: the object signed-distance field, in the object's rest frame.
    obj_quat_grounded (T, 4) wxyz / obj_trans_grounded (T, 3): the object pose per
        frame, grounded by the same z-shift as the human, so the human samples and
        the object live in one consistent frame.
    margin: query band half-width passed to ObjectSDF.query.
    """

    human_body: "HumanBody"
    cache: "PointCloudCache"
    object_sdf: "ObjectSDF"
    obj_quat_grounded: np.ndarray
    obj_trans_grounded: np.ndarray
    margin: float

    def __call__(self, t: int, quats_wxyz: np.ndarray, pelvis_grounded: np.ndarray) -> "ContactField":
        """ContactField (distance + direction per sample) at frame t. Reads only t."""
        world = self.human_body.placed_points(quats_wxyz, pelvis_grounded, self.cache, frame_idx=t)
        local = transform_points_world_to_local(self.obj_quat_grounded[t], self.obj_trans_grounded[t], world)
        return self.object_sdf.query(local, self.margin)


def build_smplx_ground_probe(task_name, omomo_dir, model_dir, object_sdf,
                             obj_poses, z_ground_offset, margin, density):
    """Build the probe: load the subject SMPL-X shape, sample its surface once, and
    ground the object pose by the same z-shift as the human.

    obj_poses: (T, 7) raw .pt object poses [qw, qx, qy, qz, x, y, z].
    z_ground_offset: the constant z dropped from the raw human to reach Grounded.
    """
    from pathlib import Path

    from ..correspondence.human_body import HumanBody
    from ..correspondence.human_metadata import load_human_metadata

    betas, gender = load_human_metadata(Path(omomo_dir), task_name)
    body = HumanBody(model_dir, betas, gender)
    cache = body.build_point_cloud_cache(density)

    obj_poses = np.asarray(obj_poses, dtype=np.float64)
    obj_quat = obj_poses[:, :4]
    obj_trans = obj_poses[:, 4:7].copy()
    obj_trans[:, 2] -= z_ground_offset       # ground the object like the human
    return SmplxGroundProbe(body, cache, object_sdf, obj_quat, obj_trans, margin)
