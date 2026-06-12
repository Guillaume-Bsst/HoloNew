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
    obj_quat (T, 4) wxyz / obj_trans (T, 3): the object pose per frame, used as-is from
        the .pt (NOT grounded). The raw capture floats the human a few cm above the
        floor while the object already sits correctly; the human is placed at its
        Grounded pose to undo that float, but the object must stay put — grounding it
        too would push it below the floor by the same z-shift.
    margin: query band half-width passed to ObjectSDF.query.
    """

    human_body: "HumanBody"
    cache: "PointCloudCache"
    object_sdf: "ObjectSDF"
    obj_quat: np.ndarray
    obj_trans: np.ndarray
    margin: float

    def __call__(self, t: int, quats_wxyz: np.ndarray, pelvis_grounded: np.ndarray) -> "ContactField":
        """ContactField (distance + direction per sample) at frame t. Reads only t."""
        world = self.human_body.placed_points(quats_wxyz, pelvis_grounded, self.cache, frame_idx=t)
        local = transform_points_world_to_local(self.obj_quat[t], self.obj_trans[t], world)
        return self.object_sdf.query(local, self.margin)


def build_smplx_ground_probe(task_name, omomo_dir, model_dir, object_sdf,
                             obj_poses, margin, density):
    """Build the probe: load the subject SMPL-X shape and sample its surface once.

    The object pose is used as-is (not grounded): the human is placed at its Grounded
    pose to undo the raw capture's float, but the object already sits correctly, so it
    must not be shifted.

    obj_poses: (T, 7) raw .pt object poses [qw, qx, qy, qz, x, y, z].
    """
    from pathlib import Path

    from ..correspondence.human_body import HumanBody
    from ..correspondence.human_metadata import load_human_metadata

    betas, gender = load_human_metadata(Path(omomo_dir), task_name)
    body = HumanBody(model_dir, betas, gender)
    cache = body.build_point_cloud_cache(density)

    obj_poses = np.asarray(obj_poses, dtype=np.float64)
    return SmplxGroundProbe(body, cache, object_sdf, obj_poses[:, :4], obj_poses[:, 4:7], margin)
