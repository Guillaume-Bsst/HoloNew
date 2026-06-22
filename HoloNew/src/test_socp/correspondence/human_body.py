# src/test_pipe_retargeting/test_pipe_retargeting/human/body.py
"""SMPL-X body model wrapper shared by run() (floor re-grounding) and the
HumanMeshPanel (rendering).

The .pt motion stores per-joint global quaternions in MuJoCo order, already with
intermimic's upright_start twist undone (see human.motion.load_pt), i.e. true SMPL-X
global orientations in the Z-up world frame. This class turns one frame of those
quaternions into a posed SMPL-X mesh placed in the world (pelvis snapped onto the
MuJoCo pelvis), and derives the single floor offset that re-grounds the human.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# smpl_2_mujoco mapping (from InterAct, smpl_mujoco.py). Indexed by MuJoCo joint
# position, returns the SMPL index: SMPL_2_MUJOCO[mujoco_idx] = smpl_idx. The .pt
# quats are stored in MuJoCo order, so we scatter each into its SMPL slot.
SMPL_2_MUJOCO = [
    0, 1, 4, 7, 10, 2, 5, 8, 11, 3, 6, 9, 12, 15, 13, 16, 18, 20,
    22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 14, 17, 19, 21,
    37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51,
]
# Same mapping as a fixed index array, for the vectorized scatter in placed_verts.
_SMPL_IDX = np.asarray(SMPL_2_MUJOCO)

# Verified SMPL-H 52-joint parent tree (body identical to SMPL-X; hands are the same
# MANO sub-tree, left 22:37 / right 37:52). Used to pose OMOMO hands, whose .pt quats
# are SMPL-H-ordered, with the correct parents (the SMPL-X model's parents place the
# 3 face joints at 22:25, which would misalign the hand chains).
SMPLH_PARENTS = np.array(
    [-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19,
     20, 22, 23, 20, 25, 26, 20, 28, 29, 20, 31, 32, 20, 34, 35,
     21, 37, 38, 21, 40, 41, 21, 43, 44, 21, 46, 47, 21, 49, 50], dtype=np.int64)


@dataclass(frozen=True)
class PointCloudCache:
    """Stable-identity surface samples for the SMPL-X body.

    Sampled once on the rest-pose mesh; each point is fixed as a barycentric
    location on one triangle. Posing the body (placed_points) moves the points
    while keeping their identity, so point i is always the same body location —
    the seam a future human→G1 optimal-transport coupling plugs into.
    """

    tri_idx: np.ndarray  # (N,)   triangle index per point
    bary: np.ndarray     # (N, 3) barycentric weights, rows sum to 1


class HumanBody:
    """Loaded SMPL-X model + posing helpers. Build once, reuse everywhere."""

    def __init__(self, model_dir: str, betas: np.ndarray | None, gender: str | None) -> None:
        import smplx
        import torch

        self._torch = torch
        num_betas = int(betas.shape[0]) if betas is not None else 10
        self.model = smplx.create(
            model_dir, model_type="smplx", gender=gender or "neutral",
            use_pca=False, flat_hand_mean=True, num_betas=num_betas, batch_size=1,
        )
        self.parents = self.model.parents.detach().cpu().numpy()
        self.faces = self.model.faces.astype(np.uint32)
        self._betas = (
            torch.from_numpy(np.asarray(betas, dtype=np.float32)[None])
            if betas is not None else None
        )
        # Per-frame cache to avoid redundant SMPL-X forward passes.
        self._cache_idx: int = -1
        self._cache_verts: np.ndarray | None = None

    def _forward_posed(self, q_smpl_xyzw, parents, pelvis_target,
                       left_hand=None, right_hand=None) -> np.ndarray:
        """Posed SMPL-X vertices (V,3) for one frame from per-joint GLOBAL rotations.

        q_smpl_xyzw: (J,4) xyzw global per-joint rotations.
        parents: (J,) parent index per joint (root = -1), in the SAME order as q.
        left_hand / right_hand: (start, stop) slices selecting the 15 left/right hand
            joints in q, or None to leave that hand at the model default.
        Converts globals to parent-relative locals, then runs the SMPL-X forward with
        global_orient + body_pose + (optional) left/right hand pose, and snaps the
        pelvis onto pelvis_target.
        """
        from scipy.spatial.transform import Rotation as R
        torch = self._torch
        q = np.asarray(q_smpl_xyzw, dtype=float)
        norms = np.linalg.norm(q, axis=1, keepdims=True)
        q = np.where(norms > 1e-6, q / norms, np.array([0, 0, 0, 1.0]))
        global_rots = R.from_quat(q).as_matrix()                      # (J,3,3)
        par = np.asarray(parents)
        rel = np.matmul(np.transpose(global_rots[par], (0, 2, 1)), global_rots)
        rel[par == -1] = global_rots[par == -1]
        rotvec = R.from_matrix(rel).as_rotvec()                       # (J,3)
        kw = dict(
            global_orient=torch.from_numpy(rotvec[0]).float().view(1, 3),
            body_pose=torch.from_numpy(rotvec[1:22]).float().view(1, -1),
            betas=self._betas, return_verts=True, return_joints=True,
        )
        if left_hand is not None:
            s0, s1 = left_hand
            kw["left_hand_pose"] = torch.from_numpy(rotvec[s0:s1]).float().view(1, -1)
        if right_hand is not None:
            s0, s1 = right_hand
            kw["right_hand_pose"] = torch.from_numpy(rotvec[s0:s1]).float().view(1, -1)
        with torch.no_grad():
            output = self.model(**kw)
        verts = output.vertices[0].detach().cpu().numpy()
        pelvis = output.joints[0, 0].detach().cpu().numpy()
        return verts - pelvis + pelvis_target

    def placed_verts(self, quats_wxyz: np.ndarray, pelvis_target: np.ndarray, frame_idx: int | None = None) -> np.ndarray:
        """Posed SMPL-X vertices (V,3) for one frame, in the Z-up world frame.

        quats_wxyz: (52,4) per-joint global orientations in MuJoCo order (wxyz), SMPL-H
            joint layout (body 0-21, left hand 22-36, right hand 37-51).
        pelvis_target: (3,) world position to snap the SMPL-X pelvis onto.
        frame_idx: optional cache key; a repeat call with the same value returns the cache.
        """
        if frame_idx is not None and frame_idx == self._cache_idx:
            assert self._cache_verts is not None
            return self._cache_verts

        q_mj_xyzw = quats_wxyz[:, [1, 2, 3, 0]]
        # Scatter each MuJoCo-order quat into its SMPL slot. SMPL_2_MUJOCO is a full
        # permutation of 0..51, so every slot is written (no identity remains).
        q_smpl_xyzw = np.zeros((52, 4))
        q_smpl_xyzw[_SMPL_IDX] = q_mj_xyzw

        # SMPL-H parents + SMPL-H hand slots (the .pt quats are SMPL-H-ordered).
        out = self._forward_posed(
            q_smpl_xyzw, SMPLH_PARENTS, pelvis_target,
            left_hand=(22, 37), right_hand=(37, 52))
        if frame_idx is not None:
            self._cache_idx = frame_idx
            self._cache_verts = out
        return out

    def rest_verts(self) -> np.ndarray:
        """SMPL-X vertices (V,3) in the rest pose (subject betas, zero pose)."""
        torch = self._torch
        with torch.no_grad():
            output = self.model(betas=self._betas, return_verts=True)
        return output.vertices[0].detach().cpu().numpy()

    def build_point_cloud_cache(self, density: float) -> PointCloudCache:
        """Sample ~density·area points on the rest-pose surface (built once).

        Records each point as (triangle, barycentric weights) so placed_points
        can carry it through any pose. density is points per m².
        """
        import trimesh

        rest = trimesh.Trimesh(
            vertices=self.rest_verts(), faces=self.faces, process=False
        )
        num_pts = max(1, int(rest.area * density))
        pts, tri_idx = trimesh.sample.sample_surface_even(rest, num_pts)
        bary = trimesh.triangles.points_to_barycentric(
            rest.triangles[tri_idx], pts
        )
        return PointCloudCache(
            tri_idx=tri_idx.astype(np.int64),
            bary=bary.astype(np.float32),
        )

    def placed_verts_smpl(self, quats_wxyz_22: np.ndarray, pelvis_target: np.ndarray,
                          frame_idx: int | None = None) -> np.ndarray:
        """Posed SMPL-X vertices for one frame from SMPL-X-order global orientations.

        Input is the per-joint GLOBAL orientations already in native SMPL-X order (the
        prep output), so no MuJoCo scatter is needed. Accepts J in {22, 55}:
          - J == 22: body only; hands stay at the model default (legacy npz).
          - J >= 55: hands posed from SMPL-X slots 25:40 (left) / 40:55 (right).
        Caches by frame_idx the same way as placed_verts (shared cache slot).
        """
        if frame_idx is not None and frame_idx == self._cache_idx:
            assert self._cache_verts is not None
            return self._cache_verts

        q_in = np.asarray(quats_wxyz_22)
        J = q_in.shape[0]
        q_smpl_xyzw = np.zeros((max(J, 22), 4))
        q_smpl_xyzw[:, 3] = 1.0                                   # identity default
        q_smpl_xyzw[:J] = q_in[:, [1, 2, 3, 0]]
        if J >= 55:
            parents = self.parents[:55]
            left_hand, right_hand = (25, 40), (40, 55)
        else:
            parents = self.parents[:max(J, 22)]
            left_hand = right_hand = None        # 22-joint legacy: hands at default
        out = self._forward_posed(q_smpl_xyzw, parents, pelvis_target,
                                  left_hand=left_hand, right_hand=right_hand)
        if frame_idx is not None:
            self._cache_idx = frame_idx
            self._cache_verts = out
        return out

    def placed_points(
        self,
        quats_wxyz: np.ndarray,
        pelvis_target: np.ndarray,
        cache: PointCloudCache,
        frame_idx: int | None = None,
        smpl_order: bool = False,
    ) -> np.ndarray:
        """Posed surface samples (N,3) for one frame, in the Z-up world frame.

        Reuses placed_verts so SMPL-X skinning is applied once; each cached
        point is the barycentric blend of its triangle's posed vertices.
        ``smpl_order=True`` selects placed_verts_smpl (AMASS 22 SMPL-order joints).
        """
        if smpl_order:
            verts = self.placed_verts_smpl(quats_wxyz, pelvis_target, frame_idx=frame_idx)
        else:
            verts = self.placed_verts(quats_wxyz, pelvis_target, frame_idx=frame_idx)
        tri_verts = verts[self.faces[cache.tri_idx]]            # (N, 3, 3)
        return np.einsum("nij,ni->nj", tri_verts, cache.bary).astype(np.float32)

    def floor_offset(self, quats_wxyz: np.ndarray, joints: np.ndarray) -> float:
        """Single global z-drop = median over frames of each frame's lowest sole
        vertex. The raw human floats a near-constant few cm above the object's
        floor (per-frame lowest sole has cm-scale mean but only mm-scale spread),
        so the median centres the nearly-stationary feet on z=0 while staying
        robust to outlier frames (a deep crouch) and to stray low vertices, which
        affect at most a minority of frames.

        quats_wxyz: (T,52,4) MuJoCo-order wxyz. joints: (T,52,3) raw MuJoCo joints.
        """
        T = quats_wxyz.shape[0]
        mins = np.array([
            float(self.placed_verts(quats_wxyz[t], joints[t, 0])[:, 2].min())
            for t in range(T)
        ])
        return float(np.median(mins))
