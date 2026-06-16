"""SMPL/SMPL-X true center-of-mass reference (replaces the pelvis CoM proxy in W^c).

Two stages, split exactly along the online-causality line:

  calibrate_smpl_com(human_body)   -- ONE-TIME, MORPHOLOGY-ONLY (the betas/shape).
      Partitions the body by its skinning weights and precomputes, per part j:
      mass m_j (= Σ_v lbs_weights[v,j]) and the part rest-CoM offset com_local_j
      (skinning-weighted vertex centroid minus the rest joint). Looks at NO motion.

  smpl_com_from_pose(calib, quats_wxyz, root_pos)   -- PER FRAME, CAUSAL, cheap.
      Rigid FK over the ~55 joints from the frame's global body orientations, then
      CoM = Σ_j m_j (R_j · com_local_j + t_j) / M. No mesh posing, no torch — just
      ~55 rigid transforms. Uses only the current frame's pose.

Exactness: LBS is linear in the joint transforms, so this per-part aggregation equals
the posed-mesh centroid up to the pose blend-shapes (~7 mm), which is negligible for a
CoM *reference* (and its acceleration). The CoM sits ~0.38 m from the pelvis joint, so
this fixes the pelvis proxy that made W^c fight every limb motion.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial.transform import Rotation as _R


@dataclass(frozen=True)
class SmplComCalib:
    masses: np.ndarray      # (J,) per-part mass = Σ_v skinning_weight
    com_local: np.ndarray   # (J, 3) part rest-CoM in the joint's rest frame
    rest_J: np.ndarray      # (J, 3) rest joint positions
    parents: np.ndarray     # (J,) kinematic parents (-1 for root)


def calibrate_smpl_com(human_body) -> SmplComCalib:
    """Morphology-only calibration (no motion). Needs human_body.model (smplx) + betas."""
    import torch
    model = human_body.model
    W = model.lbs_weights.detach().cpu().numpy()                 # (V, J)
    parents = model.parents.detach().cpu().numpy().astype(int)   # (J,)
    with torch.no_grad():
        rest = model(betas=human_body._betas, return_verts=True, return_joints=True)
    rest_v = rest.vertices[0].detach().cpu().numpy()             # (V, 3)
    J = W.shape[1]
    rest_J = rest.joints[0].detach().cpu().numpy()[:J]           # (J, 3)
    masses = W.sum(0)                                            # (J,)
    rest_com = (W[:, :, None] * rest_v[:, None, :]).sum(0) / masses[:, None]
    com_local = rest_com - rest_J                               # (J, 3)
    return SmplComCalib(masses=masses, com_local=com_local, rest_J=rest_J, parents=parents)


def smpl_com_from_pose(calib: SmplComCalib, quats_wxyz: np.ndarray,
                       root_pos: np.ndarray) -> np.ndarray:
    """World CoM at one frame from the body joints' GLOBAL orientations (wxyz, the first
    ~22 are the SMPL body joints; the rest default to identity → inherit their parent via
    FK) and the world root (pelvis) position. Pure numpy, ~55 rigid transforms.
    """
    m, com_local, rest_J, parents = calib.masses, calib.com_local, calib.rest_J, calib.parents
    J = parents.shape[0]
    # Global rotations from the provided body quats (identity beyond what's given).
    qx = np.zeros((J, 4)); qx[:, 3] = 1.0
    n = min(quats_wxyz.shape[0], J, 22)
    qx[:n] = np.asarray(quats_wxyz)[:n][:, [1, 2, 3, 0]]         # wxyz -> xyzw
    qx /= (np.linalg.norm(qx, axis=1, keepdims=True) + 1e-12)
    Rg = _R.from_quat(qx).as_matrix()                            # (J,3,3) provided globals
    rel = np.matmul(np.transpose(Rg[parents], (0, 2, 1)), Rg)    # local = R_parent^T R_global
    rel[parents == -1] = Rg[parents == -1]
    # Rigid FK: recover global rotation + world joint position for all joints.
    Rglob = np.zeros_like(Rg)
    t = np.zeros((J, 3))
    for j in range(J):
        p = parents[j]
        if p == -1:
            Rglob[j] = rel[j]
            t[j] = root_pos
        else:
            Rglob[j] = Rglob[p] @ rel[j]
            t[j] = t[p] + Rglob[p] @ (rest_J[j] - rest_J[p])
    com_world = np.einsum('jab,jb->ja', Rglob, com_local) + t   # (J,3) part CoMs in world
    return (m[:, None] * com_world).sum(0) / m.sum()
