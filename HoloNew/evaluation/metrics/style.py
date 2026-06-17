"""Style-replication metrics: pelvis-frame orientation and shape fidelity.

Both sub-metrics are computed in the pelvis frame, so they are invariant to the
global heading and translation of the motion — they measure *style* (posture and
limb configuration), not placement. See the design spec for the formulas.
"""
from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation as R


def style_series(rot: np.ndarray, pos: np.ndarray,
                rot_ref: np.ndarray, pos_ref: np.ndarray,
                pelvis_idx: int, tracked: np.ndarray) -> dict[str, np.ndarray]:
    """Per-frame, per-link pelvis-frame orientation / shape error arrays.

    rot, rot_ref: (T, K, 3, 3) per-link world rotations (solved / reference).
    pos, pos_ref: (T, K, 3) per-link world positions. pelvis_idx: pelvis index along K.
    Returns ``orient`` (T, K; geodesic, rad) and ``shape`` (T, K; m), both in the pelvis
    frame (heading/translation invariant). ``compute_style`` is the mean of these over
    the tracked mask, so series and scalar can't drift. ``tracked`` is accepted for a
    symmetric signature but does not gate the per-link arrays (the caller masks).
    """
    B = pelvis_idx
    RBt = np.transpose(rot[:, B], (0, 2, 1))          # (T,3,3)  R_B^T
    RBt_ref = np.transpose(rot_ref[:, B], (0, 2, 1))

    # --- Orientation: R~ = R_B^T R_k, geodesic distance to the reference R~_ref ---
    Rtil = np.einsum("tij,tkjl->tkil", RBt, rot)          # (T,K,3,3)
    Rtil_ref = np.einsum("tij,tkjl->tkil", RBt_ref, rot_ref)
    delta = np.einsum("tkji,tkjl->tkil", Rtil, Rtil_ref)  # R~^T R~_ref
    Tn, Kn = delta.shape[0], delta.shape[1]
    ang = np.linalg.norm(
        R.from_matrix(delta.reshape(-1, 3, 3)).as_rotvec(), axis=-1
    ).reshape(Tn, Kn)                                      # (T,K)

    # --- Shape: keypoints expressed in the pelvis frame ---
    dp = pos - pos[:, B:B + 1, :]                          # (T,K,3)
    dp_ref = pos_ref - pos_ref[:, B:B + 1, :]
    q = np.einsum("tij,tkj->tki", RBt, dp)                 # R_B^T (p_k - p_B)
    q_ref = np.einsum("tij,tkj->tki", RBt_ref, dp_ref)
    shape = np.linalg.norm(q - q_ref, axis=-1)             # (T,K)

    return {"orient": ang, "shape": shape}


def compute_style(rot: np.ndarray, pos: np.ndarray,
                 rot_ref: np.ndarray, pos_ref: np.ndarray,
                 pelvis_idx: int, tracked: np.ndarray) -> dict[str, float]:
    """Pelvis-frame orientation (rad) and shape (m) fidelity, meaned over frames and
    tracked links (reduces ``style_series``)."""
    s = style_series(rot, pos, rot_ref, pos_ref, pelvis_idx, tracked)
    m = np.asarray(tracked, dtype=bool)
    return {
        "style_orient_err": float(np.mean(s["orient"][:, m])),
        "style_shape_err": float(np.mean(s["shape"][:, m])),
    }
