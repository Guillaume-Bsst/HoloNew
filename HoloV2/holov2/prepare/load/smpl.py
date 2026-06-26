"""Concrete ``BodyModel`` (SMPL-X) built from ``SmplParams`` + the SMPL model directory.

Frame convention (single place that knows it): SMPL bodies are native Y-up; the canonical world
is Z-up. ``bone_transforms`` and ``posed_vertices`` return the Z-up WORLD; ``rest_vertices`` and
the bone-rest joints stay in the model's NATIVE rest frame — that's the frame the cloud sampler
expresses its skinning offsets in, and ``bone_transforms`` maps it to Z-up posed in one step.

Per-frame ``bone_transforms`` is pure-numpy FK (no torch forward): one rest forward at build, then
each frame just propagates rotations + joint positions down the kinematic tree. ``posed_vertices``
runs a real SMPL-X forward (offline use: sampling, viz). Ported from the previous HoloNew SMPL-X
posing (correspondence/human_body, data_loaders/hodome).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation as R

from ...contracts import SmplParams

# SMPL native Y-up -> canonical Z-up, as a proper rotation Rx(+90deg): (x,y,z)->(x,-z,y).
_YUP_TO_ZUP = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]])

# Per-joint axis-angle order for the SMPL-X 55-joint tree (matches model.parents[:55]).
_SMPLX_AA = ("global_orient", "body_pose", "jaw_pose", "leye_pose", "reye_pose",
             "left_hand_pose", "right_hand_pose")
_SMPLX_AA_N = {"global_orient": 1, "body_pose": 21, "jaw_pose": 1, "leye_pose": 1,
               "reye_pose": 1, "left_hand_pose": 15, "right_hand_pose": 15}
_N_BONES = 55

# SMPL-X body joints 0..21 — the "demo joints" used by the style treatment (shared by loaders).
SMPLX_BODY_JOINTS: tuple[str, ...] = (
    "Pelvis", "L_Hip", "R_Hip", "Spine1", "L_Knee", "R_Knee", "Spine2", "L_Ankle", "R_Ankle",
    "Spine3", "L_Foot", "R_Foot", "Neck", "L_Collar", "R_Collar", "Head", "L_Shoulder",
    "R_Shoulder", "L_Elbow", "R_Elbow", "L_Wrist", "R_Wrist",
)


def _quat_to_R(quats: np.ndarray, order: str = "wxyz") -> np.ndarray:
    """(..., 4) quaternions -> (..., 3, 3) rotation matrices."""
    q = np.asarray(quats, np.float64)
    if order == "wxyz":
        q = q[..., [1, 2, 3, 0]]
    flat = R.from_quat(q.reshape(-1, 4)).as_matrix()
    return flat.reshape(q.shape[:-1] + (3, 3))


def local_params_from_global(quats_zup: np.ndarray, root_pos_zup: np.ndarray, parents: np.ndarray,
                             j_rest0: np.ndarray, order: str = "wxyz"):
    """Turn per-joint GLOBAL orientations (Z-up) into the local SMPL params a ``BodyModel``
    expects. Local relative rotations are world-frame-invariant, so only the ROOT is rebased to
    the model's native Y-up frame (Q^-1); body joints stay as parent-relative locals. ``transl``
    places the native rest pelvis at the world root after the body model's Q (Y->Z).

    Returns ``(global_orient (T,3), body_pose (T,(J-1)*3), transl (T,3))`` (axis-angle, float32).
    """
    rg = _quat_to_R(quats_zup, order)                                  # (T, J, 3, 3) Z-up global
    qt = _YUP_TO_ZUP.T
    go_native = np.einsum("ij,tjk->tik", qt, rg[:, 0])                 # Q^-1 @ R_root
    global_orient = R.from_matrix(go_native).as_rotvec().astype(np.float32)
    locals_ = []
    for j in range(1, len(parents)):
        rl = np.einsum("tij,tjk->tik", rg[:, parents[j]].transpose(0, 2, 1), rg[:, j])
        locals_.append(R.from_matrix(rl).as_rotvec())
    body_pose = np.concatenate(locals_, axis=1).astype(np.float32)     # (T, (J-1)*3)
    transl = (np.asarray(root_pos_zup, np.float64) @ _YUP_TO_ZUP - j_rest0).astype(np.float32)
    return global_orient, body_pose, transl


def _axis_angle_55(p: SmplParams, t: int) -> np.ndarray:
    """(55, 3) local axis-angle for frame ``t`` (face/eye default to zero if absent)."""
    out = []
    for key in _SMPLX_AA:
        n = _SMPLX_AA_N[key]
        v = getattr(p, key)
        if v is None:
            out.append(np.zeros((n, 3)))
        else:
            out.append(np.asarray(v[t], np.float64).reshape(n, 3))
    return np.concatenate(out, axis=0)


def _global_rotations(aa: np.ndarray, parents: np.ndarray) -> np.ndarray:
    """(J, 3, 3) world rotation per bone = FK of the local axis-angles down the tree."""
    local = R.from_rotvec(aa).as_matrix()
    g = np.empty_like(local)
    for j in range(len(parents)):
        par = int(parents[j])
        g[j] = local[j] if par < 0 else g[par] @ local[j]
    return g


def _posed_joints(g: np.ndarray, j_rest: np.ndarray, parents: np.ndarray, transl: np.ndarray) -> np.ndarray:
    """(J, 3) posed joint positions = FK of the rest joints through ``g``, plus ``transl``."""
    jp = np.empty_like(j_rest)
    for j in range(len(parents)):
        par = int(parents[j])
        jp[j] = j_rest[j] if par < 0 else jp[par] + g[par] @ (j_rest[j] - j_rest[par])
    return jp + transl


class SmplBody:
    """``BodyModel`` for one subject (fixed betas/gender). Build via ``build_body_model``."""

    def __init__(self, params: SmplParams, model_dir: Path) -> None:
        if params.model_type != "smplx":
            raise NotImplementedError(f"SmplBody currently supports smplx, not {params.model_type!r}")
        import smplx
        import torch

        self._torch = torch
        betas = np.asarray(params.betas, np.float32).reshape(-1)
        n_expr = int(np.asarray(params.expression).shape[-1]) if params.expression is not None else 10
        self._model = smplx.SMPLX(model_path=str(model_dir), gender=params.gender, ext="npz",
                                  num_betas=betas.shape[0], num_expression_coeffs=n_expr, use_pca=False)
        self._betas_t = torch.from_numpy(betas[None])
        self.faces: np.ndarray = self._model.faces.astype(np.int64)
        self.parents: np.ndarray = self._model.parents.detach().cpu().numpy()[:_N_BONES]

        # Rest pose (betas, zero pose/transl): native joints (for FK) + native verts (for sampling).
        with torch.no_grad():
            rest = self._model(betas=self._betas_t)
        self._j_rest: np.ndarray = rest.joints[0].detach().cpu().numpy()[:_N_BONES].astype(np.float64)
        self._rest_verts: np.ndarray = rest.vertices[0].detach().cpu().numpy().astype(np.float32)

    @property
    def n_bones(self) -> int:
        return _N_BONES

    @property
    def rest_joints(self) -> np.ndarray:
        """(J_bones, 3) rest joint positions in the model's NATIVE frame (for reconstruction)."""
        return self._j_rest

    def rest_vertices(self, params: SmplParams) -> np.ndarray:
        """(V, 3) rest-pose vertices in the model's NATIVE frame (for cloud sampling)."""
        return self._rest_verts

    def bone_transforms(self, params: SmplParams, t: int) -> tuple[np.ndarray, np.ndarray]:
        """(J,3,3) world rotations and (J,3) world origins at frame ``t`` (Z-up), via pure FK."""
        aa = _axis_angle_55(params, t)
        g_native = _global_rotations(aa, self.parents)
        transl = np.asarray(params.transl[t], np.float64)
        j_posed = _posed_joints(g_native, self._j_rest, self.parents, transl)
        rot_world = _YUP_TO_ZUP @ g_native                       # Q R per bone
        pos_world = j_posed @ _YUP_TO_ZUP.T                      # Y-up -> Z-up
        return rot_world, pos_world

    def posed_vertices(self, params: SmplParams, t: int) -> np.ndarray:
        """(V, 3) world mesh vertices at frame ``t`` (Z-up), via a real SMPL-X forward."""
        torch = self._torch
        kw = {}
        for key in _SMPLX_AA + ("transl", "expression"):
            v = getattr(params, key)
            if v is not None:
                kw[key] = torch.from_numpy(np.asarray(v[t: t + 1], np.float32))
        with torch.no_grad():
            out = self._model(betas=self._betas_t, **kw)
        verts = out.vertices[0].detach().cpu().numpy()
        return (verts @ _YUP_TO_ZUP.T).astype(np.float32)


def build_body_model(params: SmplParams, model_dir: Path) -> SmplBody:
    """Build the ``BodyModel`` for ``params`` (one subject) using the SMPL model at ``model_dir``."""
    return SmplBody(params, Path(model_dir))
