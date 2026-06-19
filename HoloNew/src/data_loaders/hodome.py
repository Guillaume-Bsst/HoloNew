"""HODome (HODome) loader: raw SMPL-X params -> Z-up joints; object R/T -> poses."""
from __future__ import annotations

import tarfile
import tempfile
from pathlib import Path

import numpy as np
import smplx
import torch
from scipy.spatial.transform import Rotation as R

from HoloNew.src.data_loaders.base import MotionLoader, register_loader

# Disk cache for object meshes extracted from scaned_object/<token>.tar.
_HODOME_MESH_CACHE = Path(tempfile.gettempdir()) / "holonew_hodome_meshes"

# Y-up -> Z-up as a proper ROTATION Rx(+90 deg): (x, y, z) -> (x, -z, y). A bare axis
# SWAP (y<->z) is a reflection (det -1) that mirrors the subject and reverses face
# winding, rendering the SMPL mesh inside-out; the rotation preserves chirality and
# winding. Applied identically to joints, per-joint orientations, the object pose, and
# the mesh vertices so the whole scene stays consistent AND un-mirrored.
_YUP_TO_ZUP = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]])


def _to_zup(points: np.ndarray) -> np.ndarray:
    """Rotate (..., 3) points from the Y-up frame to Z-up (row-vector convention)."""
    return points @ _YUP_TO_ZUP.T


def extract_hodome_object_mesh(token: str, scaned_object_dir: Path,
                              cache_dir: Path | None = None) -> Path:
    """Extract <token>/<token>.obj from scaned_object/<token>.tar into a cache dir
    and return the .obj path. Idempotent (re-uses the cache)."""
    cache_dir = Path(cache_dir) if cache_dir is not None else _HODOME_MESH_CACHE
    out = cache_dir / token / f"{token}.obj"
    if out.exists():
        return out
    tar_path = Path(scaned_object_dir) / f"{token}.tar"
    if not tar_path.exists():
        raise FileNotFoundError(f"HODome object mesh archive not found: {tar_path}")
    cache_dir.mkdir(parents=True, exist_ok=True)
    with tarfile.open(tar_path) as t:
        t.extractall(cache_dir)
    if not out.exists():
        raise FileNotFoundError(f"{token}/{token}.obj not found inside {tar_path}")
    return out


def hodome_fk(npz_path: Path, model_dir: Path) -> tuple[np.ndarray, float]:
    """FK raw SMPL-X params to (T, 22, 3) Z-up joints; return (joints, rest_height_m)."""
    d = np.load(str(npz_path), allow_pickle=True)
    T = d["body_pose"].shape[0]
    model = smplx.SMPLX(model_path=str(model_dir), gender=str(d["gender"]), ext="npz",
                        num_betas=d["betas"].shape[-1], num_expression_coeffs=d["expression"].shape[-1],
                        use_pca=False)
    betas = torch.from_numpy(np.asarray(d["betas"][:1], np.float32)).repeat(T, 1)

    def _t(key):  # full-T pose component from the npz, as float32 tensor
        return torch.from_numpy(np.asarray(d[key], np.float32))

    # Pass every pose component at full batch T — letting hands/jaw/eyes fall back to
    # the model's batch-1 defaults makes the SMPL-X forward size-mismatch on T>1.
    out = model(
        betas=betas,
        global_orient=_t("global_orient"),
        body_pose=_t("body_pose"),
        transl=_t("transl"),
        left_hand_pose=_t("left_hand_pose"),
        right_hand_pose=_t("right_hand_pose"),
        jaw_pose=_t("jaw_pose"),
        leye_pose=_t("leye_pose"),
        reye_pose=_t("reye_pose"),
        expression=_t("expression"),
    )
    joints = out.joints.detach().numpy()[:, :22, :]          # SMPL-X body order
    joints = _to_zup(joints)                                 # dataset is Y-up (rotate, no mirror)

    rest = model(betas=betas[:1])
    rv = rest.vertices.detach().numpy()[0]
    height = float(rv[:, 1].max() - rv[:, 1].min())          # SMPL native Y-up stature
    return joints, height


class HodomeMeshPoser:
    """Per-frame SMPL-X body mesh for HODome, in the Z-up world frame.

    The mesh MUST be posed by a raw SMPL-X forward in the model's native (Y-up) frame,
    then rotated to Z-up on the vertices (exactly how hodome_fk builds the joints).
    Posing instead from the Y->Z conjugated global orientations (placed_verts_smpl)
    re-poses the canonical template and collapses the body. Caches the last frame so
    toggles/redraws are cheap.
    """

    _COMPONENTS = ("global_orient", "body_pose", "transl", "left_hand_pose",
                   "right_hand_pose", "jaw_pose", "leye_pose", "reye_pose", "expression")

    def __init__(self, npz_path: Path, model_dir: Path) -> None:
        d = np.load(str(npz_path), allow_pickle=True)
        self._params = {k: np.asarray(d[k], np.float32) for k in self._COMPONENTS}
        self._betas = np.asarray(d["betas"][:1], np.float32)            # (1, num_betas)
        self.model = smplx.SMPLX(
            model_path=str(model_dir), gender=str(d["gender"]), ext="npz",
            num_betas=self._betas.shape[-1],
            num_expression_coeffs=int(np.asarray(d["expression"]).shape[-1]), use_pca=False)
        self.faces = self.model.faces.astype(np.uint32)
        self._cache_idx: int = -1
        self._cache_verts: np.ndarray | None = None

    def vertices_zup(self, frame: int) -> np.ndarray:
        """Posed body vertices (V,3) for ``frame`` in the Z-up world frame."""
        if frame == self._cache_idx and self._cache_verts is not None:
            return self._cache_verts
        kw = {k: torch.from_numpy(self._params[k][frame:frame + 1]) for k in self._COMPONENTS}
        with torch.no_grad():
            out = self.model(betas=torch.from_numpy(self._betas), **kw)
        verts = out.vertices[0].detach().numpy()
        verts = _to_zup(verts)                                         # native Y-up -> Z-up (rotate)
        self._cache_idx, self._cache_verts = frame, verts
        return verts


def global_orientations_zup(global_orient: np.ndarray, body_pose: np.ndarray) -> np.ndarray:
    """Per-joint global orientations (T, 22, 4) WXYZ in Z-up from raw SMPL-X locals.

    `global_orient` (T,3) is the root axis-angle, `body_pose` (T,63) the 21 body
    joints. Global orientations are built by FK down the SMPL-X tree (reusing the
    AMASS-prep helper), then expressed in Z-up by conjugating each rotation with the
    Y->Z rotation Q (R' = Q R Q^T), consistent with the joints/object/mesh transform.
    """
    from HoloNew.data_utils.prep_amass_smplx_for_rt import (
        _SMPLX_BODY_PARENTS, compute_global_joint_orientations,
    )
    go = np.asarray(global_orient, np.float64).reshape(-1, 1, 3)
    bp = np.asarray(body_pose, np.float64).reshape(go.shape[0], 21, 3)
    aa = np.concatenate([go, bp], axis=1)                       # (T,22,3) axis-angle
    q_yup = compute_global_joint_orientations(aa, _SMPLX_BODY_PARENTS)  # (T,22,4) wxyz, Y-up
    t, j, _ = q_yup.shape
    Rm = R.from_quat(q_yup[..., [1, 2, 3, 0]].reshape(-1, 4)).as_matrix()  # xyzw -> (N,3,3)
    Rz = _YUP_TO_ZUP @ Rm @ _YUP_TO_ZUP.T                        # Q R Q^T, Z-up
    q_xyzw = R.from_matrix(Rz).as_quat().reshape(t, j, 4)
    return q_xyzw[..., [3, 0, 1, 2]].astype(np.float32)          # -> wxyz


def prep_hodome_processed(npz_path: Path, model_dir: Path) -> dict:
    """Raw HODome SMPL-X .npz -> the processed dict the smplx retargeting path expects:
    global_joint_positions (T,22,3) Z-up, global_joint_orientations (T,22,4) WXYZ Z-up,
    height, betas, gender. Mirrors data_utils/prep_amass_smplx_for_rt output keys."""
    d = np.load(str(npz_path), allow_pickle=True)
    joints, height = hodome_fk(Path(npz_path), Path(model_dir))
    quats = global_orientations_zup(d["global_orient"], d["body_pose"])
    return {
        "global_joint_positions": joints.astype(np.float32),
        "global_joint_orientations": quats,
        "height": np.float32(height),
        "betas": np.asarray(d["betas"][0], np.float32).reshape(-1),
        "gender": str(d["gender"]),
    }


def hodome_object_poses(npz_path: Path) -> np.ndarray:
    """Object 6DoF (T,7) [qw,qx,qy,qz,x,y,z] in Z-up from object_R (T,3,3) + object_T.

    HODome stores the object in the same Y-up frame as the raw SMPL-X, so the pose is
    expressed in Z-up to match the (Y->Z rotated) human joints: translation gets the
    Y->Z rotation Q, rotation the conjugation Q R Q^T (same Q as the human)."""
    d = np.load(str(npz_path), allow_pickle=True)
    rot = np.asarray(d["object_R"], np.float64)                  # (T,3,3) Y-up
    trans = np.asarray(d["object_T"], np.float64).reshape(-1, 3)  # (T,3) Y-up
    trans_z = trans @ _YUP_TO_ZUP.T                              # rotate Y-up -> Z-up per row
    rot_z = _YUP_TO_ZUP @ rot @ _YUP_TO_ZUP.T                    # Q R Q^T -> Z-up
    quat_xyzw = R.from_matrix(rot_z).as_quat()                   # (T,4) xyzw
    quat_wxyz = quat_xyzw[:, [3, 0, 1, 2]]
    return np.concatenate([quat_wxyz, trans_z], axis=1)


@register_loader("hodome")
class HoDomeLoader(MotionLoader):
    def load(self, *, model_path, motion_path, obj_path, task_type,
             constants, motion_data_config, smpl_model_dir=None):
        # hodome uses model_path as its SMPL-X body-model dir; smpl_model_dir is unused.
        human_joints, height = hodome_fk(Path(motion_path), Path(model_path))
        smpl_scale = float(constants.ROBOT_HEIGHT) / height

        n = human_joints.shape[0]
        if task_type == "robot_only" or obj_path is None:
            object_poses = np.tile(np.array([[1, 0, 0, 0, 0, 0, 0]]), (n, 1))
        else:
            object_poses = hodome_object_poses(Path(obj_path))[:n]
        return human_joints, object_poses, smpl_scale

    def object_source(self, *, motion_path, obj_path, model_path, task_type,
                      constants, motion_data_config, smpl_model_dir=None):
        return []   # real implementation lands in Task 3
