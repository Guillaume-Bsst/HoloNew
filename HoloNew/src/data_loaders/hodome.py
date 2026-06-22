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

# Bump when prep_hodome_processed's output format changes so the façade's disk cache is
# rebuilt instead of silently reused. v2 = 55-joint orientations (body+face+both hands),
# enabling hand posing in the contact probe (v1 was 22 body joints only).
PREP_FORMAT_VERSION = 2

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
    """Extract the scanned object mesh from scaned_object/<token>.tar into a cache dir and
    return its path. Idempotent (re-uses the cache).

    Prefers the decimated *_face1000.obj* (1000-face) mesh — the exact file the NeuralDome
    toolbox loads (scripts/hodome_visualize_pyrender.py: ``{obj}_face1000.obj``). Its
    centroid differs from the full ``{token}.obj`` by ~2.6 cm (decimation redistributes
    vertices), and the object_R/T poses are calibrated against THIS mesh's centroid, so
    using the full mesh leaves the object ~2.6 cm off the actor along its long axis. Falls
    back to ``{token}.obj`` when the decimated mesh is absent (e.g. synthetic test tars)."""
    cache_dir = Path(cache_dir) if cache_dir is not None else _HODOME_MESH_CACHE
    face1000 = cache_dir / token / f"{token}_face1000.obj"
    full = cache_dir / token / f"{token}.obj"

    def _resolve() -> Path | None:
        if face1000.exists():
            return face1000
        if full.exists():
            return full
        return None

    out = _resolve()
    if out is not None:
        return out
    tar_path = Path(scaned_object_dir) / f"{token}.tar"
    if not tar_path.exists():
        raise FileNotFoundError(f"HODome object mesh archive not found: {tar_path}")
    cache_dir.mkdir(parents=True, exist_ok=True)
    with tarfile.open(tar_path) as t:
        t.extractall(cache_dir)
    out = _resolve()
    if out is None:
        raise FileNotFoundError(
            f"{token}/{token}_face1000.obj (or {token}.obj) not found inside {tar_path}")
    return out


def centered_object_mesh(mesh_path: Path) -> Path:
    """Path to a centroid-centred copy of the scanned object mesh (written once next to
    the original, idempotent).

    HODome's object_R/T are defined relative to the CENTROID-CENTRED mesh: the NeuralDome
    toolbox does ``obj_verts -= obj_verts.mean(0)`` before ``verts @ R.T + T``
    (scripts/hodome_visualize_pyrender.py). The scanned .obj origin is arbitrary (e.g. at
    the bat knob, ~0.30 m off-centroid), so the solve's object SDF + surface samples (built
    from the mesh path) must use the centred mesh, not the raw scan — otherwise the object
    is placed off the actor by its centroid. Mirrors OMOMO's resolver, which already returns
    a centred mesh, so object_source is consistent across datasets."""
    import trimesh
    mesh_path = Path(mesh_path)
    out = mesh_path.with_name(mesh_path.stem + "_centered.obj")
    if not out.exists():
        m = trimesh.load(str(mesh_path), force="mesh", process=False)
        v = np.asarray(m.vertices, np.float64)
        m.vertices = v - v.mean(0)
        m.export(str(out))
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


def global_orientations_zup(global_orient: np.ndarray, body_pose: np.ndarray,
                            left_hand_pose=None, right_hand_pose=None,
                            jaw_pose=None, leye_pose=None, reye_pose=None) -> np.ndarray:
    """Per-joint global orientations (T, 55, 4) WXYZ in Z-up from raw SMPL-X locals.

    Builds the full 55-joint axis-angle (body + face + both MANO hands), FK down the
    SMPL-X tree (reusing the AMASS-prep helper) in the model's native Y-up frame, then
    expresses it in Z-up. Missing optional components default to zero (identity local).

    hodome_fk rotates the joint POINTS rigidly by Q=_YUP_TO_ZUP (joints @ Q.T), i.e.
    it physically rotates the whole scene. A frame/orientation in that rotated scene is
    obtained by LEFT-multiplying Q (R' = Q R), NOT by conjugation (Q R Q^T): conjugation
    additionally rotates each joint's LOCAL axes by Q^T, which mis-articulates every limb
    relative to the joints/mesh (re-poses the canonical template and collapses the body,
    see HodomeMeshPoser). Left-multiply leaves the articulation (relative-to-parent
    rotations) unchanged and only re-bases the root, matching the rotated joints exactly.
    """
    from HoloNew.data_utils.prep_amass_smplx_for_rt import (
        _SMPLX_PARENTS_55, compute_global_joint_orientations,
    )
    go = np.asarray(global_orient, np.float64).reshape(-1, 1, 3)
    T = go.shape[0]
    bp = np.asarray(body_pose, np.float64).reshape(T, 21, 3)

    def _opt(x, n):  # (T, n, 3) zeros if absent
        return (np.zeros((T, n, 3)) if x is None
                else np.asarray(x, np.float64).reshape(T, n, 3))

    jaw = _opt(jaw_pose, 1); leye = _opt(leye_pose, 1); reye = _opt(reye_pose, 1)
    lh = _opt(left_hand_pose, 15); rh = _opt(right_hand_pose, 15)
    aa = np.concatenate([go, bp, jaw, leye, reye, lh, rh], axis=1)   # (T,55,3) axis-angle
    q_yup = compute_global_joint_orientations(aa, _SMPLX_PARENTS_55)  # (T,55,4) wxyz, Y-up
    t, j, _ = q_yup.shape
    Rm = R.from_quat(q_yup[..., [1, 2, 3, 0]].reshape(-1, 4)).as_matrix()  # xyzw -> (N,3,3)
    Rz = _YUP_TO_ZUP @ Rm                                        # Q R (world rotation), Z-up
    q_xyzw = R.from_matrix(Rz).as_quat().reshape(t, j, 4)
    return q_xyzw[..., [3, 0, 1, 2]].astype(np.float32)          # -> wxyz


def prep_hodome_processed(npz_path: Path, model_dir: Path) -> dict:
    """Raw HODome SMPL-X .npz -> the processed dict the smplx retargeting path expects:
    global_joint_positions (T,22,3) Z-up, global_joint_orientations (T,55,4) WXYZ Z-up
    (body + face + both MANO hands, for the contact probe), height, betas, gender.
    Mirrors data_utils/prep_amass_smplx_for_rt output keys."""
    d = np.load(str(npz_path), allow_pickle=True)
    joints, height = hodome_fk(Path(npz_path), Path(model_dir))
    quats = global_orientations_zup(
        d["global_orient"], d["body_pose"],
        left_hand_pose=d["left_hand_pose"], right_hand_pose=d["right_hand_pose"],
        jaw_pose=d["jaw_pose"], leye_pose=d["leye_pose"], reye_pose=d["reye_pose"])
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
    expressed in Z-up to match the (Y->Z rotated) human joints. The object mesh is used
    in its NATIVE local frame (no Y->Z swap on the vertices), so a rigid world rotation
    Q=_YUP_TO_ZUP LEFT-multiplies both the translation (Q t) and the rotation (Q R) —
    NOT a conjugation (Q R Q^T), which would extra-rotate the object's local axes and
    mis-orient it (same reasoning as the human global_orientations_zup)."""
    d = np.load(str(npz_path), allow_pickle=True)
    rot = np.asarray(d["object_R"], np.float64)                  # (T,3,3) Y-up
    trans = np.asarray(d["object_T"], np.float64).reshape(-1, 3)  # (T,3) Y-up
    trans_z = trans @ _YUP_TO_ZUP.T                              # rotate Y-up -> Z-up per row
    rot_z = _YUP_TO_ZUP @ rot                                    # Q R (world rotation) -> Z-up
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
        if task_type == "robot_only" or obj_path is None:
            return []
        from HoloNew.src.data_loaders.base import ObjectSource
        stem = Path(obj_path).stem
        token = stem.split("_", 1)[1] if "_" in stem else stem
        scaned = Path(obj_path).parent.parent / "scaned_object"
        # Centred mesh: object_R/T reference the centroid-centred frame (see
        # centered_object_mesh), so the solve's SDF + surface samples must use it.
        mesh_path = centered_object_mesh(extract_hodome_object_mesh(token, scaned))
        poses = hodome_object_poses(Path(obj_path))    # (T,7) Z-up
        return [ObjectSource(mesh_path=Path(mesh_path), poses_raw=poses)]
