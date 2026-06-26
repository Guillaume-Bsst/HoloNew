"""HODome loader: raw SMPL-X params + object R/T -> RawMotion.

Layout of a HODome release (root = the dir holding ``smplx/``, ``object/``, ``scaned_object/``):
    smplx/<subject>_<token>.npz   per-frame SMPL-X params (global_orient, body_pose, hands, ...)
    object/<subject>_<token>.npz  object_R (T,3,3) + object_T (T,3) + mocap_frame_rate
    scaned_object/<token>.tar     the scanned object mesh

``spec.motion_path`` points at the ``smplx/`` npz; the object npz and scanned mesh are derived
from the release layout. SMPL-X is native Y-up; world arrays (joints, object poses) are returned
in the canonical Z-up world, while ``SmplParams`` keeps the native params (``BodyModel`` poses
them into Z-up). Ported from the previous HoloNew ``data_loaders/hodome.py``.
"""
from __future__ import annotations

import tarfile
import tempfile
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation as R

from ...contracts import RawMotion, SceneSpec, SmplParams
from .base import register_loader
from .smpl import build_body_model

# SMPL-X body joints 0..21 (the "demo joints" used by the style treatment).
SMPLX_BODY_JOINTS: tuple[str, ...] = (
    "Pelvis", "L_Hip", "R_Hip", "Spine1", "L_Knee", "R_Knee", "Spine2", "L_Ankle", "R_Ankle",
    "Spine3", "L_Foot", "R_Foot", "Neck", "L_Collar", "R_Collar", "Head", "L_Shoulder",
    "R_Shoulder", "L_Elbow", "R_Elbow", "L_Wrist", "R_Wrist",
)

# Y-up -> Z-up as a proper rotation Rx(+90deg): (x,y,z) -> (x,-z,y). A bare y<->z axis swap is
# a reflection (det -1) that mirrors the body and flips face winding; the rotation preserves it.
_YUP_TO_ZUP = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]])
_MESH_CACHE = Path(tempfile.gettempdir()) / "holov2_hodome_meshes"
_N_BODY = 22  # SMPL-X body joints used as the demo joints


def _betas(d) -> np.ndarray:
    b = np.asarray(d["betas"], np.float32)
    return b[0] if b.ndim > 1 else b


def _smpl_params(d) -> SmplParams:
    """Native SMPL-X params from the npz (BodyModel poses them into the Z-up world)."""
    def a(key):
        return np.asarray(d[key], np.float32)

    return SmplParams(
        betas=_betas(d), global_orient=a("global_orient"), body_pose=a("body_pose"),
        left_hand_pose=a("left_hand_pose"), right_hand_pose=a("right_hand_pose"),
        transl=a("transl"), gender=str(d["gender"]), model_type="smplx",
        jaw_pose=a("jaw_pose"), leye_pose=a("leye_pose"), reye_pose=a("reye_pose"),
        expression=a("expression"),
    )


def _object_poses_zup(obj_npz: Path) -> np.ndarray:
    """object_R (T,3,3) + object_T (T,3) -> (T,7) world pose [x,y,z,qw,qx,qy,qz] in Z-up.

    A rigid world rotation Q LEFT-multiplies both (Q t, Q R); the mesh stays in its native
    local frame. ``object_R/T`` reference the centroid-centred mesh (see ``_object_mesh``)."""
    d = np.load(str(obj_npz), allow_pickle=True)
    rot = np.asarray(d["object_R"], np.float64)                 # (T,3,3)
    trans = np.asarray(d["object_T"], np.float64).reshape(-1, 3)
    trans_z = trans @ _YUP_TO_ZUP.T
    rot_z = _YUP_TO_ZUP @ rot                                   # Q R
    quat_xyzw = R.from_matrix(rot_z).as_quat()                  # (T,4) xyzw
    quat_wxyz = quat_xyzw[:, [3, 0, 1, 2]]
    return np.concatenate([trans_z, quat_wxyz], axis=1).astype(np.float32)  # pos-first


def _object_mesh(token: str, scaned_dir: Path, cache_dir: Path) -> Path:
    """Extract scaned_object/<token>.tar and return a centroid-centred mesh path (cached).

    Prefers the decimated ``<token>_face1000.obj`` (what object_R/T were calibrated against).
    object_R/T are defined w.r.t. the centroid-centred mesh, so we centre it once."""
    import trimesh

    base = cache_dir / token
    raw = base / f"{token}_face1000.obj"
    if not raw.exists():
        raw = base / f"{token}.obj"
    if not raw.exists():
        tar = Path(scaned_dir) / f"{token}.tar"
        if not tar.exists():
            raise FileNotFoundError(f"HODome object archive not found: {tar}")
        cache_dir.mkdir(parents=True, exist_ok=True)
        with tarfile.open(tar) as t:
            t.extractall(cache_dir)
        raw = base / f"{token}_face1000.obj"
        if not raw.exists():
            raw = base / f"{token}.obj"
    centred = raw.with_name(raw.stem + "_centered.obj")
    if not centred.exists():
        m = trimesh.load(str(raw), force="mesh", process=False)
        v = np.asarray(m.vertices, np.float64)
        m.vertices = v - v.mean(0)
        m.export(str(centred))
    return centred


@register_loader("hodome")
class HodomeLoader:
    """SceneSpec -> RawMotion for a HODome smplx sequence (one object when present)."""

    def load(self, spec: SceneSpec) -> RawMotion:
        if spec.smpl_model_dir is None:
            raise ValueError("HODome needs spec.smpl_model_dir (the SMPL-X model directory)")
        npz = Path(spec.motion_path)
        d = np.load(str(npz), allow_pickle=True)

        params = _smpl_params(d)
        body = build_body_model(params, Path(spec.smpl_model_dir))
        T = params.n_frames
        # Demo joints = the first 22 SMPL-X bone positions (Z-up), via the body model's FK.
        joints = np.stack([body.bone_transforms(params, t)[1][:_N_BODY] for t in range(T)]).astype(np.float32)

        root = npz.parent.parent                                    # HODome release root
        obj_npz = root / "object" / f"{npz.stem}.npz"
        cache_dir = Path(spec.cache_dir) / "hodome_meshes" if spec.cache_dir else _MESH_CACHE

        object_poses: tuple[np.ndarray, ...] = ()
        object_mesh_paths: tuple[Path, ...] = ()
        fps = 30.0
        if obj_npz.exists():
            poses = _object_poses_zup(obj_npz)[:T]                  # (T,7) pos-first Z-up
            token = npz.stem.split("_", 1)[1] if "_" in npz.stem else npz.stem
            if spec.object_mesh_paths:
                mesh = spec.object_mesh_paths[0]
            else:
                mesh = _object_mesh(token, root / "scaned_object", cache_dir)
            object_poses = (poses,)
            object_mesh_paths = (Path(mesh),)
            od = np.load(str(obj_npz), allow_pickle=True)
            if "mocap_frame_rate" in od:
                fps = float(np.asarray(od["mocap_frame_rate"]))

        return RawMotion(
            joint_pos=joints, joint_names=SMPLX_BODY_JOINTS, fps=fps, source_format="hodome",
            object_poses_raw=object_poses, object_mesh_paths=object_mesh_paths, smpl_params=params,
        )
