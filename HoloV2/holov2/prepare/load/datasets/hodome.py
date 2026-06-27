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

from ....contracts import RawMotion, SceneSpec, SmplParams
from ..base import register_loader
from ..frames import object_pose_zup
from ..smpl import SMPLX_BODY_JOINTS, build_body_model

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


def _object_mesh(token: str, scaned_dir: Path, cache_dir: Path) -> Path:
    """Extract scaned_object/<token>.tar and return a clean, centroid-centred mesh (cached).

    The supplied decimated ``<token>_face1000.obj`` is a fragmented soup (hundreds of disjoint
    patches -> visible holes), so we use the dense scan ``<token>.obj`` instead (geometry only,
    no texture). object_R/T are calibrated against the decimated mesh's centroid, so we centre
    the dense mesh on THAT centroid to keep the poses aligned."""
    import trimesh

    base = cache_dir / token
    out = base / f"{token}_clean.obj"
    if out.exists():
        return out
    full, face = base / f"{token}.obj", base / f"{token}_face1000.obj"
    if not full.exists():
        tar = Path(scaned_dir) / f"{token}.tar"
        if not tar.exists():
            raise FileNotFoundError(f"HODome object archive not found: {tar}")
        cache_dir.mkdir(parents=True, exist_ok=True)
        with tarfile.open(tar) as t:
            t.extractall(cache_dir)
    src = full if full.exists() else face                       # dense scan preferred
    ref = face if face.exists() else full                       # pose-calibration centroid
    centroid = np.asarray(trimesh.load(str(ref), force="mesh", process=False).vertices,
                          np.float64).mean(0)
    m = trimesh.load(str(src), force="mesh", process=True, skip_materials=True)
    m.merge_vertices()
    m.vertices = np.asarray(m.vertices, np.float64) - centroid
    base.mkdir(parents=True, exist_ok=True)
    m.export(str(out))
    return out


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
        # Demo joints = the first 22 SMPL-X bone positions (Z-up), via the body model's batched FK.
        joints = body.bone_positions(params)[:, :_N_BODY].astype(np.float32)

        root = npz.parent.parent                                    # HODome release root
        obj_npz = root / "object" / f"{npz.stem}.npz"
        cache_dir = Path(spec.cache_dir) / "hodome_meshes" if spec.cache_dir else _MESH_CACHE

        object_poses: tuple[np.ndarray, ...] = ()
        object_mesh_paths: tuple[Path, ...] = ()
        fps = 30.0
        if obj_npz.exists():
            od = np.load(str(obj_npz), allow_pickle=True)
            poses = object_pose_zup(od["object_R"], np.asarray(od["object_T"]).reshape(-1, 3))[:T]
            if "mocap_frame_rate" in od:
                fps = float(np.asarray(od["mocap_frame_rate"]))
            token = npz.stem.split("_", 1)[1] if "_" in npz.stem else npz.stem
            mesh = spec.object_mesh_paths[0] if spec.object_mesh_paths else \
                _object_mesh(token, root / "scaned_object", cache_dir)
            object_poses = (poses,)
            object_mesh_paths = (Path(mesh),)

        return RawMotion(
            joint_pos=joints, joint_names=SMPLX_BODY_JOINTS, fps=fps, source_format="hodome",
            object_poses_raw=object_poses, object_mesh_paths=object_mesh_paths, smpl_params=params,
        )
