"""Chargeur HODome : params SMPL-X bruts + object R/T -> RawMotion.

Layout d'une release HODome (root = le répertoire contenant ``smplx/``, ``object/``, ``scaned_object/``) :
    smplx/<subject>_<token>.npz   params SMPL-X par frame (global_orient, body_pose, hands, ...)
    object/<subject>_<token>.npz  object_R (T,3,3) + object_T (T,3) + mocap_frame_rate
    scaned_object/<token>.tar     le mesh d'objet scanné

``spec.motion_path`` pointe vers le npz ``smplx/`` ; le npz d'objet et le mesh scanné sont dérivés
du layout de la release. SMPL-X est Y-up natif ; les tableaux du monde (joints, poses d'objets) sont
retournés dans le monde Z-up canonique, tandis que ``SmplParams`` garde les params natifs (``BodyModel``
les pose en Z-up). Porté du ``data_loaders/hodome.py`` HoloNew précédent.
"""
from __future__ import annotations

import tarfile
import tempfile
from pathlib import Path

import numpy as np

from ...contracts import RawMotion, SceneSpec, SmplParams
from ..base import register_loader
from ..frames import object_pose_zup
from ..smpl import SMPLX_BODY_JOINTS, build_body_model

_MESH_CACHE = Path(tempfile.gettempdir()) / "holov2_hodome_meshes"


def _betas(d) -> np.ndarray:
    b = np.asarray(d["betas"], np.float32)
    return b[0] if b.ndim > 1 else b


def _smpl_params(d) -> SmplParams:
    """Params SMPL-X natifs du npz (BodyModel les pose dans le monde Z-up)."""
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
    """Extraire scaned_object/<token>.tar et retourner un mesh propre, centré au centroïde (caché).

    Le ``<token>_face1000.obj`` décimé fourni est une soupe fragmentée (centaines de patches
    disjoints -> trous visibles), donc nous utilisons plutôt le scan dense ``<token>.obj``
    (géométrie uniquement, pas de texture). object_R/T sont calibrés contre le centroïde du
    mesh décimé, donc nous centrons le mesh dense sur CE centroïde pour garder les poses alignées."""
    import trimesh

    base = cache_dir / token
    out = base / f"{token}_clean.obj"
    if out.exists():
        return out
    full, face = base / f"{token}.obj", base / f"{token}_face1000.obj"
    if not full.exists():
        tar = Path(scaned_dir) / f"{token}.tar"
        if not tar.exists():
            raise FileNotFoundError(f"Archive d'objet HODome non trouvée : {tar}")
        cache_dir.mkdir(parents=True, exist_ok=True)
        with tarfile.open(tar) as t:
            t.extractall(cache_dir)
    src = full if full.exists() else face                       # scan dense préféré
    ref = face if face.exists() else full                       # centroïde de calibration de pose
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
    """SceneSpec -> RawMotion pour une séquence HODome smplx (un objet quand présent)."""

    def load(self, spec: SceneSpec) -> RawMotion:
        if spec.smpl_model_dir is None:
            raise ValueError("HODome needs spec.smpl_model_dir (the SMPL-X model directory)")
        npz = Path(spec.motion_path)
        d = np.load(str(npz), allow_pickle=True)

        params = _smpl_params(d)
        body = build_body_model(params, Path(spec.smpl_model_dir))
        T = params.n_frames
        # Joints de démo = les premières positions d'os du corps SMPL-X (Z-up), via la FK en batch du modèle de corps.
        joints = body.bone_positions(params)[:, :len(SMPLX_BODY_JOINTS)].astype(np.float32)

        root = npz.parent.parent                                    # racine release HODome
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
