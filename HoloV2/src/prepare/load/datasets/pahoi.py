"""Chargeur PA-HOI (Physics-Aware HOI, capture Noitom) : params SMPL-X directs + trajectoire objet
d'un ``_o.fbx`` -> RawMotion.

Layout d'une séquence (root = ``Mocap_data``) :
    cap_res_bvh_s{1,2}/<seq>/<seq>.npz   params SMPL-X par frame (global_orient, body_pose, lhand/rhand, ...)
    cap_res_fbx/<seq>_o.fbx              mesh objet (proxy) + Lcl Translation/Rotation animés (6-DoF)

``spec.motion_path`` pointe le npz ; le ``_o.fbx`` frère est dérivé du layout. Le npz porte des params
SMPL-X NATIFS (Y-up), donc — contrairement à SFU/OMOMO — aucune reconstruction global->local : on emballe
tel quel (``BodyModel`` les pose en Z-up). L'objet est natif Y-up dans le MÊME monde que le npz (vérifié) :
on réutilise le ``object_pose_zup`` partagé (même ``YUP_TO_ZUP`` que l'humain). Mesh = proxy embarqué
(24 verts), écrit en ``.obj`` local caché (idiome OMOMO/HODome). Aucune mise à l'échelle/ancrage ici
(du ressort de ``prepare/calibration/``).
"""
from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np

from ...contracts import RawMotion, SceneSpec, SmplParams
from ..base import register_loader
from ..fbx import read_object_fbx
from ..frames import object_pose_zup
from ..smpl import SMPLX_BODY_JOINTS, build_body_model

_MESH_CACHE = Path(tempfile.gettempdir()) / "holov2_pahoi_meshes"


def _betas(d) -> np.ndarray:
    b = np.asarray(d["betas"], np.float32)
    return b[0] if b.ndim > 1 else b


def _smpl_params(d) -> SmplParams:
    """Params SMPL-X natifs du ``<seq>.npz`` (BodyModel les pose dans le monde Z-up).

    Spécificités PA-HOI : clés de main ``lhand_pose``/``rhand_pose``, poses stockées ``(T, J, 3)``
    (aplaties en ``(T, J*3)``), pas de champ ``gender`` (-> neutre)."""
    def a(key):
        return np.asarray(d[key], np.float32)

    def flat(key):
        v = a(key)
        return v.reshape(v.shape[0], -1)

    keys = set(d.files)
    gender = str(d["gender"]) if "gender" in keys else "neutral"
    return SmplParams(
        betas=_betas(d), global_orient=flat("global_orient"), body_pose=flat("body_pose"),
        left_hand_pose=flat("lhand_pose"), right_hand_pose=flat("rhand_pose"),
        transl=a("transl"), gender=gender, model_type="smplx",
        jaw_pose=a("jaw_pose") if "jaw_pose" in keys else None,
        leye_pose=a("leye_pose") if "leye_pose" in keys else None,
        reye_pose=a("reye_pose") if "reye_pose" in keys else None,
        expression=a("expression") if "expression" in keys else None,
    )


def _object_fbx_path(motion_path: Path) -> Path:
    """``<...>/cap_res_bvh_s{1,2}/<seq>/<seq>.npz`` -> ``<...>/cap_res_fbx/<seq>_o.fbx`` (frère objet)."""
    seq = motion_path.stem
    mocap_root = motion_path.parent.parent.parent          # remonte <seq>/, cap_res_bvh_sN/, -> Mocap_data
    return mocap_root / "cap_res_fbx" / f"{seq}_o.fbx"


def _write_proxy_obj(vertices: np.ndarray, faces: np.ndarray, name: str, cache_dir: Path) -> Path:
    """Écrire le mesh proxy (repère local, mètres) en ``.obj`` caché et retourner son chemin."""
    import trimesh

    out = cache_dir / f"{name}.obj"
    if out.exists():
        return out
    cache_dir.mkdir(parents=True, exist_ok=True)
    trimesh.Trimesh(vertices=np.asarray(vertices, np.float64),
                    faces=np.asarray(faces, np.int64), process=False).export(str(out))
    return out


@register_loader("pahoi")
class PaHoiLoader:
    """SceneSpec -> RawMotion pour une séquence PA-HOI (un objet)."""

    def load(self, spec: SceneSpec) -> RawMotion:
        if spec.smpl_model_dir is None:
            raise ValueError("PA-HOI needs spec.smpl_model_dir (the SMPL-X model directory)")
        npz = Path(spec.motion_path)
        d = np.load(str(npz), allow_pickle=True)

        params = _smpl_params(d)
        body = build_body_model(params, Path(spec.smpl_model_dir))
        T = params.n_frames
        # Joints de démo = os du corps SMPL-X (Z-up), via la FK batch du modèle de corps.
        joints = body.bone_positions(params)[:, :len(SMPLX_BODY_JOINTS)].astype(np.float32)

        cache_dir = Path(spec.cache_dir) / "pahoi_meshes" if spec.cache_dir else _MESH_CACHE
        obj_fbx = _object_fbx_path(npz)
        object_poses: tuple[np.ndarray, ...] = ()
        object_mesh_paths: tuple[Path, ...] = ()
        fps = 30.0
        if obj_fbx.exists():
            ofbx = read_object_fbx(obj_fbx)
            fps = ofbx.fps
            poses = object_pose_zup(ofbx.rot_native, ofbx.transl_native)[:T]   # (T,7) Z-up, helper partagé
            mesh = spec.object_mesh_paths[0] if spec.object_mesh_paths else \
                _write_proxy_obj(ofbx.vertices, ofbx.faces, npz.stem, cache_dir)
            object_poses = (poses,)
            object_mesh_paths = (Path(mesh),)

        return RawMotion(
            joint_pos=joints, joint_names=SMPLX_BODY_JOINTS, fps=fps, source_format="pahoi",
            object_poses_raw=object_poses, object_mesh_paths=object_mesh_paths, smpl_params=params,
        )
