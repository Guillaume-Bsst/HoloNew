"""Chargeur OMOMO (motion ``.pt`` InterMimic + pickle sujet OMOMO) -> RawMotion.

Les séquences de manipulation d'objet OMOMO proviennent de deux sources :
  - le ``.pt`` InterMimic (``spec.motion_path``) : un tenseur d'état ``(T, 591)`` portant les positions
    de joints par joint SMPL-H et les orientations globales (ordre MuJoCo, Z-up) plus la pose du monde
    de l'objet ;
  - la release OMOMO originale (``spec.dataset_root``) : pickles joblib indexés par nom de séquence avec
    les ``betas``/``gender`` du sujet et ``obj_scale`` par frame, plus les meshes d'objets capturés sous
    ``data/captured_objects/``.

Simplification par rapport au HoloNew précédent : le corps est **SMPL-X** (16 betas), donc nous réutilisons
directement le ``BodyModel`` SMPL-X -- les joints corps+main du .pt (52) se cartographient proprement sur
``body_pose`` + ``left/right_hand_pose`` de SMPL-X (les 3 joints face restent zéro). Aucun modèle de corps
SMPL-H n'est nécessaire. Le .pt stocke les orientations GLOBALES (comme SFU), déjà Z-up (PHC/IsaacGym),
donc nous reconstruisons le ``SmplParams`` local avec le ``local_rotvecs_from_global`` partagé et tranchons
corps/mains hors du layout SMPL-H 52-joints ; ``BodyModel`` ré-applique Q (Y->Z) à ces params. La mise à
l'échelle/l'ancrage ne se font PAS ici -- c'est du ressort de ``prepare/calibration/`` (comme pour SFU/HODome).
"""
from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation as R

from ...contracts import RawMotion, SceneSpec, SmplParams
from ..base import register_loader
from ..smpl import SMPLX_BODY_JOINTS, local_rotvecs_from_global, rest_body_model

_MESH_CACHE = Path(tempfile.gettempdir()) / "holov2_omomo_meshes"

# Layout de colonne du .pt InterMimic (le vecteur d'état per-frame stocké).
_PT_JOINTS = slice(162, 162 + 52 * 3)   # (52,3) positions de joints, ordre MuJoCo, Z-up
_PT_OBJECT = slice(318, 325)            # [tx,ty,tz, qx,qy,qz,qw] pose monde de l'objet
_PT_QUATS = slice(383, 383 + 52 * 4)    # (52,4) orientations globales par joint, ordre MuJoCo, xyzw

# Multiplicateur-droit défaisant la torsion PHC "upright_start" d'InterMimic cuite dans les quats globaux
# stockés (interact2mimic écrit global_rot * Q^-1, xyzw) ; l'appliquer produit les vrais globaux SMPL-X.
_UPRIGHT_FIX_XYZW = np.array([0.5, 0.5, 0.5, 0.5])

# smpl_2_mujoco (de InterAct) : SMPL_2_MUJOCO[mujoco_idx] = smpl_idx. Les tableaux .pt sont en ordre
# MuJoCo ; la dispersion par cet indice place chaque joint dans son slot SMPL-H 52-joint (corps 0-21,
# main gauche 22-36, main droite 37-51).
_SMPL_2_MUJOCO = np.array([
    0, 1, 4, 7, 10, 2, 5, 8, 11, 3, 6, 9, 12, 15, 13, 16, 18, 20,
    22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 14, 17, 19, 21,
    37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51])

# Arbre parent SMPL-H 52-joint (corps 0-21 identique à SMPL-X ; main gauche 22-36 / main droite 37-51
# sont le sous-arbre MANO des poignets 20/21). Transforme les globaux dispersés en locals parent-relatifs ;
# les 15 locals de main sont layout-invariants, donc ils alimentent left/right_hand_pose de SMPL-X
# (slots de main SMPL-X 25-39 / 40-54) inchangés. Validé à <1e-6 m contre les joints .pt.
_SMPLH_PARENTS = np.array(
    [-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19,
     20, 22, 23, 20, 25, 26, 20, 28, 29, 20, 31, 32, 20, 34, 35,
     21, 37, 38, 21, 40, 41, 21, 43, 44, 21, 46, 47, 21, 49, 50], dtype=np.int64)


def _object_token(seq: str) -> str:
    """Nom d'objet = 2e segment '_' de la séquence (sub16_largetable_008 -> largetable)."""
    parts = seq.split("_")
    return parts[1] if len(parts) >= 2 else seq


def _load_pt(path: Path):
    """Lire un .pt InterMimic -> (joints (T,52,3) MuJoCo Z-up, object_pose (T,7) pos-first wxyz,
    quats (T,52,4) MuJoCo wxyz avec la torsion upright-start défaite)."""
    import torch

    d = torch.load(str(path), map_location="cpu").detach().numpy()
    joints = d[:, _PT_JOINTS].reshape(-1, 52, 3).astype(np.float64)
    obj = d[:, _PT_OBJECT]                                       # [tx,ty,tz, qx,qy,qz,qw]
    object_pose = obj[:, [0, 1, 2, 6, 3, 4, 5]].astype(np.float32)   # -> pos-first wxyz
    q_xyzw = d[:, _PT_QUATS].reshape(-1, 52, 4)
    t = q_xyzw.shape[0]
    fixed = (R.from_quat(q_xyzw.reshape(-1, 4)) * R.from_quat(_UPRIGHT_FIX_XYZW)).as_quat()
    quats_wxyz = fixed.reshape(t, 52, 4)[:, :, [3, 0, 1, 2]]
    return joints, object_pose, quats_wxyz


def _subject_meta(dataset_root: Path | None, seq: str):
    """``(betas (16,)|None, gender, obj_scale|None)`` pour ``seq`` du pickle OMOMO (train puis test).
    Se dégrade en ``(None, "neutral", None)`` quand la release / séquence est absente -> shape neutre
    moyenne, taille d'objet native."""
    if dataset_root is None:
        return None, "neutral", None
    import joblib

    for split in ("train", "test"):
        p = Path(dataset_root) / "data" / f"{split}_diffusion_manip_seq_joints24.p"
        if not p.exists():
            continue
        data = joblib.load(str(p))
        for entry in data.values():
            if str(entry.get("seq_name", "")) == seq:
                betas = np.asarray(entry["betas"], np.float32).reshape(-1)
                gender = str(entry.get("gender", "neutral"))
                scale = float(np.asarray(entry["obj_scale"]).mean()) if "obj_scale" in entry else None
                return betas, gender, scale
    return None, "neutral", None


def _object_mesh(token: str, dataset_root: Path | None, scale: float | None, cache_dir: Path):
    """Mesh d'unité capturé -> mesh recentré + mis à l'échelle par séquence dans le frame que les poses
    .pt attendent (centroïde à l'origine), caché. Retourne None quand le mesh ou son échelle manque
    (alors la séquence se dégrade en corps/sol uniquement, comme le fait le builder V1)."""
    if dataset_root is None or scale is None:
        return None
    import trimesh

    captured = Path(dataset_root) / "data" / "captured_objects" / f"{token}_cleaned_simplified.obj"
    if not captured.exists():
        return None
    out = cache_dir / f"{token}_{scale:.6f}.obj"
    if out.exists():
        return out
    m = trimesh.load(str(captured), force="mesh", process=False)
    v = np.asarray(m.vertices, np.float64)
    v = (v - v.mean(0)) * scale
    cache_dir.mkdir(parents=True, exist_ok=True)
    trimesh.Trimesh(vertices=v, faces=np.asarray(m.faces), process=False).export(str(out))
    return out


@register_loader("omomo")
class OmomoLoader:
    """SceneSpec -> RawMotion pour une séquence de manipulation OMOMO/InterMimic (un objet)."""

    def load(self, spec: SceneSpec) -> RawMotion:
        if spec.smpl_model_dir is None:
            raise ValueError("OMOMO needs spec.smpl_model_dir (the SMPL-X model directory)")
        seq = Path(spec.motion_path).stem
        joints_mj, object_pose, quats_mj = _load_pt(Path(spec.motion_path))
        T = joints_mj.shape[0]

        # MuJoCo -> ordre SMPL-H 52-joint (corps 0-21, main gauche 22-36, main droite 37-51).
        joints = np.empty_like(joints_mj); joints[:, _SMPL_2_MUJOCO] = joints_mj
        quats = np.empty_like(quats_mj);   quats[:, _SMPL_2_MUJOCO] = quats_mj

        betas, gender, scale = _subject_meta(spec.dataset_root, seq)
        if betas is None:
            betas = np.zeros(16, np.float32)                        # secours de shape neutre moyenne
        rest = rest_body_model(betas, gender, Path(spec.smpl_model_dir))

        local, transl = local_rotvecs_from_global(
            quats, joints[:, 0], _SMPLH_PARENTS, rest.rest_joints[0])
        params = SmplParams(
            betas=betas, global_orient=local[:, 0],
            body_pose=local[:, 1:22].reshape(T, -1),
            left_hand_pose=local[:, 22:37].reshape(T, -1),
            right_hand_pose=local[:, 37:52].reshape(T, -1),
            transl=transl, gender=gender, model_type="smplx")

        cache_dir = Path(spec.cache_dir) / "omomo_meshes" if spec.cache_dir else _MESH_CACHE
        mesh = spec.object_mesh_paths[0] if spec.object_mesh_paths else \
            _object_mesh(_object_token(seq), spec.dataset_root, scale, cache_dir)
        object_poses = (object_pose,) if mesh is not None else ()
        object_meshes = (Path(mesh),) if mesh is not None else ()

        return RawMotion(
            joint_pos=joints[:, :len(SMPLX_BODY_JOINTS)].astype(np.float32), joint_names=SMPLX_BODY_JOINTS,
            fps=30.0, source_format="omomo", object_poses_raw=object_poses,
            object_mesh_paths=object_meshes, smpl_params=params)
