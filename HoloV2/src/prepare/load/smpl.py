"""``BodyModel`` concret (SMPL-X) construit à partir de ``SmplParams`` + le répertoire du modèle SMPL.

Convention de frame (unique lieu qui le sait) : les corps SMPL sont Y-up natifs ; le monde canonique
est Z-up. ``bone_transforms`` et ``posed_vertices`` retournent le MONDE Z-up ; ``rest_vertices`` et
les joints bone-rest restent dans le frame NATIF du modèle — c'est le frame dans lequel l'échantillonneur
de nuage exprime ses décalages de skinning, et ``bone_transforms`` le cartographie en Z-up posé en une
étape.

``bone_transforms`` par frame est une FK pure-numpy (pas de torch forward) : une forward rest à la
construction, puis chaque frame propage simplement les rotations + positions de joints en bas de l'arbre
cinématique. ``posed_vertices`` exécute un vrai forward SMPL-X (usage hors ligne : échantillonnage, viz).
Porté du posage SMPL-X HoloNew précédent (correspondence/human_body, data_loaders/hodome).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation as R

from ..contracts import SmplParams
from .frames import YUP_TO_ZUP

# Ordre axis-angle par joint pour l'arbre 55-joints SMPL-X (correspond à model.parents[:55]).
_SMPLX_AA = ("global_orient", "body_pose", "jaw_pose", "leye_pose", "reye_pose",
             "left_hand_pose", "right_hand_pose")
_SMPLX_AA_N = {"global_orient": 1, "body_pose": 21, "jaw_pose": 1, "leye_pose": 1,
               "reye_pose": 1, "left_hand_pose": 15, "right_hand_pose": 15}
_N_BONES = 55

# Joints du corps SMPL-X 0..21 — les « joints de démo » utilisés par le traitement de style (partagés par les chargeurs).
SMPLX_BODY_JOINTS: tuple[str, ...] = (
    "Pelvis", "L_Hip", "R_Hip", "Spine1", "L_Knee", "R_Knee", "Spine2", "L_Ankle", "R_Ankle",
    "Spine3", "L_Foot", "R_Foot", "Neck", "L_Collar", "R_Collar", "Head", "L_Shoulder",
    "R_Shoulder", "L_Elbow", "R_Elbow", "L_Wrist", "R_Wrist",
)


def _quat_to_R(quats: np.ndarray) -> np.ndarray:
    """(..., 4) quaternions wxyz -> (..., 3, 3) matrices de rotation."""
    q = np.asarray(quats, np.float64)
    q = q[..., [1, 2, 3, 0]]                                           # wxyz -> xyzw
    flat = R.from_quat(q.reshape(-1, 4)).as_matrix()
    return flat.reshape(q.shape[:-1] + (3, 3))


def local_rotvecs_from_global(quats_zup: np.ndarray, root_pos_zup: np.ndarray, parents: np.ndarray,
                              j_rest0: np.ndarray):
    """Transforme les orientations GLOBALES par joint (Z-up) en axis-angles LOCAL par joint + translation
    que ``BodyModel`` attend. Les rotations relatives au parent sont invariantes du frame-monde, donc seule
    la RACINE est rebasée sur le frame Y-up natif du modèle (Q^-1) ; le reste reste parent-relative locals.
    ``transl`` place la racine rest native à la racine-monde après le Q du modèle de corps (Y->Z).

    Les datasets qui stockent les globales tranchent le ``local`` retourné selon leur layout : SFU garde
    seulement le corps (``[1:22]`` -> body_pose, mains zéro) ; OMOMO tranche aussi les chaînes de main
    SMPL-H en left/right_hand_pose. Retourne ``(local (T, J, 3), transl (T, 3))`` (axis-angle, float32).
    """
    rg = _quat_to_R(quats_zup)                                         # (T, J, 3, 3) Z-up global
    n = rg.shape[1]
    local = np.empty((rg.shape[0], n, 3), np.float64)
    local[:, 0] = R.from_matrix(                                       # Q^-1 @ R_root (rebaser racine)
        np.einsum("ij,tjk->tik", YUP_TO_ZUP.T, rg[:, 0])).as_rotvec()
    for j in range(1, n):
        rl = np.einsum("tij,tjk->tik", rg[:, parents[j]].transpose(0, 2, 1), rg[:, j])
        local[:, j] = R.from_matrix(rl).as_rotvec()                    # parent-relative (invariant)
    transl = np.asarray(root_pos_zup, np.float64) @ YUP_TO_ZUP - j_rest0
    return local.astype(np.float32), transl.astype(np.float32)


def _axis_angle_55(p: SmplParams, t: int) -> np.ndarray:
    """(55, 3) axis-angle local pour le frame ``t`` (face/oeil par défaut zéro si absent)."""
    out = []
    for key in _SMPLX_AA:
        n = _SMPLX_AA_N[key]
        v = getattr(p, key)
        if v is None:
            out.append(np.zeros((n, 3)))
        else:
            out.append(np.asarray(v[t], np.float64).reshape(n, 3))
    return np.concatenate(out, axis=0)


def _axis_angle_seq(p: SmplParams) -> np.ndarray:
    """(T, 55, 3) axis-angle local pour TOUS les frames (face/oeil par défaut zéro si absent)."""
    T = p.n_frames
    out = []
    for key in _SMPLX_AA:
        n = _SMPLX_AA_N[key]
        v = getattr(p, key)
        out.append(np.zeros((T, n, 3)) if v is None else np.asarray(v, np.float64).reshape(T, n, 3))
    return np.concatenate(out, axis=1)


def _global_rotations(aa: np.ndarray, parents: np.ndarray) -> np.ndarray:
    """(J, 3, 3) rotation du monde par os = FK des axis-angles locaux en bas de l'arbre."""
    local = R.from_rotvec(aa).as_matrix()
    g = np.empty_like(local)
    for j in range(len(parents)):
        par = int(parents[j])
        g[j] = local[j] if par < 0 else g[par] @ local[j]
    return g


def _posed_joints(g: np.ndarray, j_rest: np.ndarray, parents: np.ndarray, transl: np.ndarray) -> np.ndarray:
    """(J, 3) positions de joints posés = FK des joints rest à travers ``g``, plus ``transl``."""
    jp = np.empty_like(j_rest)
    for j in range(len(parents)):
        par = int(parents[j])
        jp[j] = j_rest[j] if par < 0 else jp[par] + g[par] @ (j_rest[j] - j_rest[par])
    return jp + transl


class SmplBody:
    """``BodyModel`` pour un seul sujet (betas/gender fixés). Construire via ``build_body_model``."""

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

        # Pose de repos (betas, zero pose/transl) : joints natifs (pour FK) + verts natifs (pour échantillonnage).
        with torch.no_grad():
            rest = self._model(betas=self._betas_t)
        self._j_rest: np.ndarray = rest.joints[0].detach().cpu().numpy()[:_N_BONES].astype(np.float64)
        self._rest_verts: np.ndarray = rest.vertices[0].detach().cpu().numpy().astype(np.float32)
        self._lbs_weights: np.ndarray = self._model.lbs_weights.detach().cpu().numpy().astype(np.float32)

        # Stature de repos du sujet (m) = étendue verticale du mesh rest dans le frame Y-up NATIF du modèle.
        # Une propriété pure de mesh rest (pas de mouvement) -> ``BodyModel.stature`` ; alimente l'échelle humain->robot.
        self.stature: float = float(self._rest_verts[:, 1].max() - self._rest_verts[:, 1].min())

    @property
    def n_bones(self) -> int:
        return _N_BONES

    @property
    def rest_joints(self) -> np.ndarray:
        """(J_bones, 3) positions de joints rest dans le frame NATIF du modèle (pour reconstruction)."""
        return self._j_rest

    @property
    def lbs_weights(self) -> np.ndarray:
        """(V, J_bones) poids de skinning LBS. Spécifique à SMPL (délibérément PAS sur le protocol
        ``BodyModel``) : seul l'échantillonneur de nuage humain dans ``prepare/`` en a besoin, pour cuire
        le skinning clairsemé par point. Les décalages rest du nuage vivent dans le frame NATIF, correspondant
        à ces poids."""
        return self._lbs_weights

    def rest_vertices(self, params: SmplParams) -> np.ndarray:
        """(V, 3) sommets pose-rest dans le frame NATIF du modèle (pour échantillonnage de nuage).
        Fixe au sujet (les betas sont définis à la construction), donc ``params`` est ignoré —
        gardé pour conformité ``BodyModel`` (les appelants peuvent passer ``None``)."""
        return self._rest_verts

    def bone_transforms(self, params: SmplParams, t: int) -> tuple[np.ndarray, np.ndarray]:
        """(J,3,3) rotations du monde et (J,3) origines du monde au frame ``t`` (Z-up), via FK pur."""
        aa = _axis_angle_55(params, t)
        g_native = _global_rotations(aa, self.parents)
        transl = np.asarray(params.transl[t], np.float64)
        j_posed = _posed_joints(g_native, self._j_rest, self.parents, transl)
        rot_world = YUP_TO_ZUP @ g_native                       # Q R par os
        pos_world = j_posed @ YUP_TO_ZUP.T                      # Y-up -> Z-up
        return rot_world, pos_world

    def bone_positions(self, params: SmplParams) -> np.ndarray:
        """(T, J, 3) positions des os du monde (Z-up) pour TOUS les frames à la fois — FK pur en batch.

        Même propagation que ``bone_transforms`` mais vectorisée dans le temps (la boucle 55-joint s'exécute
        une fois, chaque étape diffuse sur T) : pour les longues séquences (HOI-M3 ~19k frames) cela évite
        un appel Python par frame. Les chargeurs prennent ``[:, :n]`` pour les joints de démo."""
        aa = _axis_angle_seq(params)                                       # (T, 55, 3)
        T = aa.shape[0]
        local = R.from_rotvec(aa.reshape(-1, 3)).as_matrix().reshape(T, _N_BONES, 3, 3)
        transl = np.asarray(params.transl, np.float64)                     # (T, 3)
        g = np.empty((T, _N_BONES, 3, 3))
        jp = np.empty((T, _N_BONES, 3))
        for j in range(_N_BONES):
            par = int(self.parents[j])
            if par < 0:
                g[:, j] = local[:, j]
                jp[:, j] = self._j_rest[j] + transl
            else:
                g[:, j] = g[:, par] @ local[:, j]
                jp[:, j] = jp[:, par] + np.einsum("tij,j->ti", g[:, par], self._j_rest[j] - self._j_rest[par])
        return jp @ YUP_TO_ZUP.T                                          # (T, 55, 3) Z-up

    def posed_vertices(self, params: SmplParams, t: int) -> np.ndarray:
        """(V, 3) sommets du mesh du monde au frame ``t`` (Z-up), via un vrai forward SMPL-X."""
        torch = self._torch
        kw = {}
        for key in _SMPLX_AA + ("transl", "expression"):
            v = getattr(params, key)
            if v is not None:
                kw[key] = torch.from_numpy(np.asarray(v[t: t + 1], np.float32))
        with torch.no_grad():
            out = self._model(betas=self._betas_t, **kw)
        verts = out.vertices[0].detach().cpu().numpy()
        return (verts @ YUP_TO_ZUP.T).astype(np.float32)


def build_body_model(params: SmplParams, model_dir: Path) -> SmplBody:
    """Construire le ``BodyModel`` pour ``params`` (un sujet) en utilisant le modèle SMPL à ``model_dir``."""
    return SmplBody(params, Path(model_dir))


def rest_body_model(betas: np.ndarray, gender: str, model_dir: Path) -> SmplBody:
    """Un ``BodyModel`` pour ``(betas, gender)`` à pose zéro — pour ses joints rest + arbre parent
    (partagé par les chargeurs d'orientation globale, qui ont besoin de ``rest_joints[0]`` pour ``transl``)."""
    z1 = np.zeros((1, 3), np.float32)
    dummy = SmplParams(betas=np.asarray(betas, np.float32).reshape(-1), global_orient=z1,
                       body_pose=np.zeros((1, 63), np.float32),
                       left_hand_pose=np.zeros((1, 45), np.float32),
                       right_hand_pose=np.zeros((1, 45), np.float32), transl=z1,
                       gender=gender, model_type="smplx")
    return build_body_model(dummy, Path(model_dir))
