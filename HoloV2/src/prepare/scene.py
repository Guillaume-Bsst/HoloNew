"""Assemble une GroundedScene en appliquant la Calibration au mouvement chargé et poses objets.

L'ancrage est PER ENTITY (mono-humain / multi-objet) : l'humain (joints démo + params SMPL) chute par
``Calibration.human_offset``, et TOUS les objets chutent ensemble par le seul ``Calibration.object_offset``
— l'humain et les objets peuvent s'asseoir à des hauteurs différentes dans la capture brute (p. ex.
l'humain flotte tandis que les objets reposent déjà au sol), donc un décalage scène partagé pousserait
les objets à travers le sol ; les objets partagent un décalage pour que leur géométrie inter-objets soit
gardée. La scène reste à l'échelle HUMAINE : l'échelle human→robot n'est PAS appliquée ici — l'échelle
de scène (placement) est appliquée en aval sur les RÉFÉRENCES de ``targets``
(``targets.config.SceneScaleConfig``), après l'éval sur la scène réelle.

Le ``body`` du sujet (construit une fois en amont par le runner) est porté À TRAVERS vers la
``GroundedScene`` pour que le traitement per-frame pose le nuage humain (``body.bone_transforms``) ;
il est ``None`` pour une source positions-only. ``assemble`` ne stocke que la référence — reste pur
(numpy-only), le body backed-torch est construit dans ``load/smpl.py``.

``root_frame`` est identité pour l'instant (provisoire), donc seules les translations z sont appliquées.
Quand un framing non-trivial est introduit, l'appliquer aux tableaux monde ici (et rebaser les params
natives en conséquence).
"""
from __future__ import annotations

from dataclasses import replace

import numpy as np

from .contracts import BodyModel, Calibration, GroundedScene, RawMotion, SmplParams

# Les params SMPL sont dans le repère NATIVE Y-up du modèle ; le body model mappe native Y → world Z
# (la rotation Q dans load/smpl.py). Donc une chute z-monde de ``human_offset`` est une chute y-native
# de la translation racine du même montant — poser les params ancrés produit alors le monde ancré.
_NATIVE_UP_AXIS = 1   # colonne transl portant la hauteur monde


def _drop_object_z(pose: np.ndarray, dz: float) -> np.ndarray:
    """Baisse la pose per-frame d'un objet (T,7) [x,y,z,qw,qx,qy,qz] de ``dz`` en z monde."""
    out = np.asarray(pose, np.float32).copy()
    out[:, 2] -= dz
    return out


def _ground_params(params: SmplParams, dz: float) -> SmplParams:
    """Baisse l'humain de ``dz`` en z monde via la translation racine native (voir note module)."""
    transl = np.asarray(params.transl).copy()
    transl[:, _NATIVE_UP_AXIS] -= dz
    return replace(params, transl=transl)


def assemble(raw: RawMotion, calib: Calibration, body: BodyModel | None = None) -> GroundedScene:
    """Applique ``calib`` à un ``RawMotion`` chargé → ``GroundedScene`` (ancrée, échelle humaine).

    L'humain chute de ``human_offset`` ; tous les objets chutent ensemble du seul ``object_offset``.
    ``body`` (moteur de pose du sujet, construit une fois en amont) est porté inchangé ; passe
    ``None`` pour une source positions-only."""
    dz = float(calib.human_offset)
    joints = np.asarray(raw.joint_pos, np.float32).copy()
    joints[:, :, 2] -= dz
    objects = tuple(_drop_object_z(p, calib.object_offset) for p in raw.object_poses_raw)
    params = _ground_params(raw.smpl_params, dz) if raw.is_parametric else None
    return GroundedScene(
        joint_pos=joints, joint_names=raw.joint_names, object_poses=objects,
        object_mesh_paths=raw.object_mesh_paths, calibration=calib, fps=raw.fps,
        smpl_params=params, body=body,
    )
