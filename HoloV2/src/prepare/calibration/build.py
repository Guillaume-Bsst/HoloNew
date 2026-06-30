"""prepare/calibration — ancre la scène entière (humain + objets) au sol.

Produit la ``Calibration`` per-(sujet, prise) (``contracts.Calibration``) : un décalage d'ancrage
humain, un décalage d'ancrage objet partagé et une frame racine — calculés offline une fois et cachés.
ROBOT-FREE *ET* BODY-FREE par conception : les décalages viennent des joints démo mocap (sol humain)
et des meshes/poses objets (sol objet) seuls — pas de betas/body nécessaires — donc cache par prise.
La ``stature`` du sujet vit sur ``BodyModel`` (son propriétaire naturel rest-mesh, voir ``load/smpl.py``),
et l'échelle human→robot (une quantité (human, robot)) est composée en aval à la seam transport.

  - ``human_offset``  = décalage z monde ancrant l'humain. Le percentile ``CalibrationConfig.foot_percentile``
                        (défaut 50 = médiane) de la hauteur JOINT-PIED mocap inférieure sur le clip.
                        Le joint pied (pas la seule SMPL) est utilisé exprès : le pied SMPL pénètre
                        fréquemment EN DESSOUS du niveau repos lors de l'articulation des orteils, donc
                        chasser le point de surface le plus bas sur-lève l'humain ; le percentile du
                        joint cible le niveau REPOS/contact. Joints démo mocap ⇒ pas de forward SMPL
                        et un chemin pour sources paramétriques ET positions-only.
  - ``object_offset`` = UN SEUL décalage z monde partagé par TOUS les objets : ancre l'objet qui touche
                        le plus le sol (celui qui touche le sol) juste au-dessus z=0, via un percentile
                        bas des points objets posés. Partagé (pas par-objet) pour garder la géométrie
                        inter-objets. TODO : une calibration par-objet/inter-objets plus fine pourrait
                        optimiser conjointement les contacts objet↔objet & objet↔sol.
  - ``root_frame``    = une framing (4,4) de la racine. Identité pour maintenant (hook provisoire ; V1
                        gardait XY brut). Généraliser seulement quand un framing non-trivial est réel.

Décisions (confirmées) : l'échelle human→robot n'est PAS calculée ici (calibration reste robot-free,
cacheable par sujet) ; la scène reste à l'échelle HUMAINE, où le pipeline interaction vit (nuage
humain + objet sont size-consistent, et le transport SMPL→robot absorbe la différence de taille).
L'ancrage est TOUJOURS recalculé (uniforme) : un clip déjà ancré (SFU) produit simplement
``human_offset ~= 0``.

Agnostique dataset par construction : chaque différence entre datasets vit dans les DONNÉES que le loader
a déjà produites (``joint_names`` → indices joints-pieds pour le décalage humain ; meshes objets →
décalage objet partagé, liste vide → 0), jamais dans une branche ici.

Porté de HoloNew : ``holosoma/preprocess.ground_to_floor`` (ancrage joint-pied).
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation as R

from ..contracts import Calibration, RawMotion
from ..config import CalibrationConfig
from .cache import load_calibration, save_calibration


# =============================================================================
# Fonctions pures (pas I/O, pas mutation) — la math d'ancrage
# =============================================================================
def foot_floor_offset(joint_pos: np.ndarray, foot_idx: list[int], percentile: float) -> float:
    """Décalage sol humain = ``percentile`` de la hauteur JOINT-PIED mocap inférieure sur le clip.

    Par frame, le plus bas des joints pieds donne la hauteur pied ; le percentile sur frames cible le
    niveau REPOS/contact. Le joint pied mocap est utilisé (pas la seule SMPL) : le pied SMPL pénètre
    fréquemment en dessous du niveau repos lors de l'articulation des orteils, donc le point de surface
    le plus bas s'assoit en dessous du sol et le chasser (un min / percentile bas) sur-lève l'humain.
    ``percentile`` = 50 est la médiane ; plus haut pousse l'humain vers le bas (vers le niveau planté),
    plus bas le soulève. Validé visuellement sur HODome / OMOMO / SFU. Retourne la valeur à SOUSTRAIRE
    de z monde."""
    z = np.asarray(joint_pos)[:, foot_idx, 2].min(axis=1)     # (T,) pied inférieur par frame
    return float(np.percentile(z, percentile))


def object_floor_offset(object_verts: list[np.ndarray], object_poses: list[np.ndarray],
                        percentile: float, vert_cap: int = 4000, frame_cap: int = 150) -> float:
    """UN SEUL décalage sol partagé pour TOUS les objets = ``percentile`` du point objet posé le plus bas
    sur le clip, regroupés entre objets. Ancre l'objet qui touche le sol à ~z=0 (un percentile bas =
    sa « portée la plus basse », p. ex. quand on le pose) ; le MÊME décalage est alors appliqué à
    chaque objet pour que la géométrie inter-objets soit préservée. Retourne la valeur à SOUSTRAIRE de
    z monde, ou 0.0 s'il n'y a pas d'objets.

    TODO : une calibration par-objet/inter-objets plus fine pourrait ancrer chaque objet et optimiser
    conjointement les contacts objet↔objet et objet↔sol ; pour maintenant un décalage partagé
    (objet-qui-touche-le-sol juste au-dessus du sol) suffit. ``object_verts`` : liste de (V,3) local ;
    ``object_poses`` : liste de (T,7) pos-first wxyz. Verts/frames sont sous-échantillonnés (caps)
    pour borner le coût sur les scans denses."""
    if not object_verts:
        return 0.0
    rng = np.random.default_rng(0)
    lows = []
    for verts, poses in zip(object_verts, object_poses):
        v = np.asarray(verts, np.float64)
        if v.shape[0] > vert_cap:
            v = v[rng.choice(v.shape[0], vert_cap, replace=False)]
        poses = np.asarray(poses, np.float64)
        fidx = np.unique(np.linspace(0, poses.shape[0] - 1, min(poses.shape[0], frame_cap)).astype(int))
        rz = R.from_quat(poses[fidx][:, [4, 5, 6, 3]]).as_matrix()[:, 2, :]   # (F,3) ligne z-monde, wxyz->xyzw
        world_z = np.einsum("fj,vj->fv", rz, v) + poses[fidx, 2][:, None]     # (F, V) z monde de chaque vertex
        lows.append(world_z.min(axis=1))                                      # (F,) lowest object point/frame
    return float(np.percentile(np.concatenate(lows), percentile))


def _foot_indices(joint_names: tuple[str, ...]) -> list[int]:
    """Indices des joints pieds (case-insensitive 'foot', sinon 'ankle'). Lève si aucun."""
    feet = [i for i, n in enumerate(joint_names) if "foot" in n.lower()]
    if not feet:
        feet = [i for i, n in enumerate(joint_names) if "ankle" in n.lower()]
    if not feet:
        raise ValueError(f"no foot/ankle joint to ground on in {joint_names!r}")
    return feet


# =============================================================================
# CalibrationBuilder — l'AssetBuilder pour ce livrable (build / cache)
# =============================================================================
class CalibrationBuilder:
    """``AssetBuilder`` produisant la ``Calibration`` d'ancrage pour une (sujet, prise). Scopé à cette
    paire (PAS un cache géométrie) : les décalages sol dépendent du mouvement entier + objets.
    ROBOT-FREE *ET* BODY-FREE (pas d'input robot/body → cache par prise). Le runner enveloppe
    ``build``/``load`` dans un ``prof.span("calibration")``."""

    def cache_key(self, config: CalibrationConfig, raw: RawMotion) -> str:
        """Hash stable de tout ce dont l'ancrage dépend : les knobs (``foot_percentile``,
        ``object_floor_pct``) + les joints démo (décalage sol joint-pied) + les meshes objets + poses
        (le décalage sol objet partagé). Pas de terme body/betas (la stature vit sur BodyModel) et
        pas de terme robot — l'ancrage est body-free et robot-free."""
        h = hashlib.sha1()
        h.update(f"{config.foot_percentile}|{config.object_floor_pct}".encode())
        h.update(np.ascontiguousarray(raw.joint_pos, np.float32).tobytes())   # détermine le décalage pied
        for path, pose in zip(raw.object_mesh_paths, raw.object_poses_raw):   # détermine le décalage objet
            h.update(Path(path).read_bytes())   # BYTES mesh : un mesh qui change à un chemin constant
            h.update(np.ascontiguousarray(pose, np.float32).tobytes())        # doit changer la clé
        return h.hexdigest()

    def build(self, config: CalibrationConfig, raw: RawMotion) -> Calibration:
        """Calcule la ``Calibration`` d'ancrage à partir du ``RawMotion`` chargé. Body-free : le sol
        humain est le percentile joint-pied (joints démo mocap) et le sol objet un percentile du point
        objet posé le plus bas — aucun n'a besoin du body SMPL (la stature du sujet vit sur BodyModel).
        Uniforme pour sources paramétriques et positions-only."""
        # Décalage sol humain : percentile de la hauteur joint-pied inférieur (joints démo mocap).
        human = foot_floor_offset(raw.joint_pos, _foot_indices(raw.joint_names), config.foot_percentile)

        # Décalage sol objet partagé : ancre l'objet qui touche le plus juste au-dessus z=0 (le sol qu'il
        # touche), appliqué à TOUS les objets pour garder la géométrie inter-objets. Besoin des meshes.
        if raw.object_mesh_paths:
            from ..load.mesh import load_mesh
            obj_verts = [load_mesh(p)[0] for p in raw.object_mesh_paths]
            object_offset = object_floor_offset(obj_verts, list(raw.object_poses_raw), config.object_floor_pct)
        else:
            object_offset = 0.0

        return Calibration(human_offset=human, object_offset=object_offset, root_frame=np.eye(4))

    def save(self, calib: Calibration, path: Path) -> None:
        return save_calibration(calib, path)   # la persistence vit dans cache.py

    def load(self, path: Path) -> Calibration:
        return load_calibration(path)


def build_calibration(raw: RawMotion, config: CalibrationConfig) -> Calibration:
    """Wrapper commodité autour de ``CalibrationBuilder.build`` (le site d'appel courant)."""
    return CalibrationBuilder().build(config, raw)
