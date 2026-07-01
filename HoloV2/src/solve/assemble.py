"""assemble — (evals + refs + geo + robot + cfg) -> ``Problem``. Le pont entre les ÉVALUATIONS courantes
(``FrameEval`` : FK de style + champ de contact) et le sous-problème QP linéarisé : appelle les builders
``terms/`` (poids repliés) et ``terms/constraints`` (limites articulaires + boîte région-confiance),
concatène en UN ``Problem``. PUR — pas de cinématique ici (déléguée à Eval), pas cvxpy.

``geo`` = contexte géodésique par canal que ``build_contact`` lit pour le résiduel witness C-X (sourcé
du côté runner depuis ``ctx.channels`` — chaque ``Channel`` porte sa ``geodesic`` + ``sdf``)."""
from __future__ import annotations

import numpy as np

from .contracts import Problem, TrustRegion
from .config import SolveConfig
from .retract import quat_wxyz_to_mat
from .terms._ops import GeoField
from .terms.style import build_style
from .terms.contact import build_contact
from .terms.object import build_object
from .terms.reg import build_reg
from .terms.constraints import build_constraints


def _geo_field(channels, object_rot, object_pos) -> GeoField:
    """Paquet par-trame dont build_contact a besoin : frames local-d'objet + tables géodésiques par canal.
    Channel.object_idx est None pour le sol -> -1 / frame identité (convention GeoField)."""
    tables     = tuple(ch.geodesic for ch in channels)
    rot        = np.stack([np.eye(3) if ch.object_idx is None
                           else object_rot[ch.object_idx] for ch in channels])   # (C,3,3)
    pos        = np.stack([np.zeros(3) if ch.object_idx is None
                           else object_pos[ch.object_idx] for ch in channels])   # (C,3)
    object_idx = tuple(-1 if ch.object_idx is None else ch.object_idx for ch in channels)
    return GeoField(tables=tables, rot=rot, pos=pos, object_idx=object_idx)


def assemble(evals, frame_targets, geo, robot, cfg: SolveConfig,
             object_poses_cur=None) -> Problem:
    """Construit le ``Problem`` pour UNE itération SQP. ``n_obj`` est dérivé des poses d'objet de la trame ;
    si ``n_obj = 0`` les builders d'objet produisent simplement pas de blocs ``A_obj``.

    ``object_poses_cur`` = itéré COURANT des objets ``(N, 7)`` (pos + quat wxyz) au point de linéarisation,
    transmis par ``loop`` : c'est la CIBLE de linéarisation du terme O (ancre l'objet à sa pose OBSERVÉE
    ``frame_targets.object_pos``). ``None`` -> retombe sur l'observée (ancrage O nul, comportement v1)."""
    se, ce = evals.style, evals.contact
    blocks = []
    blocks += list(build_style(se, frame_targets.style, cfg))
    geo_field = _geo_field(geo, frame_targets.object_rot, frame_targets.object_pos)
    blocks += list(build_contact(ce, frame_targets.robot_interaction, geo_field, cfg))
    # Itéré courant des objets -> (rot, pos) pour l'ancre O (et le frame CO-D) ; None -> observée (v1).
    cur_rot = cur_pos = None
    if object_poses_cur is not None and len(object_poses_cur) > 0:
        cp = np.asarray(object_poses_cur, np.float64)                     # (N, 7)
        cur_pos = cp[:, :3]                                               # (N, 3)
        cur_rot = np.stack([quat_wxyz_to_mat(cp[i, 3:7]) for i in range(cp.shape[0])])  # (N, 3, 3)
    blocks += list(build_object(ce, frame_targets.env_interaction,
                                frame_targets.object_rot, frame_targets.object_pos, cfg,
                                object_rot_cur=cur_rot, object_pos_cur=cur_pos))
    blocks += list(build_reg(robot.nv, cfg))
    constraints, trust_regions = build_constraints(robot, cfg)
    n_obj = int(frame_targets.object_rot.shape[0])
    trust_regions = list(trust_regions)
    if n_obj > 0:
        obj_r = np.tile(np.concatenate([np.full(3, cfg.tr_object_pos),
                                        np.full(3, cfg.tr_object_rot)]), n_obj)   # (n_obj*6,)
        trust_regions.append(TrustRegion(var="dxi", radius=obj_r, norm=-1))
    return Problem(nv=robot.nv, n_obj=n_obj,
                   residuals=tuple(blocks),
                   constraints=tuple(constraints),
                   trust_regions=tuple(trust_regions))
