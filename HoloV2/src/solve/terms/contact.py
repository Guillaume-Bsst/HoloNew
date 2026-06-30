"""C-D / C-X — le builder résiduel CONTACT du robot. Couple ``δv`` (robot) et ``δξ`` (canal d'objet) :
les M points de correspondance du robot sont entraînés sur la géométrie de contact démontrée. C-D linéarise
l'erreur de distance signée (courant vs référence humaine transportée) ; C-X linéarise le résiduel
WITNESS géodésique (pousse le witness courant vers l'emplacement de contact démontré, cible 0).

Convention de frame (voir Assumptions 1 & 3 du plan) : le ``direction``/witness/gradient géodésique du canal
OBJECT sont LOCAL-d'objet ; mappés au monde avec ``world_normal(geo.rot[c], …)`` pour la contraction du
``point_jac`` (δv) du monde, et contracter le vecteur LOCAL BRUT avec ``probe_jac_obj[c]`` pour la tangente
d'objet (δξ). Canal sol : ``geo.rot[c] = I`` (local == monde), pas de couplage d'objet. Seulement les paires
``active`` dans le champ RÉFÉRENCE deviennent lignes. Les poids ``cfg.w_cd``/``cfg.w_cx`` × l'activation
par-ligne ``alpha(d_ref)`` sont repliés dans ``A``/``c``."""
from __future__ import annotations

import numpy as np

from ..contracts import ResidualBlock
from ..config import SolveConfig
from ._ops import GeoField, dist_jac, geo_chain, scatter_obj, world_normal
from ...targets.contracts import ContactEval, RobotInteractionTargets
from ...targets import geo_value_grad, nearest_index   # surface publique (re-export racine), pas l'interne


def _alpha(d_ref: np.ndarray, cfg: SolveConfig) -> np.ndarray:
    """Poids d'activation de contact par-ligne à partir de la distance démontrée : affaiblissement doux
    ``exp(−(max(d_ref,0)/scale)²)`` (contact démontré plus proche -> ~1, éloigné -> ~0). ``scale <= 0``
    désactive l'affaiblissement (poids 1 pour chaque ligne active). Le gating vers les lignes actives
    est fait par l'appelant."""
    if cfg.contact_d_ref_scale <= 0.0:
        return np.ones_like(d_ref)
    return np.exp(-(np.clip(d_ref, 0.0, None) / cfg.contact_d_ref_scale) ** 2)


def build_contact(contact_eval: ContactEval, robot_field_ref: RobotInteractionTargets,
                  geo: GeoField, cfg: SolveConfig) -> list[ResidualBlock]:
    """``[C-D]`` (+ ``[C-X]`` si une paire active a une table géodésique). Empile tous les paires (canal,
    point) actives en lignes ; replie ``w · alpha(d_ref)`` dans ``A``/``c``. ``n_obj`` des canaux d'objet
    du champ. Retourne ``[]`` si pas de paire active."""
    field_cur, field_ref = contact_eval.field, robot_field_ref.field
    C, M = field_cur.n_channels, field_cur.n_points
    nv = contact_eval.point_jac.shape[2]
    n_obj = sum(1 for j in geo.object_idx if j >= 0)
    active = field_ref.active                                       # contacts démontrés (Assumption 5)

    cd_A, cd_c, cd_Aobj = [], [], []
    cx_A, cx_c, cx_Aobj = [], [], []
    for cidx in range(C):
        rows = np.nonzero(active[cidx])[0]
        if rows.size == 0:
            continue
        R_i = geo.rot[cidx]
        obj = geo.object_idx[cidx]
        dir_local = field_cur.direction[cidx, rows]                 # (k,3) local-d'objet (monde si sol)
        n_world = world_normal(R_i, dir_local)                      # (k,3) -> monde pour point_jac
        w = (cfg.w_cd * _alpha(field_ref.distance[cidx, rows], cfg))[:, None]   # (k,1)

        # --- C-D : erreur de distance signée ---
        cd_A.append(w * dist_jac(n_world, contact_eval.point_jac[rows]))        # (k,nv)
        cd_c.append((w[:, 0]) * (field_cur.distance[cidx, rows] - field_ref.distance[cidx, rows]))
        if obj >= 0:
            blk = w * dist_jac(dir_local, contact_eval.probe_jac_obj[cidx, rows])  # (k,6) local-d'objet
            cd_Aobj.append(scatter_obj(blk, obj, n_obj))
        else:
            cd_Aobj.append(np.zeros((rows.size, n_obj * 6)) if n_obj else None)

        # --- C-X : résiduel witness géodésique (seulement canaux avec une table) ---
        table = geo.tables[cidx]
        if table is not None:
            wx = (cfg.w_cx * _alpha(field_ref.distance[cidx, rows], cfg))[:, None]
            src = nearest_index(table.points, field_ref.witness[cidx, rows])    # source = witness ref
            val, grad_local = geo_value_grad(table, src, field_cur.witness[cidx, rows])  # query = courant
            grad_world = world_normal(R_i, grad_local)              # grad local-d'objet -> monde
            cx_A.append(wx * geo_chain(grad_world, contact_eval.point_jac[rows]))
            cx_c.append((wx[:, 0]) * val)                           # cible 0 (pas de ref à soustraire)
            if obj >= 0:
                gblk = wx * dist_jac(grad_local, contact_eval.probe_jac_obj[cidx, rows])
                cx_Aobj.append(scatter_obj(gblk, obj, n_obj))
            elif n_obj:
                cx_Aobj.append(np.zeros((rows.size, n_obj * 6)))

    blocks: list[ResidualBlock] = []
    if cd_c:
        A_obj = np.vstack(cd_Aobj) if (n_obj and all(b is not None for b in cd_Aobj)) else None
        blocks.append(ResidualBlock(A=np.vstack(cd_A), c=np.concatenate(cd_c),
                                    A_obj=A_obj, name="C-D"))
    if cx_c:
        A_obj = np.vstack(cx_Aobj) if (n_obj and len(cx_Aobj) == len(cx_c)) else None
        blocks.append(ResidualBlock(A=np.vstack(cx_A), c=np.concatenate(cx_c),
                                    A_obj=A_obj, name="C-X"))
    return blocks
