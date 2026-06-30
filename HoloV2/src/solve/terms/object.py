"""CO-D / O — le builder résiduel OBJET-comme-variable (objet seulement, ``δξ`` ; ``A = 0``). CO-D pousse
les PROPRES contacts de l'objet à rester cohérents (objet vs sol / vs autres objets) : les points du nuage
de l'objet ``i`` reposent sur les canaux de la scène ; leur mouvement couple par ``cloud_jac_self`` (objet ``i``)
et, pour un canal objet↔objet, ``probe_jac_obj`` (l'autre objet). O ancre chaque objet à sa pose
OBSERVÉE (``se3_log_world``). L'objet est tiré par C (Task 4), retenu par CO-D + O.

Convention de frame (Assumption 4 du plan) : canal 0 = sol (frame monde I) ; canal ``c>=1`` = objet
``c−1`` (sa pose mondiale ``object_rot[c−1]``). Le terme propre utilise la normale MAPPÉE-AU-MONDE (``cloud_jac_self``
est monde) ; le terme croisé objet↔objet utilise la normale LOCAL BRUT contre ``probe_jac_obj``. La
diagonale propre (objet ``i`` vs son propre canal) est déjà ``active=False`` en amont, donc elle n'émet pas de ligne.

NOTE — CO-X (géodésique) est DÉFÉRÉ (Assumption 6 du plan) : il a besoin des tables géodésiques par-canal, que la
signature canonique 5-arg ``build_object`` ne porte pas (contrairement au ``geo`` de ``build_contact``). C'est
identique à C-X une fois qu'un paquet ``geo`` est enfilé ; ``cfg.w_cox`` est réservé pour lui. Le ``c`` de O est
zéro au point de linéarisation en v1 (pose courant == observée au départ de la trame) ; la formule générale
``se3_log_world(ref, cur)`` est implémentée pour qu'une pose courant en direct du Plan-C donne un ancrage non-zéro."""
from __future__ import annotations

import numpy as np

from ..contracts import ResidualBlock
from ..config import SolveConfig
from ._ops import dist_jac, scatter_obj, se3_log_world, world_normal
from ...targets.contracts import ContactEval, EnvironmentInteractionTargets


def build_object(contact_eval: ContactEval, env_refs: EnvironmentInteractionTargets,
                 object_rot: np.ndarray, object_pos: np.ndarray,
                 cfg: SolveConfig) -> list[ResidualBlock]:
    """``[CO-D]`` (si quelconque auto-contact d'objet actif) + ``[O]`` (toujours, un bloc 6-ligne par objet).
    ``A = 0`` pour les deux (les termes d'objet ne touchent que ``δξ``). ``nv`` depuis ``point_jac`` ; ``N`` objets."""
    N = object_rot.shape[0]
    nv = contact_eval.point_jac.shape[2]
    blocks: list[ResidualBlock] = []

    # --- CO-D : auto-cohérence d'objet, per object cloud i, empilé sur (canal, point) actifs ---
    cod_A, cod_c, cod_Aobj = [], [], []
    for i, env in enumerate(contact_eval.env):
        field_cur = env.field
        field_ref = env_refs.per_object[i]                    # MultiChannelField directly
        C = field_cur.n_channels
        for cidx in range(C):
            rows = np.nonzero(field_ref.active[cidx])[0]
            if rows.size == 0:
                continue
            jc = cidx - 1                                     # canal -> index d'objet (-1 = sol)
            R_c = np.eye(3) if jc < 0 else object_rot[jc]    # frame monde du canal
            dir_local = field_cur.direction[cidx, rows]       # (k,3) local-canal (monde si sol)
            n_world = world_normal(R_c, dir_local)            # pour cloud_jac_self (monde)
            w = cfg.w_cod

            # terme propre : l'objet i déplace son propre point de nuage -> slot d'objet i
            self_blk = w * dist_jac(n_world, env.cloud_jac_self[rows])     # (k,6)
            A_obj = scatter_obj(self_blk, i, N)
            # terme croisé objet↔objet : le canal jc != i déplace la sonde -> slot objet jc (normale locale brute)
            if jc >= 0 and jc != i:
                cross_blk = w * dist_jac(dir_local, env.probe_jac_obj[cidx, rows])
                A_obj = A_obj + scatter_obj(cross_blk, jc, N)

            cod_A.append(np.zeros((rows.size, nv)))
            cod_c.append(w * (field_cur.distance[cidx, rows] - field_ref.distance[cidx, rows]))
            cod_Aobj.append(A_obj)

    if cod_c:
        blocks.append(ResidualBlock(A=np.vstack(cod_A), c=np.concatenate(cod_c),
                                    A_obj=np.vstack(cod_Aobj), name="CO-D"))

    # --- O : ancre chaque objet à sa pose observée (courant == observé au pt de linéarisation) ---
    e = se3_log_world(object_rot, object_pos, object_rot, object_pos)      # (N,6) -> 0 en v1 (Assumption 6)
    blocks.append(ResidualBlock(A=np.zeros((N * 6, nv)), c=cfg.w_obj * e.reshape(N * 6),
                                A_obj=cfg.w_obj * np.eye(N * 6), name="O"))
    return blocks
