"""C-D / C-X — the robot CONTACT residual builder. Couples ``δv`` (robot) and ``δξ`` (object channel):
the robot's M correspondence points are driven onto the demonstrated contact geometry. C-D linearises
the signed-distance error (current vs the transported human reference); C-X linearises the geodesic
WITNESS residual (drive the current witness toward the demonstrated contact location, target 0).

Frame convention (see plan Assumptions 1 & 3): OBJECT-channel ``direction``/witness/geodesic gradient
are object-LOCAL; map to world with ``world_normal(geo.rot[c], …)`` for the world ``point_jac`` (δv)
contraction, and contract the RAW local vector with ``probe_jac_obj[c]`` for the object tangent (δξ).
Ground channel: ``geo.rot[c] = I`` (local == world), no object coupling. Only pairs ``active`` in the
REFERENCE field become rows. Weights ``cfg.w_cd``/``cfg.w_cx`` × the per-row activation ``alpha(d_ref)``
are folded into ``A``/``c``."""
from __future__ import annotations

import numpy as np

from ..contracts import ResidualBlock
from ..config import SolveConfig
from ._ops import GeoField, dist_jac, geo_chain, scatter_obj, world_normal
from ...targets.contracts import ContactEval, RobotInteractionTargets
from ...targets.interaction.geodesic import geo_value_grad, nearest_index


def _alpha(d_ref: np.ndarray, cfg: SolveConfig) -> np.ndarray:
    """Per-row contact activation weight from the demonstrated distance: soft falloff
    ``exp(−(max(d_ref,0)/scale)²)`` (closer demonstrated contact -> ~1, far -> ~0). ``scale <= 0``
    disables the falloff (every active row weight 1). Gating to active rows is done by the caller."""
    if cfg.contact_d_ref_scale <= 0.0:
        return np.ones_like(d_ref)
    return np.exp(-(np.clip(d_ref, 0.0, None) / cfg.contact_d_ref_scale) ** 2)


def build_contact(contact_eval: ContactEval, robot_field_ref: RobotInteractionTargets,
                  geo: GeoField, cfg: SolveConfig) -> list[ResidualBlock]:
    """``[C-D]`` (+ ``[C-X]`` if any active pair has a geodesic table). Stacks all active (channel,
    point) pairs into rows; folds ``w · alpha(d_ref)`` into ``A``/``c``. ``n_obj`` from the field's
    object channels. Returns ``[]`` if no active pair."""
    field_cur, field_ref = contact_eval.field, robot_field_ref.field
    C, M = field_cur.n_channels, field_cur.n_points
    nv = contact_eval.point_jac.shape[2]
    n_obj = sum(1 for j in geo.object_idx if j >= 0)
    active = field_ref.active                                       # demonstrated contacts (Assumption 5)

    cd_A, cd_c, cd_Aobj = [], [], []
    cx_A, cx_c, cx_Aobj = [], [], []
    for cidx in range(C):
        rows = np.nonzero(active[cidx])[0]
        if rows.size == 0:
            continue
        R_i = geo.rot[cidx]
        obj = geo.object_idx[cidx]
        dir_local = field_cur.direction[cidx, rows]                 # (k,3) object-local (world if ground)
        n_world = world_normal(R_i, dir_local)                      # (k,3) -> world for point_jac
        w = (cfg.w_cd * _alpha(field_ref.distance[cidx, rows], cfg))[:, None]   # (k,1)

        # --- C-D : signed-distance error ---
        cd_A.append(w * dist_jac(n_world, contact_eval.point_jac[rows]))        # (k,nv)
        cd_c.append((w[:, 0]) * (field_cur.distance[cidx, rows] - field_ref.distance[cidx, rows]))
        if obj >= 0:
            blk = w * dist_jac(dir_local, contact_eval.probe_jac_obj[cidx, rows])  # (k,6) object-local
            cd_Aobj.append(scatter_obj(blk, obj, n_obj))
        else:
            cd_Aobj.append(np.zeros((rows.size, n_obj * 6)) if n_obj else None)

        # --- C-X : geodesic witness residual (only channels with a table) ---
        table = geo.tables[cidx]
        if table is not None:
            wx = (cfg.w_cx * _alpha(field_ref.distance[cidx, rows], cfg))[:, None]
            src = nearest_index(table.points, field_ref.witness[cidx, rows])    # source = ref witness
            val, grad_local = geo_value_grad(table, src, field_cur.witness[cidx, rows])  # query = cur
            grad_world = world_normal(R_i, grad_local)              # object-local grad -> world
            cx_A.append(wx * geo_chain(grad_world, contact_eval.point_jac[rows]))
            cx_c.append((wx[:, 0]) * val)                           # target 0 (no ref to subtract)
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
