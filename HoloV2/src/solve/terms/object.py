"""CO-D / O — the OBJECT-as-variable residual builder (object only, ``δξ``; ``A = 0``). CO-D drives the
object's OWN contacts to stay consistent (object vs ground / vs other objects): object ``i``'s cloud
points sit on the scene channels; their motion couples through ``cloud_jac_self`` (object ``i``) and,
for an object↔object channel, ``probe_jac_obj`` (the other object). O anchors each object to its
OBSERVED pose (``se3_log_world``). The object is pulled by C (Task 4), retained by CO-D + O.

Frame convention (plan Assumption 4): channel 0 = ground (world frame I); channel ``c>=1`` = object
``c−1`` (its world pose ``object_rot[c−1]``). Self term uses the WORLD-mapped normal (``cloud_jac_self``
is world); the object↔object cross term uses the RAW LOCAL normal against ``probe_jac_obj``. The self
diagonal (object ``i`` vs its own channel) is already ``active=False`` upstream, so it emits no row.

NOTE — CO-X (geodesic) is DEFERRED (plan Assumption 6): it needs the per-channel geodesic tables, which
the canonical 5-arg ``build_object`` signature does not carry (unlike ``build_contact``'s ``geo``). It
is identical to C-X once a ``geo`` bundle is threaded; ``cfg.w_cox`` is reserved for it. O's ``c`` is
zero at the linearisation point in v1 (current pose == observed at frame start); the general
``se3_log_world(ref, cur)`` formula is implemented so a Plan-C live current pose yields a non-zero
anchor."""
from __future__ import annotations

import numpy as np

from ..contracts import ResidualBlock
from ..config import SolveConfig
from ._ops import dist_jac, scatter_obj, se3_log_world, world_normal
from ...targets.contracts import ContactEval, EnvironmentInteractionTargets


def build_object(contact_eval: ContactEval, env_refs: EnvironmentInteractionTargets,
                 object_rot: np.ndarray, object_pos: np.ndarray,
                 cfg: SolveConfig) -> list[ResidualBlock]:
    """``[CO-D]`` (if any active object self-contact) + ``[O]`` (always, one 6-row block per object).
    ``A = 0`` for both (object terms touch only ``δξ``). ``nv`` from ``point_jac``; ``N`` objects."""
    N = object_rot.shape[0]
    nv = contact_eval.point_jac.shape[2]
    blocks: list[ResidualBlock] = []

    # --- CO-D : object self-consistency, per object cloud i, stacked over active (channel, point) ---
    cod_A, cod_c, cod_Aobj = [], [], []
    for i, env in enumerate(contact_eval.env):
        field_cur = env.field
        field_ref = env_refs.per_object[i]                    # MultiChannelField directly
        C = field_cur.n_channels
        for cidx in range(C):
            rows = np.nonzero(field_ref.active[cidx])[0]
            if rows.size == 0:
                continue
            jc = cidx - 1                                     # channel -> object index (-1 = ground)
            R_c = np.eye(3) if jc < 0 else object_rot[jc]    # channel world frame
            dir_local = field_cur.direction[cidx, rows]       # (k,3) channel-local (world if ground)
            n_world = world_normal(R_c, dir_local)            # for cloud_jac_self (world)
            w = cfg.w_cod

            # self term: object i moves its own cloud point -> object i slot
            self_blk = w * dist_jac(n_world, env.cloud_jac_self[rows])     # (k,6)
            A_obj = scatter_obj(self_blk, i, N)
            # object<->object term: channel jc != i moves the probe -> object jc slot (raw local normal)
            if jc >= 0 and jc != i:
                cross_blk = w * dist_jac(dir_local, env.probe_jac_obj[cidx, rows])
                A_obj = A_obj + scatter_obj(cross_blk, jc, N)

            cod_A.append(np.zeros((rows.size, nv)))
            cod_c.append(w * (field_cur.distance[cidx, rows] - field_ref.distance[cidx, rows]))
            cod_Aobj.append(A_obj)

    if cod_c:
        blocks.append(ResidualBlock(A=np.vstack(cod_A), c=np.concatenate(cod_c),
                                    A_obj=np.vstack(cod_Aobj), name="CO-D"))

    # --- O : anchor each object to its observed pose (current == observed at the linearisation pt) ---
    e = se3_log_world(object_rot, object_pos, object_rot, object_pos)      # (N,6) -> 0 in v1 (Assumption 6)
    blocks.append(ResidualBlock(A=np.zeros((N * 6, nv)), c=cfg.w_obj * e.reshape(N * 6),
                                A_obj=cfg.w_obj * np.eye(N * 6), name="O"))
    return blocks
