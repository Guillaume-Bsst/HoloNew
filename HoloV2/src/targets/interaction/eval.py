"""contact_eval — pendant ÉVALUATION du flux interaction (q-dépendant) : pose le ``robot_cloud``
@FK(q), évalue contre tous les canaux (``fields.eval_fields``) et construit les jacobiennes
géométriques ANALYTIQUES -> ``ContactEval``. Reference-free, cost-free : ``solve`` compose les
résidus/coûts. Tangente world-aligned (LOCAL_WORLD_ALIGNED), cohérente ``RobotModel.link_jacobians``.

Formules figées (spec) :
- Robot (point de contrôle, offset local ``o`` sur link ``(R, t)``, offset monde ``r = R @ o``) :
  ``point_jac = J_lin - [r]_× J_ang``  (somme pondérée sur K ; ``robot_cloud`` est K=1).
- Objet, probe ``x = R_iᵀ (p - t_i)`` vs tangente ``(δt, δθ)`` monde :
  ``∂x/∂δt = -R_iᵀ`` , ``∂x/∂δθ = R_iᵀ [p - t_i]_×``.
- Objet, nuage propre ``p = t_i + R_i o`` : ``∂p/∂δt = I₃`` , ``∂p/∂δθ = -[p - t_i]_×``.

Pur, array-oriented, torch-free (la cinématique lourde est cachée dans ``ctx.robot``). Ported de la
logique V1 ``contact/*`` côté valeur ; les jacobiennes sont neuves (analytiques).
"""
from __future__ import annotations

import numpy as np

from ...prepare.contracts import Channel, InteractionContext
from ..contracts import ContactEnvEval, ContactEval
from .fields import eval_fields
from .pointclouds import pose_cloud


def _skew(v: np.ndarray) -> np.ndarray:
    """(..., 3) -> (..., 3, 3) matrice antisymétrique ``[v]_×`` (``[v]_× a = v × a``)."""
    v = np.asarray(v, np.float64)
    z = np.zeros(v.shape[:-1])
    x, y, w = v[..., 0], v[..., 1], v[..., 2]
    return np.stack([
        np.stack([z, -w, y], axis=-1),
        np.stack([w, z, -x], axis=-1),
        np.stack([-y, x, z], axis=-1),
    ], axis=-2)


def _probe_jac(channels: tuple[Channel, ...], points: np.ndarray,
               object_rot: np.ndarray, object_pos: np.ndarray) -> np.ndarray:
    """``(C, P, 3, 6)`` ∂(probe dans le frame canal)/∂(tangente SE(3) de l'objet du canal). Canal sol
    (``object_idx is None``) -> lignes 0. Canal objet ``i`` : ``∂x/∂δt = -R_iᵀ``, ``∂x/∂δθ =
    R_iᵀ [p - t_i]_×`` (probe ``x = R_iᵀ(p - t_i)``, ``p`` = point monde tenu fixe)."""
    p_count = points.shape[0]
    out = np.zeros((len(channels), p_count, 3, 6))                  # (C, P, 3, 6)
    for c, ch in enumerate(channels):
        if ch.object_idx is None:
            continue                                               # ground rows stay 0
        i = ch.object_idx
        rit = np.asarray(object_rot[i], np.float64).T              # (3, 3) = R_iᵀ
        ti = np.asarray(object_pos[i], np.float64)                 # (3,)
        out[c, :, :, 0:3] = -rit[None, :, :]                       # ∂x/∂δt = -R_iᵀ (broadcast sur P)
        out[c, :, :, 3:6] = rit[None] @ _skew(points - ti)        # R_iᵀ [p - t_i]_×   (P, 3, 3)
    return out


def _env_eval(ctx: InteractionContext, i: int,
              object_rot: np.ndarray, object_pos: np.ndarray) -> ContactEnvEval:
    """Côté env pour le nuage objet ``i`` : champ (self_idx=i) + ``cloud_jac_self`` + ``probe_jac_obj``."""
    obj_world = pose_cloud(ctx.object_clouds[i], object_rot[i][None], object_pos[i][None])  # (P_i, 3)
    field = eval_fields(obj_world, ctx.channels, object_rot, object_pos, ctx.margin, self_idx=i)

    ti = np.asarray(object_pos[i], np.float64)                     # (3,)
    p_count = obj_world.shape[0]
    cloud_jac_self = np.zeros((p_count, 3, 6))                     # (P_i, 3, 6)
    cloud_jac_self[:, :, 0:3] = np.eye(3)[None]                    # ∂p/∂δt = I₃
    cloud_jac_self[:, :, 3:6] = -_skew(obj_world - ti)            # ∂p/∂δθ = -[p - t_i]_×

    probe_jac_obj = _probe_jac(ctx.channels, obj_world, object_rot, object_pos)  # (C, P_i, 3, 6)
    return ContactEnvEval(field=field, cloud_jac_self=cloud_jac_self, probe_jac_obj=probe_jac_obj)


def contact_eval(ctx: InteractionContext, q: np.ndarray,
                 object_rot: np.ndarray, object_pos: np.ndarray) -> ContactEval:
    """État de contact courant (robot) + jacobiennes pour ``(q, object_poses)`` -> ``ContactEval``.

    ``object_rot (N, 3, 3)`` / ``object_pos (N, 3)`` sont les poses objets monde courantes (mêmes que
    la réf). Pose le ``robot_cloud`` @FK(q) via ``ctx.robot.link_jacobians(q)`` (transforms + jac de
    frame monde), évalue contre ``ctx.channels``, et assemble ``point_jac`` (monde), ``probe_jac_obj``
    (frame canal) et le côté ``env`` (un par nuage objet)."""
    robot = ctx.robot
    rot, pos, jac_lin, jac_ang = robot.link_jacobians(q)           # (L,3,3),(L,3),(L,3,nv),(L,3,nv) monde
    cloud = ctx.robot_cloud
    parts = np.asarray(cloud.parts)                               # (M, K) into FK link order
    weights = np.asarray(cloud.weights, np.float64)              # (M, K)
    offsets = np.asarray(cloud.offsets, np.float64)             # (M, K, 3) link-local

    points = pose_cloud(cloud, rot, pos)                          # (M, 3) world robot control points

    r = np.einsum("mkij,mkj->mki", rot[parts], offsets)          # (M, K, 3) world offset link->point
    contrib = jac_lin[parts] - np.einsum("mkij,mkjn->mkin", _skew(r), jac_ang[parts])  # (M,K,3,nv)
    point_jac = np.einsum("mk,mkin->min", weights, contrib)      # (M, 3, nv) = J_lin - [r]_× J_ang

    field = eval_fields(points, ctx.channels, object_rot, object_pos, ctx.margin)       # (C, M)
    probe_jac_obj = _probe_jac(ctx.channels, points, object_rot, object_pos)            # (C, M, 3, 6)
    env = tuple(_env_eval(ctx, i, object_rot, object_pos) for i in range(len(ctx.object_clouds)))
    return ContactEval(field=field, point_jac=point_jac, probe_jac_obj=probe_jac_obj, env=env)
