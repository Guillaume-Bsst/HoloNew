"""assemble — (evals + refs + geo + robot + cfg) -> ``Problem``. The bridge between current
EVALUATIONS (``FrameEval``: style FK + contact field) and the linearized QP subproblem: calls
``terms/`` builders (folded weights) and ``terms/constraints`` (joint limits + trust-region box),
concatenates into ONE ``Problem``. PURE — no kinematics here (delegated to Eval), no cvxpy.

``geo`` = geodesic context per channel that ``build_contact`` reads for witness residual C-X (sourced
from ``ctx.channels`` on runner side — each ``Channel`` carries its ``geodesic`` + ``sdf``)."""
from __future__ import annotations

import numpy as np

from .contracts import Problem, TrustRegion
from .config import SolveConfig
from .terms._ops import GeoField
from .terms.style import build_style
from .terms.contact import build_contact
from .terms.object import build_object
from .terms.reg import build_reg
from .terms.constraints import build_constraints


def _geo_field(channels, object_rot, object_pos) -> GeoField:
    """Per-frame bundle build_contact needs: object-local frames + geodesic tables per channel.
    Channel.object_idx is None for ground -> -1 / identity frame (GeoField convention)."""
    tables     = tuple(ch.geodesic for ch in channels)
    rot        = np.stack([np.eye(3) if ch.object_idx is None
                           else object_rot[ch.object_idx] for ch in channels])   # (C,3,3)
    pos        = np.stack([np.zeros(3) if ch.object_idx is None
                           else object_pos[ch.object_idx] for ch in channels])   # (C,3)
    object_idx = tuple(-1 if ch.object_idx is None else ch.object_idx for ch in channels)
    return GeoField(tables=tables, rot=rot, pos=pos, object_idx=object_idx)


def assemble(evals, frame_targets, geo, robot, cfg: SolveConfig) -> Problem:
    """Builds the ``Problem`` for ONE SQP iteration. ``n_obj`` is derived from frame object poses;
    if ``n_obj = 0`` the object builders simply produce no ``A_obj`` blocks."""
    se, ce = evals.style, evals.contact
    blocks = []
    blocks += list(build_style(se, frame_targets.style, cfg))
    geo_field = _geo_field(geo, frame_targets.object_rot, frame_targets.object_pos)
    blocks += list(build_contact(ce, frame_targets.robot_interaction, geo_field, cfg))
    blocks += list(build_object(ce, frame_targets.env_interaction,
                                frame_targets.object_rot, frame_targets.object_pos, cfg))
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
