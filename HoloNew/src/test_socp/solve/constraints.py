"""Numpy LinearConstraint / TrustRegion builders extracted from solve_single_iteration.

Same math as the inline cvxpy constraints, returning solver-agnostic blocks.
"""
from __future__ import annotations

import numpy as np

from .spec import LinearConstraint, TrustRegion


def trust_regions(rt, n_obj: int) -> list[TrustRegion]:
    trs = [TrustRegion("dqa", float(rt.step_size))]
    if n_obj:
        trs.append(TrustRegion("dxi", float(rt.step_size)))
    return trs


def box_freeze_limits(rt, q_pin) -> list[LinearConstraint]:
    """DOF freeze (activate_tb / activate_qa) + actuated joint-limit box, as LinearConstraints."""
    cons: list[LinearConstraint] = []
    nv_a = rt.nv_a
    eye = np.eye(nv_a)
    if not rt.activate_tb:
        base = np.where(rt.v_a_indices < 6)[0]
        if base.size:
            z = np.zeros(base.size)
            cons.append(LinearConstraint(A=eye[base], lb=z, ub=z, name="freeze_base"))
    if not rt.activate_qa:
        joints = np.where(rt.v_a_indices >= 6)[0]
        if joints.size:
            z = np.zeros(joints.size)
            cons.append(LinearConstraint(A=eye[joints], lb=z, ub=z, name="freeze_joints"))
    if rt.activate_joint_limits:
        lo = np.copy(rt._v_a_lb)
        hi = np.copy(rt._v_a_ub)
        joint_mask = rt.v_a_indices >= 6
        vi_joints = rt.v_a_indices[joint_mask]
        q_pin_vals = np.asarray(q_pin)[vi_joints + 1]
        lo[joint_mask] -= q_pin_vals
        hi[joint_mask] -= q_pin_vals
        cons.append(LinearConstraint(A=eye, lb=lo, ub=hi, name="joint_limits"))
    return cons
