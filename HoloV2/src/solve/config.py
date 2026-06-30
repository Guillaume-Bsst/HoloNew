"""Config of the ``solve`` stage — the QP KNOBS (one frozen, stdlib-only dataclass), co-located with
the stage (rule #2). ``SolveConfig()`` IS the default; override inline
(``SolveConfig(w_cd=5.0, tr_joints=0.2)``), exactly like ``targets.config.TargetsConfig``.

Three knob families:
  * per-term WEIGHTS (``w_pos`` … ``w_reg``) — the cost gains folded into each ``ResidualBlock``'s
    ``A``/``c`` by the ``solve/terms`` builders (the #1 tuning lever; cf. ``FrameInfo.cost_by_term``);
  * contact ACTIVATION — ``contact_gate`` (rows only for demonstrated-active pairs) + a soft
    ``contact_d_ref_scale`` falloff that down-weights far demonstrated contacts (the V1 ``alpha``);
  * trust region + loop — per-DOF box radii (heterogeneous units: base m / base rad / joints rad /
    object m+rad) and the SQP iteration budget / convergence tol / backend name.

Per-link / per-channel weight VECTORS are a future refinement; v1 weights are scalars."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SolveConfig:
    """All knobs of the ``solve`` QP loop. Frozen, stdlib-only, importable anywhere."""

    # --- per-term weights (folded into ResidualBlock A and c) ---------------------------------
    w_pos: float = 1.0     # S-pos : style link position tracking
    w_rot: float = 0.5     # S-rot : style link orientation tracking
    w_cd: float = 2.0      # C-D   : robot contact distance (vs channel)
    w_cx: float = 1.0      # C-X   : robot contact geodesic (witness on the surface)
    w_cod: float = 1.0     # CO-D  : object self-contact distance (object vs ground/other objects)
    w_cox: float = 0.5     # CO-X  : object self-contact geodesic (DEFERRED in v1, see plan)
    w_obj: float = 1.0     # O     : object pose anchor to its observed pose
    w_reg: float = 1e-2    # reg   : step damping (well-conditioned QP)

    # --- contact activation ------------------------------------------------------------------
    contact_gate: bool = True        # rows only for pairs active in the demonstrated (reference) field
    contact_d_ref_scale: float = 0.05  # soft falloff: weight *= exp(-(max(d_ref,0)/scale)^2);
                                       # <= 0 disables the falloff (active pairs weight 1)

    # --- per-DOF box trust-region radii (TrustRegion.radius, per-DOF, norm=-1) ----------------
    tr_base_pos: float = 0.05   # free-flyer translation step (m)   -> v[0:3]
    tr_base_rot: float = 0.10   # free-flyer rotation step (rad)    -> v[3:6]
    tr_joints: float = 0.10     # actuated joint step (rad)         -> v[6:6+dof]
    tr_object_pos: float = 0.05  # object translation step (m)      -> δξ[0:3] per object
    tr_object_rot: float = 0.10  # object rotation step (rad)       -> δξ[3:6] per object

    # --- SQP loop ----------------------------------------------------------------------------
    n_iter_first: int = 10       # iterations for the cold-start frame (absorbs joint refinement)
    n_iter_per_frame: int = 4    # iterations for warm-started frames
    step_tol: float = 1e-4       # convergence: ‖dv‖ < step_tol
    backend: str = "cvxpy"       # solve backend (Plan A factory key)
    robot_name: str | None = None  # optional label forwarded by runner.solve (no validation)

    def __post_init__(self) -> None:
        for name in ("w_pos", "w_rot", "w_cd", "w_cx", "w_cod", "w_cox", "w_obj", "w_reg"):
            if getattr(self, name) < 0.0:
                raise ValueError(f"SolveConfig.{name} must be >= 0, got {getattr(self, name)}")
        for name in ("tr_base_pos", "tr_base_rot", "tr_joints", "tr_object_pos", "tr_object_rot"):
            if getattr(self, name) <= 0.0:
                raise ValueError(f"SolveConfig.{name} must be > 0, got {getattr(self, name)}")
        if self.step_tol <= 0.0:
            raise ValueError(f"SolveConfig.step_tol must be > 0, got {self.step_tol}")
        if self.n_iter_first < 1 or self.n_iter_per_frame < 1:
            raise ValueError("SolveConfig.n_iter_* must be >= 1")
        if self.backend not in ("cvxpy",):
            raise ValueError(f"SolveConfig.backend must be 'cvxpy', got {self.backend!r}")
