# Brick 2 design: Temporal regularization W^r (+ shared SE(3) tooling)

Part of [the TEST-SOCP paper-formulation meta-spec](2026-06-13-test-socp-paper-formulation-meta-spec.md).
Small brick whose main value is the **shared causal SE(3) finite-difference
tooling** reused by bricks 4 (centroidal) and 5 (movable entities), plus the
paper's temporal regularization term `W^r`.

## Scope

The two `W^r` terms (joint acceleration, base-frame twist acceleration), in
place in `TestSocpRetargeter`, decision variable `dqa`. The movable-entity twist
term is inactive here (objects are driven, not variables → constant) and becomes
active only in brick 5.

## The two terms

- **Joint acceleration (actuated DOF, Euclidean):**
  `q̈ = (q_a − 2 q_{a,t−1} + q_{a,t−2}) / Δt²`, penalize `‖q̈‖²`. In the SQP the
  current actuated joints are `q_a^0 + dqa`; past frames are constant. Residual
  `(q_a^0 + dqa) − 2 q_{a,t−1} + q_{a,t−2}`, linear in `dqa` — a trivial
  quadratic term. Applies to the actuated (R^n) joints only; the floating base is
  handled by the twist term.
- **Frame twist acceleration:** `V_F = (1/Δt) log(T_{F,t−1}⁻¹ T_F)^∨`, penalize
  `‖V_F − V_{F,t−1}‖²`. `V_{F,t−1}` is constant (from past poses). In brick 2 the
  only variable frame is the base `B`; linearize `V_B` against the base pose
  increment carried by `dqa` via the SE(3) log Jacobian.

`W^r = ‖q̈‖²/(σ_q̈² Δt⁴) + Σ_F ‖V_F − V_{F,t−1}‖²/(σ_V̇² Δt²)` (here `F = {B}`).

## Shared SE(3) tooling (the real deliverable)

A small causal-kinematics module (e.g. `src/test_socp/kinematics.py`) providing:
- SE(3) `log`/`exp`, the `∨` operator;
- twist from two poses `V = (1/Δt) log(T_prev⁻¹ T_cur)^∨`;
- the **log Jacobian** `∂ log(T_prev⁻¹ T_cur) / ∂(current pose increment)` for
  linearizing the twist in `dqa`;
- Euclidean `q̇ / q̈` finite differences.

**Implemented with pinocchio** (`pin.log6`, `pin.Jlog6` on raw `pin.SE3`
objects, without building the full robot model). This seeds the pinocchio usage
of brick 4 while deferring the pinocchio↔MuJoCo model-alignment question to that
brick (raw spatial algebra needs no model).

## Data flow

`retarget` already threads `q_{t−1}` (`q_prev`, for foot-sticking); add
`q_{t−2}`. For `t < 2`, the acceleration terms are skipped (or use a replicated
first frame), defined in the plan. `W^r` terms are assembled in
`solve_single_iteration`, evaluated at the current `q`, expressed in `dqa`.

## Config / weights

Add `lambda_r`, `sigma_qddot`, `sigma_Vdot` to `TestSocpRetargeterConfig`.
Defaults regularize the null space (free limbs, internal motions) without
fighting tracking; distinct from the SOC trust region (which governs
linearization validity, not motion quality).

## Acceptance metric (validation gate)

vs the pre-brick baseline:
1. RMS joint jerk / acceleration **reduced**.
2. Base twist acceleration **reduced**.
3. Tracking error (existing position/orientation metric) **not degraded** beyond
   a small tolerance.

Plus re-baselined regression snapshot + "runs without error" smoke on a full
OMOMO clip.

## Risks

1. **SE(3) log-Jacobian linearization** of the base twist: the SOC trust region
   should absorb it; confirm no oscillation. Using pinocchio's `Jlog6` keeps the
   derivative exact.
2. **Boundary frames** (`t = 0, 1`): the second difference is undefined; the plan
   defines the warm-up behaviour (skip / replicate).
3. **pinocchio as a new dependency**: install in the `holonew` env; the brick's
   first task verifies `import pinocchio` works there. (Approved dependency.)
