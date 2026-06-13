# Brick 1 design: Interaction costs D / X / P (object + floor)

Part of [the TEST-SOCP paper-formulation meta-spec](2026-06-13-test-socp-paper-formulation-meta-spec.md).
First brick: wire the paper's uniform interaction costs into the TEST-SOCP
objective for the object SDF and the floor, using the fields and correspondence
that are already computed per frame but currently only recorded.

## Scope

Object SDF + floor entities only. Self-contact (robot links as entities) is
deferred to a later sub-brick. Objects are still **driven** per frame (their
poses are known, not solved) — `W^o` is brick 5. Decision variable stays
`dqa` (robot active DOF).

## What already exists (consumed, not rebuilt)

- `ContactField` (`contact/contact_field.py`) returns, per probe point: `distance`
  (signed `d`), `direction` (unit normal `n`, surface→probe), `witness` (closest
  surface point `x`), `active` (within margin `L`).
- Object SDF backend (`contact/backends/sdf.py`, trilinear `query`) and the
  analytic floor field (`contact/backends/floor.py`).
- Source references: the SMPL-X source placed in the robot scene and queried
  against the same fields — recorded today as `human_obj_dist` (`d_ref`),
  `human_witness` (`x_ref`), `human_flr_dist`.
- Optimal-transport correspondence (`correspondence/`): for each robot control
  point `i`, its link (`link_idx`), local offset (`offset_local`), and the source
  human point that drives it (`human_idx`). `transported_points` gives robot
  control-point world positions.
- Point Jacobian `_calc_contact_jacobian_from_point` and the SQP solver
  (`solve_single_iteration`, decision variable `dqa`, SOC trust region).

## What is new (the core of this brick)

**Query the entity fields at the robot control points each frame** to get the
robot-side `(d_0, x_0, n_0, active)`, then linearize the D/X/P residuals against
the point Jacobians and add them to the objective. Today the fields are queried
only on the source side.

## Residual formulation (orthogonal-projection approach)

Per SQP iteration, freeze the unit normal `n_0` and the activation `α` at the
current config. For control point `i` and entity `j`, with point Jacobian `J_i`
(3×nq) and the decision step `dqa`, the linearized control-point displacement is
`r = J_i dqa`. Decompose it orthogonally:

- **D (normal proximity):** residual `(d_ref − d_0) − n_0ᵀ J_i dqa`, weighted
  `α / L_j²`. Uses `∇d = n`, so `d(ξ) ≈ d_0 + n_0ᵀ J_i dqa`.
- **X (tangential placement):** residual `Π_0 [ (x_ref − x_0) − J_i dqa ]`,
  `Π_0 = I − n_0 n_0ᵀ`, weighted `α / L_j²`.
- Activation `α = (1 − d_ref / L_j)²`, clamped ≥ 0; a point contributes only when
  the source was within `L_j` of the surface (locality is automatic).
- Aggregation: `1/N_k` over the control points of carrier `k`, summed over
  entities; single weights `λ^D, λ^X` shared across entities.

**P (contact persistence):** residual `Π_0 [ Δp_i − Δp_i^ref ]`, weighted
`γ / (σ_v Δt)²`, with `Δp_i = p_i(ξ) − p_{i,t−1}` and `Δp_i^ref = p_i^ref −
p_{i,t−1}^ref`. Linearized: `Δp_i = (p_i^0 − p_{i,t−1}) + J_i dqa`. Activation
`γ = min(α, α^{t−1}, α̂^{t−1})`, where `α̂^{t−1} = (1 − d_{i,t−1}/L_j)²` uses the
**solved** robot-side distance at `t−1` (persistence never acts on a missed
contact). All activations are constants in the subproblem.

References are expressed in the entity's local frame (paper); the object pose is
known per frame, so `x_ref` / `n` are transported by the object pose. Floor is
world-frame.

## Where it plugs in

In `solve_single_iteration`, add the D/X (and P) terms to `obj_terms` before
`cp.Problem`, evaluated at the current full config `q`, expressed in `dqa` — the
same pattern as the existing position/orientation tracking terms. The existing
world-frame tracking stays (it is replaced only by brick 3, Style).

## Internal staging (for the implementation plan)

- **1a — D + X:** per-frame, no cross-frame state. The robot-side field query,
  the point Jacobians, the orthogonal-projection residuals, the config weights.
- **1b — P (persistence):** adds cross-frame state — previous robot control-point
  world positions `p_{i,t−1}`, previous source activation `α^{t−1}`, and the
  previous **solved** robot-side distance for `α̂^{t−1}`. Threaded through
  `retarget` like the existing `q_prev`.

## Config / weights

Add to `TestSocpRetargeterConfig` (or a nested interaction sub-config):
`lambda_D`, `lambda_X`, `lambda_P`, `sigma_v`. Entity ranges `L_j` come from the
existing field margins. Defaults strong enough to influence contacts without
destabilizing the base-pose tracking (which still dominates the pelvis until
bricks 3/4).

## Acceptance metric (validation gate)

On active contact frames, vs the pre-brick baseline:
1. Mean `|d(ξ) − d_ref|` over active robot contacts **reduced** (D works).
2. Mean `‖Π(x_ref − x(ξ))‖` **reduced** (X works).
3. Max penetration ≤ ε (already guaranteed by the non-penetration constraint;
   confirm D/X do not push points through the surface).
4. P: robot tangential slip tracks the reference tangential displacement (mean
   `‖Π(Δp − Δp_ref)‖` small on persistent contacts).

Plus the standard gate: re-baselined regression snapshot + "runs without error"
smoke on a full OMOMO clip.

## Validation findings (2026-06-13, default still OFF)

The D/X/P machinery is implemented, vectorized (~10x), and the math is
independently verified (numpy↔cvxpy equivalence at rtol 1e-8; Jacobians
FD-validated). Empirical enablement on `object_interaction sub3_largebox_003`
(first frames, floor channel — the object channel is inactive at rest) surfaced:

- **D (normal proximity) is conditionally unstable at high weight.** With
  `lambda_D=1`, the floating base runs away downward (base z dives to ~-36 m over
  6 frames): as the base sinks below the floor the penetrating points flip the
  field normal and reinforce the sink (positive feedback). At `lambda_D=0.05` the
  solve is stable and the floor gap slightly improves (0.019 vs 0.021 off).
- **X (tangential placement) is stable** (`lambda_X=1`, base stays ~0.7 m, gap
  ~0.03).
- The **floor metric is uninformative**: GMR tracking already keeps the feet at
  the floor (gap ~0.02 off), so floor interaction adds little. The real value is
  the **object channel**, which needs manipulation-frame validation (solve up to
  the contact frames).

**Resolution (D/X enabled).** The D downward runaway was diagnosed: the D
objective is over-determined (hundreds of per-point distance targets on rigid
feet), so the trust-region SQP never converges (takes the max 0.2 step every
iteration) and the small net downward bias accumulates; once the base dips below
the floor the penetrating points flip the field normal and the sink accelerates.
The fix is **paper-faithful**: the interaction costs require the non-penetration
constraint (`d_ij >= 0`), which the paper's optimization has alongside the costs.
`from_config` now **couples** them — when interaction is active on an object task
it auto-enables **ground** non-penetration on the plain model (`load_object_scene
= False`; the soft D term handles object contact, so loading the object scene and
adding the hard object constraint is unnecessary and actually trips CLARABEL).
With ground non-penetration, D/X are stable (base z stays ~0.7), the floor gap
improves (~0.012 vs ~0.028 off), and a full clip solves in ~3 min. `lambda_D` and
`lambda_X` default to `1.0`.

**P resolved (enabled, renormalized).** The persistence term was numerically
explosive: its paper normalization `1/(sigma_v*dt)^2` (~3.6e5 with sigma_v=0.05,
dt=1/30) made it ~3600x the D/X terms and tripped CLARABEL once the cross-frame
state engaged (frame 2+), even at `lambda_P=0.05`. P is a tangential
meter-residual exactly like X, so we **renormalize it by the field range `L^2`**
(the same scale as X) — a deliberate, spirit-preserving divergence: it keeps the
paper's intent (reproduce the source's tangential slide / no-slip) while making
`lambda_P` directly comparable to `lambda_X` and the solve well-conditioned. The
per-frame slide is already bounded by the SQP trust region, so the paper's much
tighter `sigma_v*dt` tolerance is unnecessary. With the `L^2` scale, `lambda_P =
1.0` is stable (validated over a clip with D/X + ground non-penetration; the
contact gap stays ~0.011). `sigma_v` is kept in the config for API compatibility
but no longer affects the scale. The numpy↔cvxpy equivalence test is updated to
the `L^2` normalization.

## Risks

1. **Weighting vs the existing world tracking** (which currently dominates the
   solve): `λ^D/λ^X` must be large enough to bend contacts, small enough not to
   fight the base tracking. Tuned during 1a.
2. **Robot-side query cost** per frame (`N` control points × 2 entities): object
   SDF is trilinear, floor analytic — cheap, but verify on a full clip.
3. **Correspondence mapping**: each robot control point `i` must read the
   reference field value for its source point `human_idx[i]` and the correct
   entity; verify the indexing end to end.
4. **Frozen-normal linearization** near SDF edges/corners (normal discontinuity):
   the SQP trust region should absorb it; confirm no oscillation.
