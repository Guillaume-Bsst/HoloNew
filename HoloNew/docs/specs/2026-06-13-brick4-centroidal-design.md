# Brick 4 design: Centroidal component W^c / W^L (pinocchio)

Part of [the TEST-SOCP paper-formulation meta-spec](2026-06-13-test-socp-paper-formulation-meta-spec.md).
Track CoM acceleration and centroidal angular momentum, and remove the
transitional pelvis anchor so the pelvis becomes inferred from momentum rather
than positionally targeted. Builds on the pinocchio backend established in
brick 0.

## Scope

Add `W^c` (CoM acceleration) and `W^L` (centroidal angular momentum) tracking;
remove the brick-3 pelvis translation scaffold. Decision variable `dqa`
(pinocchio tangent). Uses pinocchio for CoM Jacobian and the centroidal map —
the pinocchio↔MuJoCo alignment is already solved and validated in brick 0.

## The terms

- **`W^c` (CoM acceleration):** `c̈ ≈ (c − 2 c_{t−1} + c_{t−2}) / Δt²`. Linearize
  `c(ξ) = c_0 + J_c dqa` with the CoM Jacobian `J_c = pin.jacobianCenterOfMass`;
  past frames constant → quadratic in `dqa`.
- **`W^L` (centroidal angular momentum):** `L` is the angular part of
  `A_G [V_B; q̇]`. `V_B` (log Jacobian, brick-2 tooling) and
  `q̇ = (q_a^0 + dqa − q_{a,t−1})/Δt` are linear in `dqa`; freeze `A_G` at the
  current config (`pin.ccrba` / centroidal map) → `L` linear in `dqa`, quadratic
  cost.
- `W^c = (1/σ_a²)‖c̈ − c̈^ref‖²`, `W^L = (1/σ_L²)‖L − L^ref‖²`.
- **Linear/angular asymmetry:** only `c̈` is tracked (absolute CoM velocity is
  arbitrary); `L` itself is tracked. The ballistic CoM in flight emerges
  (`c̈^ref → −g ẑ`, contained in the data, never injected).

## Pelvis scaffold removal

Brick 3's `lambda_pelvis_anchor → 0`. The pelvis translation is now inferred
from `W^c` + contacts. This completes the paper's "pelvis inferred, not placed".

## Reference momentum

Compute `c^ref`, `L^ref` from the **reference robot trajectory** (the ground
reference mapped to the robot) via the **same pinocchio model** — keeps
units/morphology consistent — rather than from the SMPL-X source (morphology
mismatch). Causal finite differences for `c̈^ref`.

## Config / weights

Add `lambda_c`, `lambda_L`, `sigma_a`, `sigma_L` to `TestSocpRetargeterConfig`.

## Acceptance metric (validation gate)

1. `c̈` tracking error within tolerance.
2. `L` tracking error within tolerance.
3. On flight frames, `c̈ ≈ −g ẑ` (ballistic CoM emerges).
4. The pelvis stays sane with the scaffold removed (`lambda_pelvis_anchor = 0`),
   solve feasible and finite.

Plus re-baselined regression snapshot + "runs without error" smoke.

## Risks

1. **CoM / `A_G` correctness** — unit-test `J_c` and the centroidal map against
   finite differences before trusting the linearization. (Alignment itself is
   already validated in brick 0.)
2. **Scaffold removal** could destabilize the pelvis if momentum tracking is too
   weak — tune `λ^c/λ^L`, gated by the metric; reintroduce a tiny scaffold only
   if needed.
3. **Reference momentum quality** — the reference robot trajectory's CoM/momentum
   must be smooth enough that `c̈^ref`/`L^ref` are meaningful; verify on a clip.

## Validation outcome (Task 4, 2026-06-14) — DONE_WITH_CONCERNS

Tuned parameters: `lambda_c=3.0`, `lambda_L=0.5`, `pelvis_anchor_weight=1.0`,
30-frame robot_only sub3_largebox_003, activate_style=True.

Results (A = centroidal off, B = centroidal on):
- **CoM-accel error:** A=7.9028 m/s^2, B=0.0060 m/s^2 (1314x reduction). PASS.
- **Pelvis z:** B=[0.622, 0.800] m, within [0.4, 1.0] m. PASS.
- **Pelvis xy drift:** B=0.228 m max vs A=0.022 m max (10x worse despite full
  scaffold at pelvis_anchor_weight=1.0). This is an inherent limitation: W^c
  tracks CoM acceleration (second difference) not absolute CoM position. Once
  the Style objective drops joint position tracking, the pelvis can drift in xy
  to satisfy the acceleration profile. Increasing the scaffold to reduce drift
  (pelvis_anchor_weight >= 5.0) degrades CoM-accel tracking and pushes pelvis z
  below 0.5 m. There is no scaffold level that keeps xy drift < 0.05 m and
  preserves the centroidal benefit + sane z. FAIL.

Decision: `activate_centroidal` remains **False** by default. The W^c/W^L terms
are validated as mathematically correct and dramatically reduce CoM-acceleration
error, but they cause pelvis xy drift (~0.23 m) that the position scaffold cannot
cure without sacrificing the centroidal benefit. The feature is available behind
the flag for research use. A future fix would add a CoM absolute-position term
(e.g., W^c_pos = lambda_c_pos * ||c - c_ref||^2) alongside W^c to anchor the
integration, or use a Model Predictive Control formulation that tracks the full
CoM trajectory rather than only its second difference.
