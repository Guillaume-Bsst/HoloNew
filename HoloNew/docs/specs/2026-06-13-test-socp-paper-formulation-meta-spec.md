# Meta-spec: evolving TEST-SOCP toward the paper's kinematic retargeting formulation

## Goal

Incrementally evolve the `TestSocpRetargeter` objective into the formulation of
*Kinematic Retargeting Formulation for the G1 Robot* (G. Besset, June 2026):
style (pelvis-relative), uniform interaction costs (normal proximity `D`,
tangential placement `X`, contact persistence `P`), centroidal momentum
(`W^c`, `W^L`), movable entities as variables (`W^o`), and temporal
regularization (`W^r`), under the existing non-penetration / self-collision /
joint-limit constraints.

This is a **decomposition / meta-spec**: it fixes the global breakdown,
ordering, shared interfaces, and validation protocol. **Each brick gets its own
spec → plan → implementation cycle**; this document is not itself an
implementation plan.

## Scope

In scope: all five paper components, applied to the G1 on the existing OMOMO
clips, **in place** in `src/test_socp/test_socp.py` (+ its `contact/` and
`correspondence/` subpackages). Movable-entities-as-variables (`W^o`) is in
scope but scheduled **last**. Out of scope here: any change to the other solvers
(GMR-SOCP, holosoma `InteractionMeshRetargeter`), which stay untouched.

## Cohabitation model: incremental replacement

The paper's objective **replaces** the current TEST-SOCP objective (world-frame
pose tracking via the IK match tables) brick by brick. We do **not** keep a
bit-exact parity invariant: once a brick is validated, its behaviour becomes the
new default. The paper's Style component is an *alternative* to the current
world-frame tracking, not an addition — so a parallel/opt-in mode is explicitly
rejected in favour of staged replacement.

## Guiding principles

1. **In place, incremental replacement.** Each brick evolves the objective;
   once validated (metric + re-baselined snapshot + smoke, see below) it is the
   new default. No parallel "paper mode".
2. **Per-frame SQP structure preserved.** The solver stays a linearised step in
   `dqa` with an SOC trust region; past frames are constant. All new costs
   (`D, X, P, W^c, W^L, W^r, W^o`) are causal, hence quadratic in the current
   step's decision variable. No solver-paradigm change.
3. **Decision variable evolves.** Bricks 1–4 keep `dqa` = robot active DOF only
   (objects driven per frame, as today). Brick 5 (`W^o`) extends it to
   `[dqa ; dξ_objects]`. New costs are written against a **world-position
   Jacobian with respect to the full decision variable**, so extending to
   objects stays localized.
4. **Transitional pelvis anchor (scaffold).** The paper never positionally
   targets the pelvis — but that only holds once the centroidal brick is in.
   Between Style (roll/pitch anchor only) and Centroidal, **keep a weak pelvis
   position anchor** as scaffolding, removed when Centroidal lands. Prevents an
   under-constrained pelvis drifting between bricks.
5. **Weights/scales in the per-solver config.** The `λ•` weights and `σ•`
   scales live in `TestSocpRetargeterConfig` (or a dedicated nested sub-config).
   Defaults are chosen so each landed brick is active.

## The five bricks (recommended order)

### Brick 1 — Interaction `D / X / P`
Wire the **already-computed** fields (object+floor SDF via `contact/backends`,
optimal-transport correspondence via `correspondence/`, transported control
points) into the objective. `D`: normal proximity `α(d_ref − d)²`. `X`:
tangential placement `α‖Π(x_ref − x)‖²`. `P`: persistence `γ‖Π(Δp − Δp_ref)‖²`
with activations `α, γ, α̂`. Decision variable: `dqa` unchanged (objects driven).
Dependencies: none. **Acceptance metric:** max penetration ≤ ε on contact
frames; `D`/`X` residuals reduced vs the pre-brick baseline; `P` slip ≈ the
reference tangential displacement.

### Brick 2 — Temporal regularization `W^r`
Add the `q̈` and `V̇_F` terms. Small, but it **introduces the shared causal
finite-difference tooling** on SE(3) (log/exp, twist `V_F`, `q̇`/`q̈`) reused by
bricks 4 and 5. Decision variable: `dqa`. Dependencies: none (foundation).
**Acceptance metric:** RMS jerk/acceleration reduced; tracking error not
degraded.

### Brick 3 — Pelvis-relative Style
Replace world-frame tracking with pelvis-relative joint orientation
(`R̃_k = R_B⁻¹ R_k`) plus a **roll/pitch-only** pelvis anchor (yaw free). Keep
the weak pelvis position anchor (scaffold, principle 4). Decision variable:
`dqa`. Dependencies: brick 1 (contacts hold the base while yaw is freed).
**Acceptance metric:** pelvis roll/pitch tracking error; yaw demonstrably freed;
joint-orientation fidelity preserved.

### Brick 4 — Centroidal `W^c / W^L`
Build CoM + centroidal momentum matrix `A_G` tooling **with pinocchio** (an
approved dependency), then track CoM acceleration `c̈` and centroidal angular
momentum `L`. **Remove the pelvis scaffold**: the pelvis becomes inferred from
momentum. Decision variable: `dqa`. Dependencies: brick 2 (finite differences),
brick 3 (completes "pelvis inferred, not placed"). **Acceptance metric:** `c̈`
tracking error; `L` tracking; ballistic CoM (`c̈ → −g ẑ`) on flight frames.

### Brick 5 — Movable entities as variables `W^o`
Extend the decision variable to `[dqa ; dξ_objects]`. Bilateral robot↔object
persistence, transitive transport (robot→tray→cup), `W^o` (acceleration + spin,
mass-free `ω` proxy). Dependencies: brick 1 (object interaction), brick 2
(twists `V_m`); structural. **Acceptance metric:** object pose error vs
reference; grasp = relative robot–object motion pinned; no object
interpenetration.

## Shared tooling (cross-brick interfaces)

- **Causal SE(3) finite-difference module** (log/exp, twist, `q̇`/`q̈`):
  created in brick 2, consumed by bricks 4 and 5.
- **Field / correspondence references**: already exist (`contact/`,
  `correspondence/`); brick 1 consumes them; brick 5 extends them to object
  carriers.
- **Full-decision-variable world-position Jacobian**: introduced in brick 1,
  generalized in brick 5 to cover object pose DOFs.
- **Centroidal tooling (pinocchio)**: `A_G`, CoM, CoM Jacobian — introduced in
  brick 4, available thereafter.

## Validation protocol (per brick)

Because bit-exact parity is dropped, every brick is gated by three checks:

1. **Targeted quantitative metric** — the brick-specific metric listed above;
   the test fails if it regresses.
2. **Re-baselined regression snapshot** — a golden output snapshot,
   **deliberately updated** when the brick lands (guarding against accidental
   breakage: crash, NaN, shape drift), kept distinct from the intentional
   behaviour change.
3. **"Runs without error" smoke** — the brick, enabled, solves a full clip,
   feasible, finite output.

(Visual review of the rendered motion on the OMOMO clips remains the final human
judgement but is not part of the automated gate.)

## Risks and decisions deferred to per-brick specs

- **Linearization fidelity.** `D/X/P` and centroidal costs are nonlinear; the
  per-frame SQP linearizes them. Inner-iteration count / trust-region size may
  need tuning — decided in each brick's spec, not here.
- **`A_G` via pinocchio** (brick 4): the pinocchio↔MuJoCo model alignment
  (joint order, root convention) is the brick-4 spec's first task.
- **Snapshot churn.** Incremental replacement means the default output changes
  each brick; the re-baselined snapshot must be updated intentionally and
  reviewed, never silently.
- **Weight tuning.** The `λ•`/`σ•` defaults that make each brick "active"
  without destabilizing the solve are tuned per brick.

## Process

Each brick is a separate spec → plan → implementation cycle, in the order above.
After this meta-spec is approved, the next step is to brainstorm **Brick 1
(Interaction D/X/P)** into its own design.
