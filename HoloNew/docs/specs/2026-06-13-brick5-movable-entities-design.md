# Brick 5 design: Movable entities as variables W^o

Part of [the TEST-SOCP paper-formulation meta-spec](2026-06-13-test-socp-paper-formulation-meta-spec.md).
The final, most structural brick: object poses become solved decision variables
(no longer driven), with mass-free motion tracking and bilateral robot↔object
interaction.

## Scope

Movable entities `T_m ∈ SE(3)` become decision variables, with the paper's
`W^o` (linear acceleration + spin tracking), bilateral robot↔object interaction
(D/X/P already from brick 1, now two-sided), object↔ground and object↔object
pairs (transitive transport), and entity↔entity non-penetration. Decision
variable extends to `[dqa_robot ; dξ_m]`.

## Decision-variable extension

`dqa → [dqa_robot ; dξ_m]` per movable entity `m`, where `dξ_m` is the SE(3)
tangent increment of the object pose; integration `T_m ← T_m ⊕ dξ_m` (SE(3) exp,
brick 0/2 tooling). The SOC trust region extends over the larger vector.

## The terms

- **`W^o`:** `W_m^o = (1/σ_a^o²)‖v̇_m − v̇_m^ref‖² + (1/σ_ω²)‖ω_m − ω_m^ref‖²`.
  Linear: acceleration only (position/velocity inherited from contacts; ballistic
  flight emerges). Angular: the spin `ω_m` itself. `V_m` from the log map (brick
  2), linearized in `dξ_m`. `λ^o` stays **weak** relative to the interaction
  weights — an object moves because it is carried, not scripted.
- **Bilateral interaction:** the brick-1 robot↔object `D/X/P` residuals now
  depend on `dξ_j` too (the object surface moves), via the full-decision-variable
  world-position Jacobian (meta-spec principle 3). Add object↔ground and
  object↔object pairs for transitive transport (robot→tray→cup).

## Constraints

Entity↔entity non-penetration (object↔ground, object↔object): `mj_geomDistance`'s
witness/normal gives the distance gradient wrt **both** bodies, so the existing
robot-only non-penetration linearization gains an object-DOF Jacobian column.

## Driving removed

Objects are no longer set from `_obj_poses_mj`; they are **initialized** at the
reference pose per frame, then solved. The reference object poses become the
`W^o` references (`v̇_m^ref`, `ω_m^ref`) and the persistence references.

## Activation gates (from brick 1, now bilateral)

The `α, γ, α̂` gates already designed in brick 1's persistence drive the behaviour:
an untouched object is held by persistence with its support; a grasped object is
transported bilaterally; on a missed grasp the hand↔object persistence never
activates and the object stays on its support.

## Config / weights

Add `lambda_o`, `sigma_a_o`, `sigma_omega`, plus object↔object pair selection
(which pairs, prefilter). Depends on bricks 0 (SE(3) tangent), 1 (object
interaction), 2 (twists `V_m`).

## Acceptance metric (validation gate)

1. Object pose error vs reference within tolerance on driven-equivalent frames.
2. Grasp = relative robot↔object motion pinned (bilateral persistence holds).
3. No object interpenetration (object↔ground / object↔object constraints hold).
4. Transitive transport works on a multi-object clip (robot→tray→cup), if such a
   clip is available.

Plus re-baselined regression snapshot + "runs without error" smoke.

## Risks

1. **Decision-variable growth** (+6 per object) → solver scaling; verify solve
   time on multi-object clips.
2. **Object↔object pairing**: which pairs, prefiltering (mirror the robot
   `_prefilter_pairs_with_mj_collision` approach for entity pairs).
3. **Object-DOF non-penetration Jacobian** correctness — unit-test against finite
   differences.
4. **Known limitation (paper):** without inertia, in long flight the angular
   momentum `L` is not conserved — `ω_m^ref` is copied from the reference, not
   simulated. Consistent with the kinematic scope; documented, not fixed.
