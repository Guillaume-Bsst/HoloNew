# Brick 0 design: MuJoCo → pinocchio kinematics migration

Part of [the TEST-SOCP paper-formulation meta-spec](2026-06-13-test-socp-paper-formulation-meta-spec.md).
Front-loaded foundation: migrate the kinematics/Jacobian layer of
`TestSocpRetargeter` from MuJoCo to pinocchio so all later bricks share one
rigid-body source of truth, with no MuJoCo↔pinocchio convention mismatches
downstream.

## Scope

Replace TEST-SOCP's rigid-body kinematics/dynamics layer with pinocchio. Keep
MuJoCo + coal **only** for collision/SDF queries. This is a migration: no new
cost terms, no new behaviour — but the default solve is re-baselined (the
backend change + cleaner tangent-space integration may shift the output
slightly).

## What moves to pinocchio

- Build a pinocchio g1 model from the URDF with a **free-flyer** root.
- Replace: `body_position`, `body_rotation`, `_body_jac`,
  `_get_robot_link_positions`, `_calc_contact_jacobian_from_point`,
  `_calc_manipulator_jacobians`. `_build_transform_qdot_to_qvel_fast` likely
  disappears (pinocchio's tangent space handles the quaternion bridge natively).
- CoM / CoM Jacobian become available here (`pin.centerOfMass`,
  `pin.jacobianCenterOfMass`), used later by brick 4.

## What stays MuJoCo / coal

Non-penetration (`mj_geomDistance`), self-collision (`mj_collision`), per-frame
object qpos driving, and the SDF/contact fields (coal). These are
collision/distance queries pinocchio does not provide.

## The alignment seam (the crux)

- MuJoCo `qpos` (g1): `[pos(3), quat wxyz(4), joints(29)]` (+ object free joint
  when a scene xml is loaded). pinocchio free-flyer `q`:
  `[pos(3), quat xyzw(4), joints(29)]`. Conversion `qpos_mj ↔ q_pin`: reorder the
  quaternion (wxyz↔xyzw) and **map joints by name** (MuJoCo joint order vs
  pinocchio joint order).
- The decision variable `dqa` now lives in the **pinocchio tangent space**
  `[v_base(6); v_joints(29)]`; integration is `pin.integrate(q, dqa)`; the SOC
  trust region applies in that space.
- The holosoma-style constraint Jacobians (non-penetration, self-collision) flow
  through the migrated point-Jacobian helper, so they end up in the same
  pinocchio tangent space automatically — no mixed-space slicing.
- Joint-limit constraints (`q_a_lb/ub`, `q_a_indices`) are re-expressed against
  the pinocchio configuration/tangent indexing.

## Validation (the gate)

1. **Kinematic parity (alignment proof):** for sampled configs, body
   positions/orientations and body Jacobians from pinocchio match MuJoCo
   (`xpos`, `xmat`, `mj_jac`) within a tight tolerance. This is the
   load-bearing test — it proves the seam is correct.
2. **End-to-end re-baselined snapshot:** the full TEST-SOCP solve on an OMOMO
   clip; deliberately re-baseline (the output may shift slightly), and confirm
   tracking quality is preserved (the existing tracking metric stays within
   tolerance).
3. **Smoke:** full clip, feasible, finite.

## Config

No new weights — this is a backend migration. A pinocchio model path / build is
added to the solver setup.

## Risks

1. **Touches the working solver's core.** The `q`-convention, tangent-space
   integration, joint-limit indexing, and constraint-Jacobian slicing all change
   at once. The kinematic-parity test + re-baseline + smoke gate it; the plan
   migrates one helper at a time, re-running parity after each.
2. **Quaternion + joint-order mapping** is the most error-prone seam — unit-test
   `qpos_mj → q_pin → qpos_mj` round-trip and the per-joint name mapping first.
3. **Tangent-space trust region**: the SOC `‖dqa‖ ≤ step_size` now bounds a
   6+29 tangent vector mixing linear/angular base DOF; confirm the scaling is
   sensible (the base angular and linear parts have different units).
4. **pinocchio install** in the `holonew` env — first task verifies
   `import pinocchio`. (Approved dependency.)

## Effect on later bricks

- Brick 2's SE(3) tooling builds on this pinocchio model (raw `log6/Jlog6` still
  fine, but the base twist now uses the same tangent convention).
- Brick 4's `A_G`/CoM no longer needs a separate alignment step — the seam is
  already established and validated here.
