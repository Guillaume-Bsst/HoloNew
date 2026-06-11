# GMR-SOCP retargeter (increment 3) — design

**Date:** 2026-06-11
**Status:** Design — pending implementation plan
**Scope:** `modules/01_retargeting/HoloNew` (the HoloNew package)
**Builds on:** `2026-06-11-holosoma-modular-retargeting-viz-design.md` (stages registry, Viewer, multi-trajectory display).

## Goal

Add a **second retargeter** alongside the native interaction-mesh SOCP retargeter:
a GMR-style retargeter whose resolution is a copy of holosoma's SOCP solve, with
the objective and constraints changed so the result is **similar to GMR/mink**.
It produces extra trajectories shown next to the native one in the same viser.

Two versions, **`v1` and `v2`, identical at creation**. `v2` is the designated
extension point for later blending holosoma's own logic; it diverges from `v1`
only in a future increment.

## Background

- **Native** (`src/interaction_mesh_retargeter.py`, `InteractionMeshRetargeter`):
  per-frame SQP/SOCP solve (cvxpy + CLARABEL) on the actuated-joint displacement
  `dqa`. Objective = interaction-mesh Laplacian tracking + nominal + regularization
  + smoothness. Constraints = SDF non-penetration, self-collision, joint limits,
  foot-sticking/lock, trust-region (`cp.SOC`). It already has `activate_*` flags
  for the optional constraints. It loads only human joint **positions** `(T, J, 3)`.
- **GMR/mink** (test_pipe `solver/gmr`): per-body `FrameTask` tracking solved by a
  two-pass velocity IK. Two weight tables (`IK_MATCH_TABLE1/2`): pass 1 tracks
  orientation-led with planted pelvis/feet; pass 2 adds limb position and tightens
  foot orientation. mink's only constraint is configuration (joint) limits, plus
  Levenberg–Marquardt damping. No collision / non-penetration constraints.

## Decisions (from brainstorming)

1. **No step1/step2 stages** — expose only the final qpos per version.
2. **Three standalone retargeter modules**: native (exists) + `gmr_socp_v1` +
   `gmr_socp_v2`. Each GMR module **starts as a copy-paste of the native solve
   framework** (model load, Jacobians, SDF helpers, cvxpy scaffold) so it stays
   close to native and divergence is limited. `v2` is a literal copy-paste of `v1`,
   done last.
3. **Result similar to mink ⇒ same cost function and constraints as mink:**
   - **Cost** = GMR body-frame tracking (position + orientation), weighted per
     `IK_MATCH_TABLE1/2`. Not the interaction-mesh Laplacian cost.
   - **Constraints** = mink-equivalent: **joint limits + trust-region only**. The
     native non-penetration, self-collision and foot-sticking constraints are
     **disabled** (mink had none) — straightforward via the existing `activate_*`
     flags / by not emitting those constraint blocks in the copied solve.
   - **Two passes** per frame: solve with `table1` weights to convergence, then
     `table2` weights, warm-started across frames (matches GMR).
4. **Full position + orientation tracking** ⇒ load per-body **quaternions** from the
   OMOMO `.pt` (as test_pipe's `load_pt` does); native does not load them today.

## Architecture

New, isolated package `src/gmr_socp/` (additive; native retargeter untouched):

- `tables.py` — GMR IK match tables ported **verbatim** from test_pipe
  (`solver/gmr/tables.py`): `IK_MATCH_TABLE1`, `IK_MATCH_TABLE2`,
  `HUMAN_BODY_TO_IDX`, scale/height constants. Credit: GMR (YanjieZe/GMR). Shared
  data — not duplicated per version.
- `targets.py` — builds per-frame SE3 body targets for the mapped robot frames:
  takes human joint positions + per-body quaternions + a table, applies
  `pos_offset` (in the re-oriented body frame) and composes `rot_offset`. Returns,
  per mapped robot frame, `(p_target, R_target, pos_weight, rot_weight)`. Shared.
- `quat_loader` (in `targets.py` or a small helper) — loads per-body quaternions
  from the OMOMO `.pt` for a sequence.
- `gmr_socp_v1.py` — `GmrSocpRetargeterV1`: a standalone copy of the native solve
  framework, with the **GMR tracking cost** and **mink-equivalent constraints**,
  and the **two-pass** per-frame loop. Owns its full resolution (freely modifiable).
- `gmr_socp_v2.py` — `GmrSocpRetargeterV2`: a literal copy-paste of `v1`, identical
  at creation; the future evolution point.

### The copied resolution (per GMR module)

Same machinery as native (so divergence is limited):
- Variable `dqa` (actuated-joint displacement), same q layout
  (`[0:3]` pos, `[3:7]` wxyz quat, `[7:7+dof]` joints).
- **Objective (GMR tracking)**, summed over mapped robot frames `f` with weights
  `(w_p, w_r)` from the active table:
  - position: `w_p · ‖ J_pos(f) · dqa − (p_target(f) − p_current(f)) ‖²`
  - orientation: `w_r · ‖ J_rot(f) · dqa − e_rot(f) ‖²`, where `e_rot(f)` is the
    linearized orientation error (log map of `R_current(f)ᵀ R_target(f)`) in the
    world/body frame consistent with `J_rot`.
  `J_pos`/`J_rot` come from the copied manipulator-Jacobian helper (same as native).
- **Constraints**: joint limits (`activate_joint_limits`) + trust-region
  (`cp.SOC(step_size, dqa)`). Non-penetration / self-collision / foot-sticking
  **omitted** to match mink.
- **Solver**: cvxpy + CLARABEL, identical to native.
- **Two passes**: per frame, run SQP iterations with `table1` weights to
  convergence, then with `table2` weights to convergence; warm-start the next
  frame from the previous solution.

### Integration with the viewer

- Add two `StageSpec(produces_qpos=True)` entries to `src/stages.py`:
  `("GMR-SOCP v1", "gmr_socp_v1", True)` and `("GMR-SOCP v2", "gmr_socp_v2", True)`.
- `examples/view_stages.py` runs the native retarget plus both GMR-SOCP versions
  and binds all qpos trajectories, so the viewer shows the native SOCP robot and
  the two GMR-SOCP robots together (selectable / comparable), each under its own
  `/world/robot_<key>`.
- The multi-robot `Viewer` already supports this via `stage_keys` (increment 2).

## Data flow

1. Load sequence: human joint positions (as today) **and** per-body quaternions
   (new, from the `.pt`).
2. `targets.py` builds per-frame SE3 targets for the mapped robot frames from the
   active table (offsets applied).
3. Each GMR module solves per frame: pass 1 (table1) → pass 2 (table2), producing
   one qpos `(T, 7+dof)` trajectory.
4. The native retarget runs as before. All trajectories feed the viewer.

## Testing

- The **native golden test stays unchanged** (proves the native path is untouched —
  the GMR modules are fully separate, so this should hold trivially).
- **GMR-SOCP integration test**: runs `gmr_socp_v1` on the demo OMOMO sequence;
  asserts it returns a qpos of shape `(T, 7+dof)` without error, and basic sanity
  (pelvis position tracks the human pelvis within a tolerance; feet roughly planted).
  Full numeric parity with mink is **not** a goal (conic SOCP ≠ velocity IK); the
  cost/constraints are aligned for a *similar* result, not bit-identical.
- **Registry/viewer test**: `STAGE_SPECS` includes the two GMR stages; the Viewer
  builds robots for `gmr_socp_v1`/`gmr_socp_v2`.
- `v2` being a copy of `v1` is covered by the same integration test parameterized
  over both classes.

## Out of scope (later increments)

- `v2`'s actual divergence (re-introducing interaction-mesh / holosoma logic).
- Datasets other than OMOMO `.pt` for orientation targets.
- Contact/OT overlays (separate increment).

## Open items for the implementation plan

- Confirm `HUMAN_BODY_TO_IDX` (test_pipe's 52-joint indices) matches holosoma's
  `DEMO_JOINTS` ordering for OMOMO; add a small assertion/parity check.
- Exact reuse vs copy of the native Jacobian / model-loading helpers when copying
  the solve into `gmr_socp_v1` (the design says copy the framework; the plan fixes
  which concrete methods are copied).
- The per-frame convergence criterion and iteration counts for the two passes
  (mirror native's `iterate` n_iter, or GMR's convergence_eps).
- Sign/frame conventions for `J_rot` and `e_rot` (world vs body) — verify against
  the copied Jacobian helper.
