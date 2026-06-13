# Brick 5 — Movable entities as variables W^o — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: superpowers:subagent-driven-development. Checkbox steps.

**Goal:** Make the manipulated object's pose a solved decision variable (not driven), with `W^o` motion regularization and bilateral robot↔object coupling in the D/X interaction terms — behind a flag, validated on an object task, enabled by default **only if stable**.

**Architecture:** Gated `activate_movable` (default off ⇒ object driven, as today ⇒ parity). When on, `solve_single_iteration` extends the decision variable to `[dqa (nv_a) ; dξ_obj (6)]` — a robot tangent step plus an object SE(3) tangent step — integrated separately (`pin.integrate` for the robot, an SE(3) exp for the object). Three additions: (1) `W^o = λ_o‖v̇_obj − v̇_ref‖² + λ_ω‖ω_obj − ω_ref‖²` regularizing the object's linear acceleration + spin toward the reference object motion; (2) the object-channel D/X residuals become **bilateral** — the robot control point's object-local displacement now also depends on `dξ_obj` (the object's rigid motion at the contact point); (3) object↔floor non-penetration gains the object DOF. The object is only present on object tasks, so robot_only is untouched.

**Scope note:** This is the most structural brick. It is foundational: object↔object transitive transport (robot→tray→cup) and object hard non-penetration are deferred follow-ons; this plan delivers single-object W^o + bilateral D/X.

**Tech Stack:** Python, pinocchio (SE(3) exp/log), numpy, cvxpy, pytest. Run from `modules/01_retargeting/HoloNew/HoloNew`, PY = `~/.holonew_deps/miniconda3/envs/holonew/bin/python`.

**Design:** `docs/specs/2026-06-13-brick5-movable-entities-design.md`.

## Confirmed interfaces
- The object pose per frame: `rt._obj_poses_raw[t]` = `[qw,qx,qy,qz,x,y,z]` (loaded in from_config for object tasks). `retarget` passes `obj_pose` into `iterate`/`solve_single_iteration`. The interaction terms (`build_dx_terms` in `src/test_socp/interaction.py`) take `obj_pose` and query the object SDF in object-local frame via `_robj_from_pose` / `_world_to_object_local`.
- `solve_single_iteration` decision var is `dqa = cp.Variable(self.nv_a)`. `pin` SE(3): `pin.exp6(xi) -> SE3` (xi a 6-vector [v; ω]), `pin.SE3`, `pin.log6`. An object pose `[qw,qx,qy,qz,x,y,z]` ↔ `pin.SE3(R, t)`.
- `rt.smplx_ground_probe.margin`, `rt.correspondence`, `rt.pin.point_jacobians`.

---

### Task 1: Object SE(3) tangent state + W^o term (object motion only, robot fixed)

**Files:** Create `src/test_socp/movable.py`; Test: `tests/test_movable.py`

`W^o` penalizes the object's linear acceleration and angular velocity toward the
reference. With the object pose a variable `T_obj = T_obj^0 ⊕ dξ` (`dξ` 6-vector),
and the previous two object poses known, the object velocity `V_obj = (1/Δt)
log(T_{obj,t-1}^{-1} T_obj)` and acceleration are linear in `dξ` (via the brick-2
`difference_and_jac`-style SE(3) Jacobian on a standalone SE3, using `pin.Jlog6`).

- [ ] **Step 1: Write the failing test** (the W^o objective at a random dξ matches an independent numpy ground truth) — model `T_obj^0`, `T_{obj,t-1}`, `T_{obj,t-2}` as random SE3; reference `v̇_ref`, `ω_ref`; assert `build_wo_term(...).value` equals the numpy `λ_o‖v̇_obj − v̇_ref‖² + λ_ω‖ω_obj − ω_ref‖²` at a numeric `dξ.value`. (Use `pin.exp6`/`pin.log6`/`pin.Jlog6` for the SE(3) ops; mirror the brick-2 temporal validation pattern.)

- [ ] **Step 2: Run** → FAIL.

- [ ] **Step 3: Implement** `src/test_socp/movable.py::build_wo_term(T_obj0, T_obj_tm1, T_obj_tm2, vdot_ref, omega_ref, dxi, lambda_o, lambda_omega, dt)` returning one cvxpy expression. SE(3) velocity `V_t = (1/dt) log6(T_{t-1}^{-1} (T0 ⊕ dξ))`; linearize via `Jlog6` at dξ=0. Acceleration `v̇ = (v_t − v_{t-1})/dt` (linear part), spin `ω_t` (angular part). Weighted squared residuals folded into a single `cp.sum_squares`.

- [ ] **Step 4: Run** → PASS. **Step 5: Commit** `feat(test_socp): object SE(3) W^o motion term (numpy-validated)`.

---

### Task 2: Extended decision variable + object integration in the solve

**Files:** Modify `src/test_socp/config.py`, `src/test_socp/test_socp.py`; Test: append.

- [ ] **Step 1: Config.** Add `activate_movable: bool = False`, `lambda_o: float = 0.0`, `lambda_omega: float = 0.0`. Constructor + from_config wiring.

- [ ] **Step 2: Decision variable.** In `solve_single_iteration`, when `self.activate_movable and obj_pose is not None`, additionally create `dxi = cp.Variable(6)` (object tangent). The trust region SOC currently bounds `dqa`; add a separate `cp.SOC(self.step_size, dxi)` (or a combined bound). After solving, integrate the object: `T_obj_new = pin.exp6(dxi.value) * pin.SE3(R_obj0, t_obj0)` (left-compose in the object frame — confirm convention vs the D/X frame), and store the solved object pose for the frame on a list `obj_solved`. Add the W^o term (Task 1) to `obj_terms` (needs the two previous solved object poses + the reference object accel/spin; thread them like the robot's CoM history). When off, the object stays driven (`_obj_poses_raw`), unchanged.

- [ ] **Step 3: Thread** the previous-two solved object poses + reference object motion through `retarget`→`iterate`→`solve_single_iteration` (mirror the CoM-history threading from brick 4). Reference `v̇_ref`/`ω_ref` from `_obj_poses_raw` (causal SE(3) differences).

- [ ] **Step 4: Test (append).** `activate_movable` default off ⇒ object_interaction solve unchanged from driven (snapshot a few frames). On ⇒ finite, and the solved object pose stays close to the reference (W^o tracks it). `PY -m pytest tests/test_movable.py -q` → PASS.

- [ ] **Step 5: Commit** `feat(test_socp): object pose as a solved variable with W^o (default off)`.

---

### Task 3: Bilateral robot↔object coupling in the object D/X terms

**Files:** Modify `src/test_socp/interaction.py` (`build_dx_terms`), `src/test_socp/test_socp.py`; Test: append.

When the object is a variable, the robot control point's **object-local**
displacement gains the object's rigid motion at that point: for point `p_i`, the
object-frame relative displacement is `Robj^T (J_i dqa − (v_obj + ω_obj × (p_i −
t_obj)))`, i.e. the D/X object-channel residual rows gain object-DOF columns
`−Robj^T [I, −[p_i − t_obj]_×]` acting on `dξ`. Implement an optional `dxi`
argument to `build_dx_terms` (default None ⇒ current behaviour); when given, the
object-channel `A`/`B` matrices are widened with the object-DOF block and the
returned expressions are affine in `[dqa; dxi]`.

- [ ] **Step 1: Write the failing test** (numpy equivalence of the bilateral object-channel residual at random `dqa`,`dxi`).
- [ ] **Step 2: Run** → FAIL.
- [ ] **Step 3: Implement** the object-DOF columns in `build_dx_terms`; the solver passes `dxi` when `activate_movable`.
- [ ] **Step 4: Run** → PASS (numpy equivalence + the object-clip solve finite).
- [ ] **Step 5: Commit** `feat(test_socp): bilateral robot<->object coupling in D/X (object DOF)`.

---

### Task 4: Validate + enable IF stable

**Files:** Test: `tests/test_movable_metric.py`; config defaults + (no robot_only re-baseline — object only).

- [ ] **Step 1: Metric.** On `object_interaction sub3_largebox_003`, `retarget(max_frames=30)` with `activate_movable=True` + tuned `lambda_o/lambda_omega`. Assert: finite; the solved object pose tracks the reference within a band (W^o + bilateral coupling); the robot↔object contact gap (from brick 1) is not worse; runtime reasonable. Record numbers.
- [ ] **Step 2: Enable IF clean.** If stable + the object stays sensible: set `activate_movable=True` + tuned lambdas as defaults. (robot_only has no object ⇒ its parity is untouched; the object-task behaviour changes — note it; there is no robot_only re-baseline.) If unstable, leave off + document, DONE_WITH_CONCERNS.
- [ ] **Step 3: Regression.** `PY -m pytest tests/test_movable.py tests/test_movable_metric.py tests/test_test_socp_parity.py tests/test_interaction_dxp.py tests/test_style.py -q` → PASS. Full object clip finite.
- [ ] **Step 4: Commit** `feat(test_socp): validate movable-entity W^o + bilateral coupling; <enable | keep off>`.

---

## Self-review notes
- **Spec coverage:** W^o term (T1), object variable + integration (T2), bilateral D/X (T3), validate + conditional enable (T4). Object↔object transitive transport + object hard non-penetration are deferred.
- **Naming:** `movable.py::build_wo_term`, `activate_movable`, `lambda_o`, `lambda_omega`, `dxi`.
- **Safety:** default off ⇒ object driven ⇒ object-task behaviour and robot_only parity unchanged; enable only if stable. robot_only has no object so its parity is structurally unaffected.
- **Open items:** (a) the SE(3) object integration convention (left vs right compose) must match the D/X object-local frame convention — the bilateral test gates it. (b) `pin.exp6`/`log6`/`Jlog6` exact API in 4.0.0. (c) W^o references from `_obj_poses_raw` causal differences. (d) the object DOF couples into the trust region — confirm the SOC bound is sensible.
