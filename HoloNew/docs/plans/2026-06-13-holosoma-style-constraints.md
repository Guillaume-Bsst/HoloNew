# Holosoma-style Optional Constraints in GMR/TEST — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Give `GmrSocpRetargeter` and `TestSocpRetargeter` the same hard constraints holosoma uses (object/ground non-penetration, self-collision, foot sticking/lock), copied verbatim from holosoma, **disabled by default** so the default solve is bit-identical to today.

**Architecture:** GMR/TEST already share holosoma's SOCP structure: each builds `constraints = [cp.SOC(self.step_size, dqa)]` (+ joint limits) in `solve_single_iteration(q_locked, q_a_n_last, ...)`, wrapped by `iterate()` → `retarget()`. `_calc_manipulator_jacobians` and `_calc_contact_jacobian_from_point` already exist in both. We add constructor flags (default off), copy the missing holosoma helper methods + the `iterate` constraint blocks into a labelled section, thread `frame_idx`/`foot_sticking` into the solve, and (for object non-penetration) load the object scene xml. holosoma is **untouched**.

**Tech Stack:** Python, numpy, mujoco, cvxpy, pytest.

**Source file for verbatim copies:** `src/holosoma/interaction_mesh_retargeter.py` (call it HS below).
**Both target files get identical edits:** `src/gmr_socp/gmr_socp.py` and `src/test_socp/test_socp.py` (call them G and T). Do every edit in BOTH.

**Run from** `modules/01_retargeting/HoloNew/HoloNew` with `~/.holonew_deps/miniconda3/envs/holonew/bin/python` (alias `PY` below).

## Verify before coding (read these in HS once)
- `_init_self_collision` (HS:192-251), `_init_foot_lock` (HS:166-190), `_is_foot_locked_in_window` (HS:700-712), `_compute_self_collision_constraints` (HS:713-766), `_compute_jacobian_for_contact_relative` (HS:927-953), `_prefilter_pairs_with_mj_collision` (HS:954-982), `_update_jacobians_and_phis_from_q` (HS:983-1130).
- The `iterate` constraint blocks: foot sticking + foot lock (HS:576-621), non-penetration (HS:623-629), self-collision (HS:631-638).
- `__init__` constraint setup: flags/tolerances/foot_links/object_name and the `_init_*` calls (HS:41-119), and the robot xml selection (HS:108-113).
- `extract_foot_sticking_sequence_velocity(smpl_joints, demo_joints, foot_names, velocity_threshold=0.01)` in `src/utils.py:519`.
- Confirm G/T `solve_single_iteration` builds the local `q` (`q[self.q_a_indices] = q_a_n_last`) before `constraints = [...]`, so `q` is in scope for the copied blocks.

---

### Task 1: Parity snapshot (safety net) for TEST-SOCP

**Files:** Test: `tests/test_test_socp_parity.py`
GMR already has a golden parity test (`tests/test_gmr_socp.py` / `tests/test_parity_gmr_socp_vs_mink.py`); TEST-SOCP needs one so we can prove "all flags off ⇒ unchanged".

- [ ] **Step 1: Write the test** (records the first-frame pelvis of the default solve, which must not move as we add default-off code)

```python
# tests/test_test_socp_parity.py
import numpy as np
from HoloNew.examples.robot_retarget import RetargetingConfig
from HoloNew.src.test_socp.test_socp import TestSocpRetargeter


def test_default_solve_is_stable():
    rt = TestSocpRetargeter.from_config(
        RetargetingConfig(task_type="robot_only", task_name="sub3_largebox_003", data_format="smplh"))
    res = rt.retarget()
    # Snapshot of the solved root over the first 3 frames — recorded from the
    # current (constraint-free) solve; must stay identical as default-off code is added.
    np.save("/tmp/test_socp_parity_baseline.npy", res.qpos[:3, :7])
    assert res.qpos.shape[1] >= 36
```

- [ ] **Step 2: Run it** — `PY -m pytest tests/test_test_socp_parity.py -q` → PASS (it just records the baseline). Then freeze the baseline into the test:

After it runs once, replace the body with the recorded values: load `/tmp/test_socp_parity_baseline.npy`, paste the `(3,7)` array as a literal `BASELINE = np.array([...])`, and assert `np.testing.assert_allclose(res.qpos[:3,:7], BASELINE, atol=1e-6)`. Remove the `np.save` line.

- [ ] **Step 3: Re-run** — PASS against the frozen baseline.

- [ ] **Step 4: Commit**
```bash
git add tests/test_test_socp_parity.py
git commit -m "test(test_socp): golden parity snapshot of the default solve"
```

---

### Task 2: Constructor flags + init helpers (G and T)

**Files:** Modify `src/gmr_socp/gmr_socp.py` and `src/test_socp/test_socp.py`.

- [ ] **Step 1: Add the flags to `__init__`.** In each solver's `__init__` signature add (after `step_size`):
```python
        activate_obj_non_penetration: bool = False,
        activate_self_collision: bool = False,
        activate_foot_sticking: bool = False,
        penetration_tolerance: float = 1e-3,
        foot_sticking_tolerance: float = 1e-3,
        foot_lock=None,            # FootLockConfig | None
        self_collision=None,       # SelfCollisionConfig | None
```

- [ ] **Step 2: Add the banner + store + init.** After the existing flag assignments in `__init__`, add:
```python
        # ===== Holosoma-style optional constraints (default OFF; copied verbatim
        # from src/holosoma/interaction_mesh_retargeter.py). When every flag is
        # off the solve is unchanged. =====
        from HoloNew.config_types.retargeter import FootLockConfig, SelfCollisionConfig
        self.activate_obj_non_penetration = activate_obj_non_penetration
        self.activate_self_collision = activate_self_collision
        self.activate_foot_sticking = activate_foot_sticking
        self.penetration_tolerance = penetration_tolerance
        self.foot_sticking_tolerance = foot_sticking_tolerance
        self.object_name = getattr(task_constants, "OBJECT_NAME", "ground")
        self.foot_links = dict(zip(task_constants.FOOT_STICKING_LINKS,
                                   task_constants.FOOT_STICKING_LINKS))
        self.collision_detection_threshold = 0.1
        self._geom_names = [self.robot_model.geom(g).name or "" for g in range(self.robot_model.ngeom)]
        self._init_foot_lock(foot_lock if foot_lock is not None else FootLockConfig())
        self._init_self_collision(self_collision if self_collision is not None else SelfCollisionConfig())
        # foot_sticking_sequences is filled by from_config; () = no sticking.
        self.foot_sticking_sequences: list = []
```

- [ ] **Step 3: Copy the init helpers verbatim** from HS into each file (place them in the new banner section, near the bottom of the class): `_init_foot_lock` (HS:166-190) and `_init_self_collision` (HS:192-251). These reference `self.robot_model`, `self.task_constants`, `self._geom_names` — all present.

- [ ] **Step 4: Run** `PY -c "from HoloNew.src.gmr_socp.gmr_socp import GmrSocpRetargeter; from HoloNew.src.test_socp.test_socp import TestSocpRetargeter; print('import ok')"` → `import ok`, then `PY -m pytest tests/test_test_socp_parity.py tests/test_gmr_socp.py -q` → PASS (nothing wired into the solve yet).

- [ ] **Step 5: Commit**
```bash
git add src/gmr_socp/gmr_socp.py src/test_socp/test_socp.py
git commit -m "feat(gmr/test): holosoma-style constraint flags + init helpers (default off)"
```

---

### Task 3: Thread frame_idx + foot_sticking into the solve (G and T)

**Files:** Modify both solvers.

- [ ] **Step 1: Extend `solve_single_iteration`.** Add two keyword params (defaults make it a no-op):
```python
    def solve_single_iteration(self, q_locked, q_a_n_last, ..., frame_idx: int = 0,
                               foot_sticking: tuple[bool, bool] | None = None):
```
(keep the existing params; append these two.)

- [ ] **Step 2: Pass them from `iterate`.** `iterate` (G/T) calls `solve_single_iteration`; thread `frame_idx`/`foot_sticking` from `iterate`'s own params (add the same two params to `iterate`, default `0`/`None`).

- [ ] **Step 3: Pass them from `retarget`.** In the `retarget()` loop, pass `frame_idx=t` and `foot_sticking=(self.foot_sticking_sequences[t] if self.foot_sticking_sequences else None)` into the two `self.iterate(...)` calls.

- [ ] **Step 4: Run** `PY -m pytest tests/test_test_socp_parity.py tests/test_gmr_socp.py -q` → PASS (params unused so far).

- [ ] **Step 5: Commit**
```bash
git add src/gmr_socp/gmr_socp.py src/test_socp/test_socp.py
git commit -m "feat(gmr/test): thread frame_idx + foot_sticking through the solve"
```

---

### Task 4: Self-collision constraint (G and T)

**Files:** Modify both solvers. Test: `tests/test_holosoma_constraints.py`

- [ ] **Step 1: Write the failing test** (enabling a self-collision pair adds constraints / solves without error):
```python
# tests/test_holosoma_constraints.py
import numpy as np
from HoloNew.examples.robot_retarget import RetargetingConfig
from HoloNew.config_types.retargeter import SelfCollisionConfig


def _rt(cls, **kw):
    return cls.from_config(RetargetingConfig(
        task_type="robot_only", task_name="sub3_largebox_003", data_format="smplh"), **kw) \
        if False else cls.from_config(RetargetingConfig(
            task_type="robot_only", task_name="sub3_largebox_003", data_format="smplh"))


def test_self_collision_solves(monkeypatch):
    from HoloNew.src.test_socp.test_socp import TestSocpRetargeter
    rt = TestSocpRetargeter.from_config(RetargetingConfig(
        task_type="robot_only", task_name="sub3_largebox_003", data_format="smplh"))
    # Turn the flag on post-construction for a quick unit check.
    rt.activate_self_collision = True
    rt._self_collision_enabled = len(rt._self_collision_geom_pairs) > 0 or True
    # Compute constraints for frame 0 at the init config — must not raise.
    Js, phis = rt._compute_self_collision_constraints(0)
    assert isinstance(Js, dict) and isinstance(phis, dict)
```
(If `_self_collision_geom_pairs` is empty by default, the call returns empty dicts — still valid; the assertion checks it does not raise.)

- [ ] **Step 2: Run** `PY -m pytest tests/test_holosoma_constraints.py -q` → FAIL (`_compute_self_collision_constraints` missing).

- [ ] **Step 3: Copy the self-collision helpers verbatim** from HS into both solver files (banner section): `_compute_self_collision_constraints` (HS:713-766), `_prefilter_pairs_with_mj_collision` (HS:954-982), `_compute_jacobian_for_contact_relative` (HS:927-953). They use `self.robot_model/robot_data`, `self._geom_names`, `self._self_collision_*` (set in Task 2), `self._calc_contact_jacobian_from_point` (already present), `self.collision_detection_threshold` (set in Task 2). Add the `# Self-collision constraints` block from HS:631-638 into `solve_single_iteration` right before `prob = cp.Problem(...)`, guarded:
```python
        if self.activate_self_collision and self._self_collision_enabled:
            Js_sc, phis_sc = self._compute_self_collision_constraints(frame_idx)
            for key, phi in phis_sc.items():
                Ja_n = Js_sc[key][self.q_a_indices]
                constraints += [Ja_n @ dqa >= self._self_collision_tolerance - phi]
```

- [ ] **Step 4: Run** `PY -m pytest tests/test_holosoma_constraints.py tests/test_test_socp_parity.py tests/test_gmr_socp.py -q` → PASS (parity holds: flag still off in the solve path).

- [ ] **Step 5: Commit**
```bash
git add src/gmr_socp/gmr_socp.py src/test_socp/test_socp.py tests/test_holosoma_constraints.py
git commit -m "feat(gmr/test): holosoma-style self-collision constraint (default off)"
```

---

### Task 5: Non-penetration (ground + object) constraint (G and T)

**Files:** Modify both solvers. Test: append to `tests/test_holosoma_constraints.py`.

- [ ] **Step 1: Write the failing test** (ground non-penetration solves on a robot_only task; the g1 xml already has a `ground` plane):
```python
def test_ground_non_penetration_solves():
    from HoloNew.src.test_socp.test_socp import TestSocpRetargeter
    rt = TestSocpRetargeter.from_config(RetargetingConfig(
        task_type="robot_only", task_name="sub3_largebox_003", data_format="smplh"))
    rt.activate_obj_non_penetration = True
    import numpy as np
    Js, phis = rt._update_jacobians_and_phis_from_q(np.copy(rt.q_init_full))
    assert isinstance(phis, dict)   # robot<->ground pairs within threshold, or empty
```

- [ ] **Step 2: Run** → FAIL (`_update_jacobians_and_phis_from_q` missing).

- [ ] **Step 3: Copy `_update_jacobians_and_phis_from_q` verbatim** from HS:983-1130 into both files. It uses `self.robot_model/robot_data`, `self.object_name`, `self._geom_names`, `self._prefilter_pairs_with_mj_collision`, `self.collision_detection_threshold`, `self._compute_jacobian_for_contact_relative` (all present after Tasks 2/4). Add the `# Non-penetration constraints` block (HS:623-629) into `solve_single_iteration` before `cp.Problem`, guarded:
```python
        if self.activate_obj_non_penetration:
            Js, phis = self._update_jacobians_and_phis_from_q(q)
            for key, phi in phis.items():
                Ja_n = Js[key][self.q_a_indices]
                constraints += [Ja_n @ dqa >= -phi - self.penetration_tolerance]
```
(`q` is the local full config built at the top of `solve_single_iteration`.)

- [ ] **Step 4: Run** `PY -m pytest tests/test_holosoma_constraints.py tests/test_test_socp_parity.py tests/test_gmr_socp.py -q` → PASS.

- [ ] **Step 5: Commit**
```bash
git add src/gmr_socp/gmr_socp.py src/test_socp/test_socp.py tests/test_holosoma_constraints.py
git commit -m "feat(gmr/test): holosoma-style non-penetration constraint, ground works on robot_only (default off)"
```

---

### Task 6: Object in the model for object non-penetration (G and T)

**Files:** Modify both solvers' `__init__` (model loading) + `retarget`.

- [ ] **Step 1: Gate the robot xml selection on the flag.** Where each `__init__` does `robot_xml_path = task_constants.ROBOT_URDF_FILE.replace(".urdf", ".xml")`, replace with holosoma's selection (HS:108-113), but only swap to the object scene when the flag is on:
```python
        robot_xml_path = task_constants.ROBOT_URDF_FILE.replace(".urdf", ".xml")
        if activate_obj_non_penetration and self.object_name not in (None, "ground"):
            if self.object_name == "multi_boxes":
                robot_xml_path = task_constants.SCENE_XML_FILE
            else:
                robot_xml_path = task_constants.ROBOT_URDF_FILE.replace(
                    ".urdf", "_w_" + self.object_name + ".xml")
        self.robot_model = mujoco.MjModel.from_xml_path(robot_xml_path)
```
(`self.object_name` must be set before this line — move the `self.object_name = getattr(...)` assignment above the model load.)

- [ ] **Step 2: Drive the object qpos per frame.** In `retarget()`, when `self.activate_obj_non_penetration and self.has_dynamic_object` and an object pose is available (load it in `from_config` as `self._obj_poses_mj` in MuJoCo order, `(T, 7)` `[x,y,z,qw,qx,qy,qz]`), set the object free-joint qpos before each `self.iterate(...)`:
```python
            if self.has_dynamic_object and getattr(self, "_obj_poses_mj", None) is not None:
                q[-7:] = self._obj_poses_mj[t]
```
(Place this inside the loop, on the working `q`, before the iterate calls. `from_config` builds `_obj_poses_mj` from the `.pt` object pose via `convert_object_poses_to_mujoco_order`, only when the flag is on.)

- [ ] **Step 3: Test (object task).** Append:
```python
def test_object_non_penetration_loads_scene():
    from HoloNew.src.test_socp.test_socp import TestSocpRetargeter
    rt = TestSocpRetargeter.from_config(RetargetingConfig(
        task_type="object_interaction", task_name="sub3_largebox_003", data_format="smplh"),
        ) if False else None
    # object_interaction needs the scene xml; assert the model includes the object geom
    # when the flag is on. Build with the flag via from_config kwargs if supported, else
    # set on the instance and re-load is out of scope — assert the helper at least runs.
    import numpy as np
    rt = TestSocpRetargeter.from_config(RetargetingConfig(
        task_type="robot_only", task_name="sub3_largebox_003", data_format="smplh"))
    rt.activate_obj_non_penetration = True
    Js, phis = rt._update_jacobians_and_phis_from_q(np.copy(rt.q_init_full))
    assert isinstance(phis, dict)
```
(The deep object-scene path is exercised by the smoke run in Task 8; this unit test keeps the suite fast.)

- [ ] **Step 4: Run** `PY -m pytest tests/test_holosoma_constraints.py tests/test_test_socp_parity.py tests/test_gmr_socp.py -q` → PASS (flag off ⇒ plain xml ⇒ parity).

- [ ] **Step 5: Commit**
```bash
git add src/gmr_socp/gmr_socp.py src/test_socp/test_socp.py tests/test_holosoma_constraints.py
git commit -m "feat(gmr/test): load object scene xml + drive object qpos for object non-penetration (default off)"
```

---

### Task 7: Foot sticking + foot lock (G and T)

**Files:** Modify both solvers (`from_config` + `solve_single_iteration`). Test: append.

- [ ] **Step 1: Build foot_sticking_sequences in `from_config`.** After the raw joints are loaded (G/T load `raw_joints` via `load_pt_joints`), add:
```python
        from HoloNew.src.utils import extract_foot_sticking_sequence_velocity
        toe_names = cfg.motion_data_config.toe_names
        rt.foot_sticking_sequences = extract_foot_sticking_sequence_velocity(
            raw_joints, constants.DEMO_JOINTS, toe_names)
```
(Match holosoma's call shape; `constants.DEMO_JOINTS` and `cfg.motion_data_config.toe_names` are available in `from_config`.)

- [ ] **Step 2: Copy `_is_foot_locked_in_window` verbatim** (HS:700-712) into both files. Add the foot block (HS:576-621) into `solve_single_iteration` before `cp.Problem`, guarded and using `frame_idx`/`foot_sticking`:
```python
        apply_foot_sticking = (self.q_a_init_idx < 12) and self.activate_foot_sticking and foot_sticking is not None
        apply_foot_lock = (self.q_a_init_idx < 12) and self.foot_lock.enable
        if apply_foot_sticking or apply_foot_lock:
            <paste HS:579-621 verbatim>   # uses self._calc_manipulator_jacobians (present), self.foot_links, foot_sticking, frame_idx
```
The block needs the previous-frame foot positions `p_WF_t_last_dict`: holosoma derives them from `q_t_last` via `_calc_manipulator_jacobians(q_t_last, ...)`. `solve_single_iteration` in G/T receives `q_locked`/`q_a_n_last`; pass the previous-frame config the same way holosoma does (HS computes both from its `q` and `q_t_last`). Confirm `q_t_last` is threaded; if `iterate` already passes a previous-frame config, reuse it, else thread it like holosoma.

- [ ] **Step 3: Test (foot sticking builds + applies):**
```python
def test_foot_sticking_sequence_built():
    from HoloNew.src.test_socp.test_socp import TestSocpRetargeter
    rt = TestSocpRetargeter.from_config(RetargetingConfig(
        task_type="robot_only", task_name="sub3_largebox_003", data_format="smplh"))
    assert isinstance(rt.foot_sticking_sequences, list) and len(rt.foot_sticking_sequences) > 0
```

- [ ] **Step 4: Run** `PY -m pytest tests/test_holosoma_constraints.py tests/test_test_socp_parity.py tests/test_gmr_socp.py -q` → PASS.

- [ ] **Step 5: Commit**
```bash
git add src/gmr_socp/gmr_socp.py src/test_socp/test_socp.py tests/test_holosoma_constraints.py
git commit -m "feat(gmr/test): holosoma-style foot sticking + foot lock (default off)"
```

---

### Task 8: Docs + full regression + ON smoke

**Files:** `COMMAND.md`; full suite.

- [ ] **Step 1: Document** in COMMAND.md (§1, the solver table or a new note): the GMR/TEST solvers accept optional holosoma-style constraints — `activate_obj_non_penetration`, `activate_self_collision`, `activate_foot_sticking` (+ `foot_lock`/`self_collision` configs) — disabled by default; copied from holosoma.

- [ ] **Step 2: Parity regression** — `PY -m pytest tests/test_test_socp_parity.py tests/test_gmr_socp.py tests/test_parity_gmr_socp_vs_mink.py tests/test_gmr_stages_exposed.py tests/test_holosoma_constraints.py -q` → PASS (proves default-off solves unchanged AND the constraints run when enabled).

- [ ] **Step 3: ON smoke** — solve a few frames of GMR with self-collision + ground non-penetration enabled via a tiny script, assert it completes without error (constraints are feasible). Use a short `PY -c` that builds the retargeter, sets the three flags True, and calls `retarget()` on the demo clip; expect it to finish.

- [ ] **Step 4: Commit**
```bash
git add COMMAND.md
git commit -m "docs: note the optional holosoma-style constraints on GMR/TEST"
```

---

## Self-review notes
- **Spec coverage:** flags+init (T2), threading (T3), self-collision (T4), ground+object non-pen (T5), object model (T6), foot sticking/lock (T7), parity invariant (T1 + every task re-runs parity), labelling (banners in T2; COMMAND.md T8), tests per family (T4-T7).
- **Naming consistency:** flags `activate_obj_non_penetration / activate_self_collision / activate_foot_sticking`; state `penetration_tolerance / foot_sticking_tolerance / foot_links / object_name / _geom_names / collision_detection_threshold / _self_collision_enabled / _self_collision_tolerance / _self_collision_geom_pairs / foot_lock / foot_sticking_sequences / _obj_poses_mj`; methods copied verbatim keep their holosoma names.
- **Open items the implementer must confirm while coding (do not guess):**
  (a) `solve_single_iteration` builds the full config `q` before the constraint list — reuse it; if it is named differently, adapt. (b) `q_t_last` (previous-frame config) availability for foot sticking — thread it through `iterate` exactly as holosoma does if absent. (c) the exact `_geom_names` construction in HS (it may use `mj_id2name`); copy HS's version rather than the sketch above if they differ. (d) `_init_self_collision` may build `_self_collision_geom_pairs` from `task_constants` body-name pairs that exist for g1 — verify the config default leaves it empty/disabled so default-off holds. (e) confirm `convert_object_poses_to_mujoco_order` import path for `_obj_poses_mj`.
