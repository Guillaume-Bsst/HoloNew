# Holosoma-style optional constraints in GMR-SOCP and TEST-SOCP

## Goal

Let the GMR-SOCP and TEST-SOCP solvers optionally enforce the same hard
constraints holosoma uses — object/ground non-penetration, self-collision, and
foot sticking / foot lock — **disabled by default**, so the solvers behave exactly
as today unless a flag is turned on. The constraint code is **copied verbatim from
holosoma** (the three solvers already share the same SOCP structure and variable
names, so the blocks transplant directly) and lives in a clearly labelled
"Holosoma-style optional constraints" section in each solver, so it is not mixed
with the GMR/TEST tracking objective.

This is a duplication-on-purpose: holosoma stays **untouched** (its golden/parity
tests must keep passing), and GMR/TEST each carry their own copy.

## Source (copy verbatim from `src/holosoma/interaction_mesh_retargeter.py`)

The following are transplanted unchanged (same names, same bodies):
- `_update_jacobians_and_phis_from_q(q)` — object+ground non-penetration distances
  via `mj_collision` prefilter + `mj_geomDistance`; `masks_ok` keeps robot↔object
  and robot↔ground pairs only.
- `_prefilter_pairs_with_mj_collision(threshold)`.
- `_compute_self_collision_constraints(frame_idx)`, `_init_self_collision(cfg)`.
- `_compute_jacobian_for_contact_relative(...)` (needed by self-collision; the
  `_calc_contact_jacobian_from_point` it calls already exists in both solvers).
- `_init_foot_lock(cfg)`, `_is_foot_locked_in_window(link, frame_idx)`.
- The constraint blocks from `iterate()`: foot sticking (XY window), foot lock (Z
  window), non-penetration, self-collision — appended to the solver's existing
  `constraints` list.

## Per-solver changes (identical edits to `gmr_socp.py` and `test_socp.py`)

### Constructor flags (all default OFF) + state
```
activate_obj_non_penetration: bool = False   # holosoma non-pen block (ground + object)
activate_self_collision:      bool = False
activate_foot_sticking:       bool = False
penetration_tolerance:  float = 1e-3
foot_sticking_tolerance: float = 1e-3
foot_lock:       FootLockConfig | None = None        # disabled unless given
self_collision:  SelfCollisionConfig | None = None   # disabled unless given
```
Store them and run `_init_foot_lock` / `_init_self_collision` (copied from holosoma)
in `__init__`. `FootLockConfig` / `SelfCollisionConfig` come from
`config_types/retargeter.py` (already used by holosoma).

### Model loading (object non-penetration prerequisite)
GMR/TEST today load the plain robot xml. To make the object geom available for
non-penetration, replicate holosoma's xml selection (lines 108-113), **gated by the
flag** so the default path is unchanged:
- `activate_obj_non_penetration` False → plain robot xml (`g1_29dof.xml`), as today.
- True and the task has an object → load `<robot>_w_<object>.xml` (object_interaction)
  or `SCENE_XML_FILE` (climbing), exactly like holosoma. This adds the object's free
  joint to qpos (`has_dynamic_object` becomes True); the object qpos is set from the
  `.pt` object pose each frame before `_update_jacobians_and_phis_from_q` (the object
  is positioned, not solved). Ground non-penetration works either way (the g1 xml has
  a `ground` plane geom).

### Foot-sticking data
`foot_sticking` is a per-frame `(left, right)` boolean sequence. Compute it in
`from_config` from the raw human joints with
`extract_foot_sticking_sequence_velocity(human_joints, demo_joints, toe_names)`
(from `src/utils.py`, as holosoma's pipeline does), store on the retargeter, and pass
`foot_sticking[t]` into `iterate()` per frame.

### `iterate()` integration
Append the copied constraint blocks to the existing `constraints` list, each guarded
by its flag:
```
if self.activate_foot_sticking / self.foot_lock.enable: <foot blocks>
if self.activate_obj_non_penetration:                   <non-penetration block>
if self._self_collision_enabled:                        <self-collision block>
```
The existing joint-limit + SOC trust-region constraints stay as-is. **All flags off
⇒ the constraint list is identical to today ⇒ the solve is bit-identical** (parity
invariant — the GMR golden/parity tests must still pass).

## Labelling
- Each transplanted block sits under a `# ===== Holosoma-style optional constraints
  (default OFF; copied verbatim from holosoma) =====` banner in both files.
- A line in each class docstring noting the optional holosoma-style constraints.
- A short note in `COMMAND.md`.

## Plumbing from config
`from_config` (both solvers) reads the flags/tolerances/configs from `cfg.retargeter`
(the `RetargeterConfig` already carries `self_collision`, foot-lock, tolerances) and
passes them to the constructor, defaulting off when absent.

## Testing (per solver)
- **Parity OFF (most important):** with all flags off, the solved qpos equals the
  current output (run the existing GMR golden/parity test; add an equivalent for
  TEST-SOCP if missing). Proves the default path is unchanged.
- **Self-collision ON:** a `SelfCollisionConfig` with a known geom pair solves without
  error and the constraint count grows.
- **Ground non-penetration ON:** `activate_obj_non_penetration=True` on a robot_only
  task solves and keeps the lowest body above the floor within tolerance.
- **Object non-penetration ON:** with an object task, the `<robot>_w_<object>.xml`
  loads, the object qpos is set per frame, and the solve runs.
- **Foot-sticking ON:** the sequence is built and the XY-window constraints apply.

## Files
- `src/gmr_socp/gmr_socp.py` — flags, copied helpers, model loading, iterate blocks.
- `src/test_socp/test_socp.py` — the same edits.
- `src/gmr_socp/from_config` / `src/test_socp/from_config` — plumbing + foot-sticking.
- `COMMAND.md` — one note.
- Tests: `tests/test_holosoma_constraints_gmr.py`, `tests/test_holosoma_constraints_test_socp.py`.
