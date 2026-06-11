# Modular retargeting visualization for HoloNew (holosoma)

**Date:** 2026-06-11
**Status:** Design — pending implementation plan
**Scope:** `modules/01_retargeting/HoloNew` (a copy of `holosoma_retargeting`)

## Goal

Reproduce, inside HoloNew and faithfully on holosoma's own framework, the
viser-based workflow currently done in `test_pipe`: load an OMOMO sequence,
retarget it, and inspect **one or several retargeted trajectories in the same
viser session** (holosoma's native interaction-mesh result, a future
"GMR-style" result solved by the same SOCP solver, and the intermediate stages
that precede the solve).

The SOCP solver is **never modified**. mink is not ported. The work is a
viz/architecture refactor of holosoma that adds modularity, plus reserved seams
for later contact/correspondence overlays.

## Background: the two pipelines

- **HoloNew / holosoma** — `src/interaction_mesh_retargeter.py`
  (`InteractionMeshRetargeter`). Per-frame retargeting solved as an **SOCP**
  (`cvxpy` + `CLARABEL`) on the actuated-joint displacement `dqa`. Objective:
  interaction-mesh Laplacian tracking + nominal tracking + regularization +
  smoothness. Constraints: SDF non-penetration (linearized Jacobian),
  self-collision, joint limits, foot sticking/lock, bounded step (`cp.SOC`).
  Already ingests OMOMO (`--data_format smplh`) and already embeds a viser
  server with drawing helpers.
- **test_pipe** — GMR-style retargeting: per-body `mink.FrameTask` targets
  solved by velocity IK (`mink.solve_ik`). Provides the architecture we want to
  mirror: a `StageSpec` registry, a panel-based viser app, and (point 4) OT
  human→robot correspondence (`transport/`) and contact fields (`fields/`).

## The three coupling locks to undo

The blocker is not the solver; it is three couplings inside the retargeter.

### Lock 1 — compute and render are interleaved
`retarget_motion` draws (`draw_keypoints`, `draw_q`) **inside** the per-frame
compute loop, and at the end wires `create_motion_control_sliders` to the single
robot. Fix: make `retarget_motion` **pure** — it returns a trajectory plus
per-frame stage data, and draws nothing. Rendering moves to a separate layer.

### Lock 2 — single robot instance
Visualization owns exactly one `ViserUrdf` (`self.viser_robot` /
`self.robot_base`). Fix: extract a `Viewer` that owns the viser server and a
**dict of robot instances** keyed by stage, each under its own scene root
(`/world/<stage>`), plus the object, grid, and named keypoint sets.
`draw_keypoints(p, name=...)` is already multi-name, so it needs no change.

### Lock 3 — no stage registry
There is no single source of truth for "what can be displayed". Fix: adopt
test_pipe's pattern — a `StageSpec` registry that the dropdown, ghost overlay,
playback and mesh gating all derive from.

## Proposed architecture

New, small, isolated units (names indicative):

### `StageSpec` registry — `stages.py`
Single source of truth for trajectory stages. Mirrors test_pipe's
`stage_registry.py`.

```
StageSpec(label, key, produces_qpos)
```

Natural holosoma stages:

| label        | key         | produces_qpos | rendered as            |
|--------------|-------------|---------------|------------------------|
| Original     | `None`      | False         | raw human skeleton     |
| Mapped       | `mapped`    | False         | mapped joints          |
| InObject     | `in_object` | False         | joints in object frame |
| SOCP         | `socp`      | True          | robot mesh (native)    |
| GMR-SOCP     | `gmr_socp`  | True          | robot mesh (2nd prod.) |

The dropdown options, ghost overlay options, and "does this drive a robot mesh"
gating all derive from this list. Adding a stage = one entry here plus a
producer that fills its data.

### `Viewer` — `viewer.py`
Owns the viser server and all scene state, extracted out of the retargeter:
- one `ViserUrdf` robot instance **per `produces_qpos` stage**, under
  `/world/<stage_key>`;
- the object `ViserUrdf` and grid;
- named keypoint layers (reusing `draw_keypoints`);
- a **single** playback slider / play-pause / fps that drives the current
  frame; each visible robot stage updates its own `update_cfg` + base frame,
  each skeleton stage updates its keypoints;
- a stage dropdown + ghost dropdown derived from the registry.

The retargeter holds no viser state. It may receive a `Viewer` to push results
into, or (cleaner) just return data the caller hands to the `Viewer`.

### Pure `retarget_motion`
Returns a structured result per sequence:
- final qpos trajectory (`socp`);
- per-frame stage data (mapped joints, object-frame joints, interaction mesh /
  tetrahedra, object points);
- cost.

No `draw_*` calls inside. Existing `.npz` saving is preserved. **Verification:
output qpos must be byte-for-byte identical to the current code on a reference
sequence** (pure refactor — the solve is untouched).

### Second trajectory producer: `GMR-SOCP`
"GMR-style resolution with the SOCP solver" = a producer that builds per-body
position/orientation targets (the GMR `IK_MATCH_TABLE` idea) and solves them
through holosoma's existing `iterate()` SOCP — **not** mink. It outputs a second
qpos trajectory registered under stage `gmr_socp`, displayed alongside the
native `socp` robot in the same viser. The objective/target assembly is new
code; the solver call is reused unchanged.

> Note: whether `iterate()` can accept frame-task-style targets directly, or
> needs a thin alternative objective-assembly entry point that still calls the
> same `solve_single_iteration`, is an implementation-plan question. The
> constraint is: reuse the SOCP solve, do not fork the cvxpy formulation's
> constraint machinery.

## Forward compatibility — point 4 (contact / correspondence)

The framework must let OT correspondence and contact fields plug in **later
without touching stage logic**. Reserve the seam now via a separate concept:

### `OverlayLayer` (reserved, not implemented yet)
A toggleable visual layer fed by **optional per-frame data** attached to the
sequence, distinct from trajectory stages:
- **contact field** — holosoma already computes SDF distances inline as
  constraints; an overlay would surface them as a visualizable field;
- **OT correspondence** — human→robot surface coupling ported from
  `test_pipe/transport/` (the one genuinely new module), shown as
  point/line layers.

A per-sequence `SharedData`-like container carries stages plus optional overlay
data (`contact_field`, `correspondence`), mirroring test_pipe's `SharedData`.
Overlays are independently toggled, like test_pipe's `ContactFieldPanel` /
`G1ContactTransportPanel`. Defining `OverlayLayer` and the data container now
(even with zero implemented overlays) is what makes point 4 additive.

## Out of scope (this design)

- Any change to `iterate()`, `solve_single_iteration()`, the cvxpy/CLARABEL
  formulation, or the constraint set (SDF, self-collision, foot lock).
- Porting mink or the GMR velocity-IK solver.
- Implementing OT correspondence or contact-field overlays (point 4) — only the
  seam is reserved here.

## Incremental delivery

1. **Extract `Viewer` + make `retarget_motion` pure.** Pure refactor; verify
   identical qpos on a reference OMOMO sequence.
2. **Add `StageSpec` registry + multi-robot viewer** (dropdown, ghost, single
   slider) with the native `socp` trajectory only.
3. **Add the `GMR-SOCP` producer**, displayed next to the native result.
4. *(later)* Add `OverlayLayer` implementations: contact field, then OT
   correspondence from `test_pipe/transport/`.

## Open questions for the implementation plan

- Does the retargeter push into `Viewer`, or return data the caller renders?
  (Leaning: return data — keeps the retargeter render-agnostic.)
- Exact entry point for `GMR-SOCP` targets into the SOCP solve.
- Which holosoma intermediate quantities are worth exposing as stages beyond
  `mapped` / `in_object`.
```
