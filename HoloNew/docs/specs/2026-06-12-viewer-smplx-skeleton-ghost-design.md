# Stage viewer: SMPL-X mesh, body/finger skeleton, and ghost overlay

## Goal

Enrich the per-method stage viewer (`examples/view_stages.py` + `src/viewer.py`)
with three capabilities ported from the `test_pipe` reference viewer, scaled down
to HoloNew's lean `Viewer`:

1. Show the **original motion** as a posed **SMPL-X mesh** plus the manipulated
   **object mesh**.
2. Render the **52-joint source skeleton** with independent toggles for body
   vs. finger **bones** and **joints**.
3. Expose the original motion as the **`Original` stage of every method**, and add
   a **ghost** overlay — a second, independent `(method, stage)` selection drawn
   faded for side-by-side comparison.

## Non-goals

- No port of `test_pipe`'s panel architecture (`SharedData`, `Panel`,
  `stage_registry`). We extend the existing `Viewer` class in place.
- The ghost never draws a second full robot URDF. Ghost applies to skeleton
  stages (`Original` + the per-method mapped-body stages) only; the robot stage
  is excluded from the ghost stage options.
- No new SMPL-X posing code: reuse `correspondence/human_body.HumanBody`.

## Constraints / context

- The `.pt` motion (smplh / object_interaction tasks) carries per-joint global
  quaternions at slice `[383:591]`, with intermimic's `upright_start` twist baked
  in. `load_intermimic_data` currently reads joints + object poses but drops the
  quaternions; the SMPL-X mesh needs them.
- `SMPLH_DEMO_JOINTS` is the same 52-joint MuJoCo order as `test_pipe`'s
  `JOINT_NAMES`, so the bone/joint index topology transfers verbatim.
- `correspondence/human_body.HumanBody.placed_verts(quats_wxyz, pelvis_target,
  frame_idx)` already poses an SMPL-X mesh in the Z-up world frame.
- `smplx`, `torch`, `scipy` are available in the `holonew` env and the SMPL-X
  model dir exists (`correspondence.constants.SMPLX_MODEL_DIR_DEFAULT`). All are
  treated as optional at runtime — see Graceful degradation.

## Components

### `src/skeleton.py` (new)

Pure-data topology for the 52-joint SMPLH skeleton, ported from `test_pipe`
`constants.py` and indexing `SMPLH_DEMO_JOINTS`:

- `BODY_BONES: list[tuple[int, int]]`, `FINGER_BONES: list[tuple[int, int]]`
- `BODY_JOINT_INDICES: list[int]`, `FINGER_JOINT_INDICES: list[int]`
- Colours: `COLOR_BODY`, `COLOR_FINGER`, `COLOR_GHOST_BODY`, `COLOR_GHOST_FINGER`,
  `COLOR_STAGE`, `COLOR_GHOST_STAGE` (uint8 RGB).

No imports beyond numpy. The indices are validated by tests to stay in `[0, 52)`.

### `src/utils.py` — `load_intermimic_quats(path) -> np.ndarray`

Reads `[383:591]` → `(T, 52, 4)`, undoes the `upright_start` twist
(right-multiply xyzw by `Q = [0.5, 0.5, 0.5, 0.5]`), returns wxyz. Ported from
`test_pipe/human/motion.py::_undo_upright_start`. Independent of
`load_intermimic_data` so existing callers are untouched.

### `src/viewer.py` — `Viewer` extension

New shared state, set in `bind_methods` (or a small `set_original`):

- `original_joints: (T, 52, 3)` — the raw source skeleton, identical across
  methods (this is what every method's `Original` stage renders).
- `original_quats: (T, 52, 4) | None` — for the SMPL-X mesh; `None` when
  unavailable.
- `object_poses: (T, 7) | None` and the object URDF (already supported via
  `object_model_path` / `has_dynamic_object`).
- `human_body: HumanBody | None`.

New GUI folders/handles (created in `bind_methods`):

- **Ghost**: `ghost_method_dd`, `ghost_stage_dd` (options: `Off` + skeleton
  stages, robot stage excluded), default `Off`.
- **Skeleton**: `show_body_bones`, `show_finger_bones`, `show_body_joints`,
  `show_finger_joints` checkboxes.
- **Meshes**: `show_smplx_mesh`, `show_object_mesh` checkboxes (no-op when the
  underlying data is absent).

New render helpers:

- `_draw_skeleton(prefix, joints_frame, *, ghost)` — for the 52-joint `Original`:
  body/finger bones via `add_line_segments` (per-segment colour) and joints via
  `add_point_cloud`, gated by the four toggles.
- `_draw_stage_points(prefix, pts_frame, *, ghost)` — the per-method mapped
  stages (`Mapped`/`Scaled`/`Offset`/`Ground`/`Grounded`) render as joint points
  (no bones; HoloNew ships no mapped-body bone topology), gated by the
  `show_body_joints` toggle. Solid orange / faded orange for ghost.
- `_draw_smplx_mesh(frame)` / `_draw_object(frame)` — independent of the selected
  stage; driven by their toggles.
- Ghost overlay in `_redraw`: when `ghost_stage_dd != Off`, draw the chosen
  `(method, stage)` skeleton faded under a `/ghost` prefix.

`_redraw(frame)` orchestration:

1. Active stage: robot stage → URDF (unchanged); `Original` → 52-joint skeleton;
   mapped stage → orange mapped skeleton.
2. Meshes: draw/hide SMPL-X + object per their toggles, any stage.
3. Ghost: draw/hide the faded ghost skeleton.

### `examples/view_stages.py` — wiring

- Load `original_quats` via `load_intermimic_quats` and `object_poses` via the
  existing `load_motion_data` (smplh path); guard for formats without quats.
- Build `HumanBody` from `SMPLX_MODEL_DIR_DEFAULT` inside a try/except; on any
  failure log a warning and pass `human_body=None`.
- Inject `"Original"` into every method's `stages` dict as `original_joints[:T]`
  so the stage exists uniformly (GMR methods already include it; holosoma gains
  it).
- Pass the object URDF + `has_dynamic_object` to `Viewer` for object_interaction.

## Graceful degradation

`HumanBody` build failure (missing `smplx`, missing model dir) or a data format
without per-joint quaternions (lafan / mocap / smplx robot_only) ⇒
`human_body=None`, `original_quats=None`. The SMPL-X mesh toggle becomes a no-op;
the 52-joint skeleton, object mesh, and ghost all still work. No exception
reaches the viewer. The object mesh is only wired when the task actually has an
object; otherwise its toggle is a no-op.

## Testing

- `tests/test_skeleton.py`: all bone/joint indices in `[0, 52)`; body and finger
  index sets are disjoint and cover the intended joints; bones reference declared
  joints.
- `load_intermimic_quats`: returns `(T, 52, 4)`, rows unit-norm.
- Graceful path: a viewer constructed with `human_body=None` / `original_quats=None`
  draws the skeleton without raising (logic-level, no live viser scene where
  avoidable).
- Ghost/toggle wiring exercised at the helper level (pure functions), keeping the
  viser server out of the assertion path where possible.

## Files

- `src/skeleton.py` (new)
- `src/utils.py` (+`load_intermimic_quats`)
- `src/viewer.py` (extend `Viewer`)
- `examples/view_stages.py` (wire original motion + meshes + ghost)
- `tests/test_skeleton.py` (new); extend `tests/test_viewer.py` as needed
