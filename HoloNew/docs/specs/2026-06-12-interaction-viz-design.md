# Interaction visualization in the stage viewer (test_pipe parity)

## Goal

Show, in `view_stages`, the same contact/interaction visuals as the `test_pipe`
viewer: the SMPL-X human contact probes, the object/floor probes, the contact
**directions** (probe → witness), and the per-frame **transport of the contact
field onto the G1** (the human point set placed on the solved robot via the OT
correspondence) — all coloured by signed distance. The G1 transport is the new
computation; the human-side field is already produced by the TEST-SOCP solve.

This feeds a later task: the same transported field will drive constraints inside
the TEST-SOCP objective.

## What already exists (TEST-SOCP)

`TestSocpRetargeter` (`src/test_socp/test_socp.py`) already, per frame at the
**Grounded** pose:
- `self.object_sdf` — boosted SDF (distance + witness + active).
- `self.contact_fields` — the bundled 4-channel field (`assets/contact/contact_<task>.npz`),
  Coal-computed, when present (human_object / object_human / human_floor / floor_human).
- `self.correspondence` — the human→G1 OT table (`link_idx`, `offset_local`,
  `link_names`, `human_idx`, embedded human cache `tri_idx`/`bary`).
- `self.smplx_ground_probe` — `SmplxGroundProbe(t, quats, pelvis_grounded) -> ContactField`,
  which computes the SMPL-X surface probe world points and queries the object SDF.
- `self.smplx_sdf_fields` — list of per-frame `ContactField` (human_object channel).

## New computation (in the solver, exposed on the result)

### Cache alignment (correctness)
The OT `human_idx` indexes the correspondence's embedded human cache
(`tri_idx`/`bary`). The probe must therefore use **that same cache**, not a freshly
resampled one. `build_smplx_ground_probe` will rebuild `PointCloudCache` from
`self.correspondence.tri_idx/bary` (when a correspondence is present) so the probe
point order matches `human_idx`. This is what makes the transported result
identical to test_pipe.

### Per-frame records (added to `retarget()`)
For each frame `t` (only when `smplx_ground_probe` and `correspondence` are set):
- `human_probe_pts[t]` (N, 3): the probe world points (`world` already computed
  inside the probe; expose it alongside the field).
- `human_floor[t]`: the analytic floor field for the probes (signed z distance,
  cheap), so the human probe can be coloured by `min(human_object, human_floor)`
  like test_pipe.
- `g1_transport_pts[t]` (M, 3): FK the solved robot links for frame `t`, then place
  the correspondence G1 points: `p = R_link @ offset_local + t_link` per G1 point,
  using `link_idx`/`link_names`. (A small `transport.py` helper in
  `src/test_socp/correspondence/`.)

### Exposed on `RetargetResult`
New optional fields (None for solvers that do not produce them):
- `human_probe_pts`, `human_obj_dist`, `human_flr_dist`, `human_witness`
  (the human_object witness, object-local — kept for the direction lines),
- `g1_transport_pts`, plus `human_idx` so each G1 point reads its human probe's
  distance/witness,
- pass-through `object_sdf`, `contact_fields`, and the per-frame `obj_pose`
  (to bring the object-local witness to world for the direction lines).

## Viewer

`view_stages` pulls these from the TEST-SOCP result and passes them to `Viewer`.
Only TEST-SOCP populates them; other methods leave them `None` (toggles no-op).

### "Test" GUI folder (extends the existing SDF Object toggle)
- **Human contact** — SMPL-X probes coloured by `min(human_object, human_floor)`.
- **Object contact** — object_human probes (from `contact_fields`, bundled only).
- **Floor contact** — floor_human probes (from `contact_fields`, bundled only).
- **G1 transport** — transported G1 points coloured by their human source's distance.
- **Directions** — line segments probe → witness (human) and G1 point → witness.
- **SDF Object** (exists), **SDF Floor** — the analytic floor band.

### Stage gating (test_pipe semantics, mapped to HoloNew stages)
- Human contact + human directions: shown on the **Grounded** stage (the grounded
  SMPL-X, HoloNew's equivalent of test_pipe's "Original").
- G1 transport + G1 directions: shown on the **Robot** stage (the solved robot).
- Object / Floor contact + SDF Object / SDF Floor: **any stage** (world/object
  anchored).

### Rendering
Per-frame point clouds + `add_line_segments` for directions, coloured by
`signed_distance_colors`. Persistent handles updated in place (no flicker), like
the object points / SDF. Object-frame witnesses are lifted to world by the active
stage's object pose (reusing `_object_pose`).

## Data availability
- Human contact, G1 transport, SDF, directions (human→object): **online**, work for
  any sequence that has an object SDF.
- Object / Floor contact (reverse channels): only when the **bundled**
  `contact_<task>.npz` exists (currently `sub3_largebox_003`).

## Files
- `src/test_socp/correspondence/transport.py` (new): `g1_link_transforms` + `transported_points`.
- `src/test_socp/contact/smplx_field.py`: use the correspondence cache; return probe points.
- `src/test_socp/test_socp.py`: record per-frame interaction data; expose on result.
- `src/retarget_result.py`: new optional fields.
- `examples/view_stages.py`: pull + pass interaction data for the TEST-SOCP method.
- `src/viewer.py`: Interaction toggles + `_draw_*` helpers + gating.
- Tests: `test_transport.py` (FK + placement), `test_viewer` (toggles/gating),
  cache-alignment (human_idx indexes the probe points).

## Testing
- `transported_points`: identity base + known link offset → expected world points.
- Cache alignment: probe built from `corr.tri_idx/bary` has `N == len(tri_idx)` and
  `human_idx.max() < N`.
- Viewer: each Interaction toggle creates/hides its handle on the gated stage;
  no-op when data is `None`.
