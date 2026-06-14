# Inertia Mode — design

**Date:** 2026-06-14
**Author:** driven by G. Besset (paper formulation), implemented in TEST-SOCP
**Status:** implemented (validated on largebox; flight branch unvalidated — no flight clip)

**Implementation note (2026-06-14):** during validation we found the apparent
inertia "drift" was mostly a constant ~0.25 m frame offset: GMR scaled the root XY
toward the origin (`root_xy_scale=smpl_scale≈0.68`) while the interaction fields
used the raw grounded pelvis. Fixed separately by setting `root_xy_scale=1.0` for
the whole TEST-SOCP pipeline (commit aligning GMR targets to the raw pelvis;
λ_D/λ_X re-tuned 5→20 in the aligned frame). With that fixed, inertia mode places a
physically coherent body (feet planted, foot slip ~4 mm) at a constant offset from
the human pelvis — the robot's morphology puts its pelvis-over-feet differently
than the human. Decision: accept inertia as *physical placement, not human-pelvis
tracking*. The coherence metric checks feet-planted + no-runaway, not distance to
the human pelvis.

## Goal

Make the robot's **placement emerge from contacts and centroidal dynamics**,
not from a positional pelvis target — the paper's actual philosophy ("the pelvis
is never placed by a positional target; its placement is inherited from the
centroidal momentum"). When contacts are present they place the body; when they
thin out or vanish (free flight) the ballistic centroidal trajectory
($\ddot c_{ref}\to -g\hat z$, already in `cddot_ref`) takes over.

This replaces today's anti-paper crutches — the **pelvis position scaffold**
(`pelvis_anchor_weight`) and the **CoM position anchor** (`lambda_c_pos`) — which
were band-aids for the fact that `robot_only` strips all interaction (so nothing
anchored the body) and W^c is acceleration-only (position-blind).

## Root cause (why W^c "drifted")

It was never a W^c bug. On `robot_only` the interaction costs (D/X + persistence)
are force-disabled for native parity, so there are **zero contacts** to anchor the
body; W^c only matches the local CoM curvature (second difference) and cannot pin
absolute position, so a constant-velocity error integrates into drift. On a real
clip the feet↔floor contacts exist and anchor the body. The fix is therefore to
**make the floor a permanent interaction entity** (feet track the floor field
always), drop the positional crutches, and let a **weak** W^c govern only the
residual freedom — so the "contacts place / ballistics fill" switch is automatic,
with no phase logic.

## The central insight: weak W^c needs no phase switch

W^c's effective weight on the CoM is `lambda_c / dt^4` (the `c_ddot = Δ²c/dt²`
folding), with `dt=1/30` giving `1/dt^4 ≈ 8.1e5`. If `lambda_c` is tuned so that
W^c stays **well below the interaction weights** (`lambda_D = lambda_X = 5`):

- **Stance / contact:** the contact costs dominate and place the body absolutely;
  W^c only shapes the redundant DoF (legs/torso) — a smoothing prior.
- **Flight / few contacts:** the contact costs deactivate; W^c becomes the *only*
  term acting on the CoM, so it follows the reference acceleration (ballistic
  parabola) integrated from the last stance state.

No explicit phase detection is coded; the activation arithmetic of the contacts
provides it for free.

## Architecture: two modes

A single config flag selects the behaviour. **The default is unchanged**, so all
existing parity guarantees hold.

- **Parity mode (`inertia_mode=False`, default):** pelvis scaffold ON, floor-as-
  entity OFF, W^c/W^L OFF. `test_retarget_golden` and `test_test_socp_parity`
  remain **bit-exact**. Nothing moves.
- **Inertia mode (`inertia_mode=True`):** `from_config` applies the bundle:
  `floor_as_entity=True`, `pelvis_anchor_weight=0`, `lambda_c_pos=0`,
  `activate_centroidal=True`, `lambda_c>0` (weak, tuned), `lambda_L>0` (weak).
  One flag, not five.

`inertia_mode` and `floor_as_entity` are added to `TestSocpRetargeterConfig`
(both default `False`). `inertia_mode` is the user-facing switch; `floor_as_entity`
is the lower-level mechanism it turns on (kept separable for testing).

## Components

### C1 — Floor as a permanent interaction entity

Today the floor channel (correspondence + SMPL-X ground probe + analytic floor
field) is built only when `object_sdf is not None` (`from_config` ~line 1564),
and `robot_only` sets `smplx_ground_probe=None`. The object SDF and the ground
probe are coupled: `build_smplx_ground_probe` takes `object_sdf` and `obj_poses`.

Changes:
1. **Floor-only ground probe.** `build_smplx_ground_probe` gains a floor-only path
   (`object_sdf=None`, `obj_poses=None`): it still samples the SMPL-X surface and
   produces robot control points + floor distances, but performs no object SDF
   query. The probe's per-frame output keeps the object field but marks it
   inactive (distance `+inf`, `active=False`).
2. **Load floor assets under `floor_as_entity`.** In `from_config`, build the
   correspondence + ground probe whenever `floor_as_entity` OR an object is
   present — not only for object tasks.
3. **`query_entities` guards `object_sdf=None`.** When `rt.object_sdf is None`,
   return an all-inactive object `ContactField` (distance `+inf`, `active=False`,
   zero direction/witness) instead of calling `object_sdf.query`. The floor field
   is computed as today.
4. **Interaction guards allow floor-only.** The D/X / persistence activation
   guards in `solve_single_iteration` currently require `object_sdf is not None`.
   Relax to: require `correspondence is not None` AND
   (`object_sdf is not None` OR `floor_as_entity`). The object channel naturally
   contributes nothing when its activations are zero, so D/X/P run floor-only.
5. **`from_config` weight gating.** Under `floor_as_entity` (even for
   `object_name in (None, "ground")`): do **not** zero `lambda_D/lambda_X`, set
   `activate_persistence=True`, enable ground non-penetration
   (`activate_obj_non_penetration=True`, `load_object_scene=False`).

Result: the feet (and any link touching the floor — the correspondence samples
every G1 link) are pinned to their reference floor positions by D (normal),
X (tangential placement) and the persistence band (no-slip), anchoring the body
absolutely during stance.

### C2 — Drop the positional crutches (inertia mode only)

Under `inertia_mode`, `from_config` sets `pelvis_anchor_weight=0` (Style does
orientation only: pelvis-relative joint orientations + pelvis roll/pitch tilt)
and `lambda_c_pos=0` (no CoM position anchor). No code path is removed — both
remain available in parity mode and as independent knobs.

### C3 — W^c weak, enabled

`activate_centroidal=True`, `lambda_c>0`. `lambda_c` is the central tuning knob:
swept so that W^c stays clearly below the interaction terms while still governing
the CoM when contacts vanish. Expected range ~1e-6…1e-4 given the `1/dt^4`
amplification. Tuning target: on a grounded clip, enabling W^c must NOT worsen the
contact-anchored body placement (it only smooths the residual); the metric is
"pelvis drift bounded without any positional target".

`W^c_pos` is left in the code (parity/opt-in) but off in inertia mode.

### C4 — W^L kept weak toward zero (decision (a))

The current `build_centroidal_terms` W^L drives angular momentum to **zero**
(`b_L = AgL @ v0`, no `L_ref`) — a divergence from the paper's `||L - L_ref||²`.
Tracking `L_ref` properly needs a reference robot **velocity** (not just target
orientations) and a flight clip to matter; we have neither. Decision: keep W^L as
a **weak toward-zero spin regularizer** (`lambda_L` small), which is sane for the
grounded manipulation/climb clips we have. Document that true `L_ref` tracking is
deferred pending (i) a reference-velocity pipeline and (ii) a free-flight clip.

### C5 — Free-flight branch

The ballistic branch is implemented (it is just W^c being the sole CoM term once
contacts deactivate, with `cddot_ref` carrying `-g\hat z` from the data). It is
**marked implemented-but-unvalidated**: no clip in `demo_data` contains a flight
phase, and per the chosen scope we add no synthetic flight test. Documented in the
spec and the centroidal config comment.

## Data flow (inertia mode, per frame t)

1. `smplx_ground_probe(t, ...)` → robot control points + floor distances (object
   field inactive when floor-only).
2. `query_entities` → `(fobj inactive | active, fflr)`.
3. `frame_references(t)` → floor reference distances/positions (object refs absent
   or zero-activation).
4. Objective: Style (orientation only) + D/X (floor channel, λ=5) + W^r + W^c
   (weak) + W^L (weak) [+ object channel + W^o when an object is present].
5. Constraints: trust region + persistence band (floor) + ground non-penetration.
6. Body placement emerges: stance → feet pinned by D/X/P; residual → W^c.

## Validation & tests

- **Largebox (object task, already has the floor channel).** `inertia_mode=True`:
  assert finite, pelvis z sane, **bounded pelvis drift with `pelvis_anchor_weight=0`
  and `lambda_c_pos=0`** (the key test — the body holds by contacts alone), contact
  gap not worse than parity mode, jerk not worse.
- **Floor-only on a non-object task.** Confirm the floor channel loads and the
  solve runs finite with `floor_as_entity=True` and no object (the C1 path).
- **Climb (varied hand/foot contacts).** Finite + bounded drift. **Risk to verify
  first:** climb clips are `.npy` + multi-box assets, not the OMOMO `.pt`/smplh
  path `from_config` expects; if not loadable, climb falls out of validation scope
  and largebox + the floor-only smoke test carry it.
- **Parity preserved.** `test_retarget_golden` + `test_test_socp_parity` stay green
  (mode off by default; bit-exact).
- **Inertia-mode snapshot.** New re-baselined golden for the `inertia_mode` output
  (future regression guard).
- **Unit guards.** `query_entities` with `object_sdf=None` returns an all-inactive
  object field; `build_smplx_ground_probe` floor-only path produces finite points
  + floor distances.

## Error handling / risks

- **`lambda_c` mis-tuning** (the main risk): too small → residual drift in weak-
  contact stretches; too large → W^c fights the contacts (the 2.5 m drift we saw
  when centroidal was naively enabled on object tasks). Mitigation: sweep against
  the largebox placement metric; keep W^c strictly below the interaction weights.
- **Floor-only probe refactor** touching `build_smplx_ground_probe` could perturb
  the object-task path. Mitigation: the floor-only branch is gated on
  `object_sdf is None`; the object path is unchanged and guarded by the existing
  object-task tests.
- **D-term floating-base runaway** (historically base z → -36 m) is prevented by
  the coupled ground non-penetration, kept on under `floor_as_entity`.

## Out of scope

- True `L_ref` tracking (needs reference-velocity pipeline + flight clip).
- Synthetic flight test (per chosen validation scope).
- Making `inertia_mode` the default (revisit once validated; parity stays default).
