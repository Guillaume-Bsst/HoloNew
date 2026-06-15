# SMPL-derived centroidal targets (CoM + angular momentum) for ballistic coherence

## Goal

Replace the two proxy targets of the TEST-SOCP centroidal weights with quantities
computed from the **grounded SMPL-X body**, so the robot's free-flight (ballistic)
phases are physically coherent:

- **W^c_pos / W^c** (CoM): track the **real SMPL CoM** of the grounded human instead
  of the current `pelvis + structural offset` proxy. During a rotation the pelvis does
  *not* follow a clean parabola but the CoM does, so this makes the reference truly
  ballistic.
- **W^L** (angular momentum): instead of tracking a lumped 14-link orbital `L_ref`,
  give the robot the **same rotation rate** as the human, realised with the robot's own
  inertia: `L_target = I_robot(q) · ω_human`. This is the only physically meaningful way
  to reconcile a different-mass / morphologically-scaled robot — it produces the same
  number of flips/turns in the same flight time.

Both targets are precomputed once at build time; the per-iteration solve cost is
unchanged.

## Physics rationale

- **Ballistic CoM.** In free flight only gravity acts, at the CoM: the CoM follows a
  parabola, the centroidal angular momentum is conserved. The apex height for a given
  flight time is `h = g·T²/8` — **independent of body mass and size**. The retargeter is
  frame-locked to the clip timing (`T` fixed), so the robot's vertical CoM excursion in
  flight must **equal** the human's to stay ballistic. Hence the Z of the CoM target is
  kept **un-scaled** (see CoM target below).
- **Rotation rate, not raw L.** Matching the human's absolute `L` is dimensionally wrong
  (different mass/inertia → wrong rotation). The scale-free invariant is the effective
  angular velocity `ω = I⁻¹ L` (`L ∝ m·ℓ²/T`, `I ∝ m·ℓ²` ⟹ `ω ∝ 1/T`, a pure rate in
  rad/s). Tracking `I_robot(q)·ω_human` gives the robot the human's rotation rate at the
  robot's own inertia.

## CoM target: frame and anchoring

The raw SMPL CoM sits at human height (~0.9 m standing); the robot is shorter and its
skeleton targets are already Z-scaled. Using the absolute human Z would bias the robot
upward (the bug the old `com_init − pelvis_init` offset cured). So:

```
c_target.xy(t) = scale_xy · c_human.xy(t)                       # XY scaled (sx = 1 today → no-op)
c_target.z(t)  = c_robot_init.z + ( c_human.z(t) − c_human_init.z )   # Z arc un-scaled, anchored to robot
```

- **Stance:** target ≈ the robot's own natural CoM height (no height bias).
- **Flight:** rises exactly like the human → correct apex under real `g` at the locked timing.

`c_robot_init.z` is the robot's CoM at the init config (`pin.com(q_init)`); `c_human_init.z`
is the SMPL CoM at frame 0. XY scaling reuses `scale_xy_robot` (the same placement convention
as `preprocess.scale()`); Z is never scaled. Per-axis morphological proportion scaling of the
CoM is intentionally **not** modelled (YAGNI: `sx = 1` today and Z must stay un-scaled).

`cddot_ref` (the W^c acceleration reference) is derived from the **second difference of this
real CoM target** instead of the pelvis, so W^c and W^c_pos are consistent (and `cddot_ref ≈ −g`
in flight automatically).

## SMPL-side precompute (build time, once)

`HumanBody` (SMPL-X) is currently built only in `view_stages`. Move its construction into
`builder.py` (the SMPL-X model is already available — the correspondence uses it). One pass
of `T` LBS forward evaluations at the **grounded** pose; negligible, and **zero per-iteration
solve cost**.

**Mass model (fixed from the rest pose, mass-conserving):** each mesh triangle gets a constant
mass from its **rest-pose** tetrahedron volume (triangle → origin, uniform density). We then
track the **posed triangle centroid** `p_e(t)` per frame. This conserves mass per material
element and avoids the vertex-density bias (hands/face) a naive per-vertex average would have.

Per frame `t`, from posed centroids `p_e(t)` and masses `m_e`:

```
c_h(t) = Σ m_e p_e / M                                  # human CoM
v_e(t) = (p_e(t) − p_e(t−1)) / dt                       # element velocity (finite diff)
ċ_h(t) = (c_h(t) − c_h(t−1)) / dt
L_h(t) = Σ m_e (p_e − c_h) × (v_e − ċ_h)                # full L (orbital + segment spin)
I_h(t) = Σ m_e (|r_e|² Id − r_e r_eᵀ),  r_e = p_e − c_h # instantaneous composite inertia
ω_h(t) = I_h(t)⁻¹ L_h(t)                                # effective angular velocity (mass/scale free)
```

Frames 0 (and where `q_prev` is unavailable) get zero velocity-dependent quantities, matching
the existing warm-up guard.

Stored on the retargeter:
- `rt._c_ref_all` (T, 3) — CoM target (XY-scaled, Z-anchored); **replaces** the pelvis-proxy
  computation currently in `retarget()`.
- `rt._omega_ref_all` (T, 3) — human effective angular velocity, consumed by the L term.

## W^L term rewrite (solver)

`build_lumped_L_term` is rewritten: track the robot's **full centroidal** angular momentum
toward `I_robot(q)·ω_h`.

**Robot side — centroidal L, linearised in `dqa`:**
```
L_robot(dqa) = (A_G(q0) · v)[3:6],   v = difference(q_prev, q0)/dt + (Jd/dt)·dqa
A_G(q0) = centroidal_map(q0)          # already exposed by the pinocchio backend
```
This is the conserved quantity in flight (the true ballistic invariant) and is exactly what
the "solved L" diagnostic already computes.

**Target — robot composite inertia × human ω:**
```
I_robot(q0) = Ig(q0).angular          # 3×3 angular block of the CCRBA (pin.ccrba → data.Ig)
L_target(t) = I_robot(q0) · ω_h(t)
```

**Residual:** `√λ · ( A_L·dqa + (b_L0 − I_robot·ω_h) )`, affine in `dqa`.

`I_robot(q0)` and `A_G(q0)` are frozen at the current config in the linearisation (consistent
with the rest of the SQP) and recomputed each SQP iteration.

Consequences:
- `pin_model` backend gains an accessor for the centroidal composite inertia `Ig` (via
  `pin.ccrba`). `centroidal_map` already exists.
- The lumped helpers `mapped_frame_masses_and_names` and `reference_orbital_angular_momentum`
  become unused → removed. `_L_ref_all`, `_lumped_frames`, `_lumped_masses` are dropped and
  replaced by `_omega_ref_all`.
- Activation is **unchanged**: W^L fires every frame from `frame_idx ≥ 1` (needs `q_prev`).
  No flight/contact gating (explicit decision: keep it always-on).

## Solver wiring (`test_socp.py`)

- The CoM-reference block in `retarget()` no longer computes `g_pelvis + offset`; it reads the
  precomputed `_c_ref_all`. W^c_pos consumes it unchanged (same `c_ref` signature).
- `cddot_ref` from the second difference of `_c_ref_all`.
- The W^L tracking block calls the rewritten term with `_omega_ref_all[t]` + `Ig(q0)`.

## Diagnostics (`_fill_diagnostics` + viewer)

Updates the diagnostic wiring already added for the viewer:
- `res.com_ref` ← `_c_ref_all` (now the real anchored SMPL CoM).
- `res.angular_momentum_ref` ← **`I_robot(q_t)·ω_h(t)`** recomputed post-hoc on the solved
  trajectory (the *actual* target the solver sees), replacing the old `_L_ref_all`. The orange
  "target" arrow is now the same quantity as the cyan "solved" arrow (`A_G·v`) → directly
  comparable, **no scale ambiguity**.
- The L arrows switch from per-series peak normalisation to a **shared scale factor**, so the
  magnitude gap is visible.
- The `(→tgt)` text readout stays, now consistent.

## Out of scope (YAGNI)

- No flight/contact gating of W^L (always-on, by decision).
- No per-segment anthropometric density (uniform density; `ω` is density-invariant, and the
  CoM uses the uniform-density centroid).
- No temporal rescaling of the CoM arc.
- No per-axis morphological proportion scaling of the CoM (`sx = 1`, Z un-scaled).

## Files

- `src/test_socp/builder.py`: build `HumanBody`; precompute `_c_ref_all` (anchored SMPL CoM)
  and `_omega_ref_all`; drop `_L_ref_all` / `_lumped_*`.
- `src/test_socp/centroidal.py`: rewrite `build_lumped_L_term` (centroidal L vs `I_robot·ω`);
  remove `mapped_frame_masses_and_names` + `reference_orbital_angular_momentum`. New helper to
  compute the per-frame SMPL `c_h` / `ω_h` from the posed mesh (or a new small module).
- `src/test_socp/test_socp.py`: read precomputed `_c_ref_all` / `cddot_ref`; call the rewritten
  L term; fill `angular_momentum_ref` from `I_robot·ω` in `_fill_diagnostics`.
- `src/test_socp/pin_model.py`: expose centroidal composite inertia `Ig` (`pin.ccrba`).
- `src/viewer.py`: shared-scale L arrows (target now same quantity as solved).
- Tests (see below).

## Testing

- **ω is scale/mass-free:** scaling the mesh by `s` and/or its density by `k` leaves `ω_h`
  unchanged (within tolerance).
- **CoM anchoring:** at frame 0, `c_target ≈ c_robot_init` (Z), and `c_target.z(t) −
  c_target.z(0) == c_human.z(t) − c_human.z(0)` (un-scaled arc). XY equals `sx · c_human.xy`.
- **Ballistic check:** on a synthetic free-fall mesh trajectory, `cddot_ref ≈ (0,0,−g)` and
  `L_h` (hence `ω_h`) is ~constant.
- **L term:** with `ω_h = 0` the residual reduces to `||L_robot||²` (old "drive to zero"
  behaviour); the target `I_robot·ω` matches a hand-computed value on a known config.
- **`pin_model`:** `Ig` angular block matches `centroidal_map`-derived inertia on a fixed config.
- **Diagnostics:** `angular_momentum_ref[t] == I_robot(q_t)·ω_h(t)`; viewer L arrows share one
  scale factor; no-op when fields are `None`.
```
