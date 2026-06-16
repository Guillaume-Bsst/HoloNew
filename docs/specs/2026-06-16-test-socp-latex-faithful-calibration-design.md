# TEST-SOCP — LaTeX-faithful calibration (explicit σ, style table, per-entity Lⱼ, surfaced ε)

**Date:** 2026-06-16
**Status:** design approved, pending implementation plan
**Author:** Guillaume Besset (design w/ Claude)

## Problem

The TEST-SOCP retargeter is calibrated against the LaTeX formulation
("Kinematic Retargeting Formulation for the G1 Robot"). The LaTeX deliberately
separates two families of constants:

- **priorities** `λ•` — dimensionless relative weights of the normalized terms;
- **characteristic scales** `σ•` — per-residual normalizers that make every
  residual dimensionless, so that *"weights express pure, frame-rate-invariant
  priority"*.

`config.py` exposes **all the λ**, but **silently drops almost all the σ**: they
are implicitly 1 and their scale is absorbed into the corresponding λ. Audit of
the current code:

| LaTeX constant | Current status | Evidence |
|---|---|---|
| `λ^s,λ^D,λ^X,λ^P,λ^c,λ^L,λ^o,λ^r` | present | config fields |
| `σ_qddot, σ_Vdot` | present **and used** | `temporal.py:23-24` |
| `σ_v` (P) | present but **dead** | `interaction.py:498-513` ("no longer used in the scale") |
| `σ_R` (style) | **absent** | no `/σ_R²` in `style.py` / `test_socp.py:478-501` |
| `σ_a` (W^c) | **absent** | `centroidal.py:160` `s_c = sqrt(λ_c)/dt²` |
| `σ_L` (W^L) | **absent** | absorbed into `λ_l` |
| `σ_a^o, σ_ω` (W^o) | **absent** | `movable.py:167`; asymmetry carried by `λ_o`/`λ_omega` |
| `ω_k^s, ω^B` (intra-style) | **not calibratable** | derived from GMR `w_r`: `omega = λ_ws·w_r/w_tot` (`test_socp.py:491`) |
| `Lⱼ` (field range, per entity) | **single shared value** | `rt.smplx_ground_probe.margin` for both floor + object |
| `ε` (self-collision margin) | **in companion config** | `SelfCollisionConfig.tolerance` (`holosoma_constraints.py:50`) |

Absorbing σ into λ is mathematically neutral (`λ‖r/σ‖² = (λ/σ²)‖r‖²`), but it
costs the three things the LaTeX promises:

1. **fps-invariance** — W^c folds `1/dt²`, W^r folds `1/dt⁴` without the σ that
   compensates, so changing `fps` silently requires re-tuning λ;
2. **inter-term comparability** — `λ_c=1e-5` vs `λ_d=20` are not on the same
   physical scale, defeating "λ = relative priority";
3. **honesty** — `σ_v` is a documented-but-inert field (a trap).

## Goal

Make the calibration rigorously faithful to the LaTeX: reintroduce the explicit
σ normalizers (λ become pure O(1) priorities), make the intra-style distribution
independently configurable via a dedicated table, give each interaction entity
its own range Lⱼ, and surface ε in the TEST-SOCP config.

**Non-goals.** No change to the cost *math* beyond normalization; no new cost
terms; the W^L "track L_ref vs drive-to-zero" limitation (no reference velocity
available) is out of scope — the σ_L work applies to whichever residual is
active. No data-dependent σ (scales stay fixed calibratable constants).

## Approach (chosen)

**Approach A — σ as flat config fields, folded at residual assembly.** Each
`build_*` divides its residual by σ (i.e. `sqrt(λ) → sqrt(λ)/σ`), exactly the
pattern already in `temporal.py` for `σ_qddot`/`σ_Vdot`. Matches the config's
"flat and explicit: every field maps 1:1" philosophy. (Rejected: a
`CharacteristicScales` dataclass — needless indirection; data-derived σ —
breaks reproducibility across clips.)

**Calibration policy — physical σ + clean O(1) re-tune.** σ take physically
meaningful defaults (below). **Every σ is a flat constant in `config.py` — no σ
is auto-computed from the model or the clip data** (keeps the config "flat and
explicit", and the values reproducible and inspectable in one place). All λ are
reset to O(1) pure priorities. To avoid
blind re-tuning, each λ is *seeded* from the absorption identity then rounded:

```
λ_new ≈ λ_old · σ² · (dt factors already folded)
```

then re-validated brick by brick on the reference clips (the existing brick1-5
discipline / scoreboard).

## Brick 1 — explicit σ normalizers (core)

New flat config fields with physically motivated defaults:

| σ field | Term | Default | Rationale |
|---|---|---|---|
| `sigma_R` | S_k, S_B (style) | `0.2` rad (~11°) | typical joint-orientation tracking error; residual is an SO(3) rotvec in rad |
| `sigma_a` | W^c (CoM accel) | `9.81` m/s² (= g) | the scale the LaTeX itself invokes (ballistic CoM → \|c̈\|=g) |
| `sigma_L` | W^L (ang. momentum) | `10.0` kg·m²/s | flat tunable scale (LaTeX "morphology-scaled"); ≈ G1 M·R_g²·ω_char order of magnitude, hand-set, **not** computed |
| `sigma_ao` | W^o linear | `9.81` m/s² (= g) | equivalence principle: free fall is −g regardless of mass |
| `sigma_omega` | W^o angular | `2π` rad/s (1 rev/s) | characteristic manipulation spin rate |
| `sigma_v` | P (slide) | `0.05` m/s (existing, **reactivated**) | per-frame characteristic slide `σ_v·Δt` |

`σ_qddot`/`σ_Vdot` unchanged (already correct).

**Residual changes (`sqrt(λ) → sqrt(λ)/σ`):**

- `style.py` / `test_socp.py:478-501`: S_k and S_B residuals divided by `σ_R`.
- `centroidal.py`: W^c residual `/σ_a`; W^L residual `/σ_L`.
- `movable.py:build_wo_term`: linear residual `/σ_ao`, angular residual `/σ_omega`.

**Two faithfulness wins that fall out:**

1. **Collapse W^o to a single `λ^o`.** With `σ_ao` and `σ_omega` explicit, the
   linear/angular asymmetry moves into the σ, recovering the LaTeX's single
   `λ^o`. `lambda_omega` is **removed** from config; `build_wo_term` takes one
   `lambda_o`. Builder plumbing (`builder.py:101`) updated.

2. **Restore the faithful P normalization.** `interaction.py:504-513` fell back
   to `L²` because the faithful `(σ_v·Δt)²` gave weight ~3.6e5 that wrecked
   CLARABEL — an artifact of the *un-re-tuned* λ_p. With the O(1) re-tune, λ_p
   drops to compensate and `(σ_v·Δt)²` becomes well-conditioned again. P is
   restored to the paper form (`build_p_terms` uses `scale_sq = λ_p / (σ_v·dt)²`),
   and `sigma_v` becomes live again.

**Files:** `config.py`, `style.py`, `test_socp.py`, `centroidal.py`,
`movable.py`, `interaction.py`, `builder.py`.

## Brick 2 — `STYLE_WEIGHT_TABLE` (independent intra-style distribution)

New table in `tables.py`, modeled on `IK_MATCH_TABLE1`:

```python
# robot_frame -> omega_k   (raw per-body style weight)
# special key "__pelvis_tilt__" -> omega_B   (the S_B gravity-tilt term)
STYLE_WEIGHT_TABLE: dict[str, float] = { ... }
```

Normalized internally so `Σ ω_k + ω_B = 1` (the LaTeX constraint).
`test_socp.py:484-491` stops reading the GMR `w_r` and reads this table instead,
decoupling style priority from the IK tracking weights.

**Default = uniform** over the rotation-tracked bodies + pelvis tilt → behavior
identical to today until the user re-weights (e.g. arms > legs for style).

**Files:** `tables.py`, `test_socp.py`.

## Brick 3 — per-entity Lⱼ

Replace the single shared `margin` (`rt.smplx_ground_probe.margin`) with two
config fields `L_floor` / `L_object`, threaded through `query_entities`,
`frame_references`, `floor_field`, and every `_activation(d_ref, L)` call.

**Note:** Lⱼ plays a double role — activation distance **and** positional scale
(the `/L²` in D and X). Each channel (floor, object) uses its own L for **both**
roles. Default: both = current `margin` value → behavior preserved until the
user diverges them.

**Files:** `config.py`, `interaction.py`, the probe wiring (`contact/`), `builder.py`.

## Brick 4 — surface ε in config

New field `self_collision_margin` in `TestSocpRetargeterConfig` that **feeds**
`SelfCollisionConfig.tolerance` rather than replacing the machinery — ε is
surfaced without relocating the inherited Holosoma plumbing (zero risk to the
companion config). Default = current Holosoma value.

**Files:** `config.py`, `builder.py`.

## Implementation order

1. **B1 (σ)** — first; it carries the re-tune.
2. **B4 (ε)** — trivial surface field.
3. **B3 (Lⱼ)** — per-entity range.
4. **B2 (style table)** — last; the only brick opening a new fine-tuning lever.

Each brick is independently activatable/testable, behavior-preserving by default
(except B1, which intentionally re-tunes), and validated before the next.

## Re-tune & validation

For every weighted term, the spec/plan tabulates: `λ_old`, `σ`, the seed
`λ_new`, and the validation criterion. Validation per brick:

- no regression on the reference clips (scoreboard: 7 metric families, style the
  headline criterion);
- the existing parity / contact / correspondence tests stay green where they are
  σ-agnostic; tests asserting absolute objective values get their expected
  numbers regenerated alongside the re-tune (the re-tune is intended, not a bug).

## Risks

- **Re-tune invalidates current tuned λ** (commits `259cff7`, `3a89f2b`) — accepted;
  seeds from the absorption identity keep it tractable.
- **Restored P conditioning** — must confirm CLARABEL stays stable at the new λ_p;
  fallback is to keep the L² normalization if conditioning regresses (documented
  divergence), but the goal is the faithful form.
- **σ_L flat default** — `10.0` is an order-of-magnitude guess; W^L is currently
  a weak regularizer anyway, so re-tune `λ_l` against it on a clip with real
  rotational content before trusting the pair.
