# Holosoma-Parity Weights — Design Spec

Date: 2026-06-16
Status: approved (design), implementation pending

## Goal

Port the four missing Holosoma objective terms into TEST-SOCP as **flat, independent
config fields** (one field = one effect, no opaque bundle), so the same TEST retargeter
can replicate, via config only:
- **GMR-SOCP mode**: per-point position + orientation tracking (already in TEST).
- **Holosoma mode**: the Holosoma objective (the four terms below), with pos/orient
  tracking OFF.

Reference for parity = the native Holosoma retargeter already in HoloNew
(`src/holosoma/interaction_mesh_retargeter.py`).

## Finding: the native Holosoma objective (the parity target)

`interaction_mesh_retargeter.solve_single_iteration` sums exactly four cost terms
(lines 650-676) — there is **no** pos/orient tracking term; the Laplacian shape match
is Holosoma's primary objective:

1. **Laplacian deformation** (l.653), `laplacian_weights = 10`:
   `||sqrt(w) (L v(q) - target_lap)||^2`, over the interaction mesh
   v = [robot mapped-joint points (JOINTS_MAPPING) ; object points].
2. **Nominal tracking** (l.656), `w_nominal_tracking_init = 5.0`, `tau = 10`:
   `w_nom ||dqa[idx] - (q_nominal[idx] - q_a_last[idx])||^2` on
   `NOMINAL_TRACKING_INDICES`, with `w_nom = w_init * exp(-i / tau)` over SQP iter i.
3. **Q_diag reg** (l.663): `||sqrt(Q_diag) (dqa + q_a_last)||^2` — pulls the absolute
   new actuated config toward zero (the nominal standing pose).
4. **Smoothness** (l.667), `smooth_weight = 0.2`:
   `smooth_w ||dqa - (q_prev_frame_a - q_a_last)||^2`.

So **Holosoma mode** = these four ON + `activate_pos_tracking=False`,
`activate_rot_tracking=False`. **GMR mode** = tracking ON, these four OFF (≈ current
TEST default).

## Flat config fields (no bundle)

Add to `TestSocpRetargeterConfig` §2, mirroring the existing `activate_w<sym>` + weight
style; defaults set to Holosoma's values so Holosoma mode is four flag flips:

```
activate_wlap: bool = False     # Laplacian interaction-mesh deformation
lambda_lap: float = 10.0
activate_nominal: bool = False  # nominal-pose tracking on selected joints
lambda_nominal: float = 5.0
nominal_tau: float = 10.0
activate_qdiag: bool = False    # absolute joint-config regularizer
lambda_qdiag: float = 1.0       # scalar scaling the per-joint Q_diag vector (from constants)
activate_smooth: bool = False   # step-toward-previous-frame smoothness
lambda_smooth: float = 0.2
```

Builder maps each `activate_X` to the weight passed into the solve (0.0 when off),
exactly like `lambda_d = sc.lambda_d if sc.activate_wd else 0.0`.

## Term implementations (pure where possible, assembled in solve_single_iteration)

All four are appended to `obj_terms` in `solve_single_iteration`, gated on their weight
> 0, using TEST's active-tangent `dqa` (actuated-joint step; for revolute joints this
equals Holosoma's joint delta, so the residuals are directly comparable).

- **Smoothness**: `sqrt(lambda_smooth) (dqa - (q_prev_a - q_a_last))`, where `q_prev_a`
  = previous frame's actuated config (`q_t_last[q_a_indices]`), `q_a_last` = current
  accepted actuated config. `cp.sum_squares`.
- **Q_diag**: `sqrt(lambda_qdiag * Q_diag) (dqa + q_a_last)`, `Q_diag` per-joint vector
  from `task_constants.Q_DIAG` (mirror Holosoma's `self.Q_diag`).
- **Nominal**: on `idx = task_constants.NOMINAL_TRACKING_INDICES`,
  `sqrt(w_nom) (dqa[idx] - (q_nominal[idx] - q_a_last[idx]))`,
  `w_nom = lambda_nominal * exp(-i / nominal_tau)` with i the inner SQP iteration index
  (threaded from `iterate`). `q_nominal` = the nominal actuated config (TEST's
  `q_init_full` actuated slice, matching Holosoma's nominal source).
- **Laplacian** (new module `evaluation`-independent, in `src/test_socp/`):
  - vertices `v = [P_robot_mapped ; P_object]`: robot mapped-joint world points via FK
    (the JOINTS_MAPPING links), object points from `object_surface_local` (object tasks)
    or none (robot-only → skeleton-only mesh).
  - adjacency: `get_adjacency_list(Delaunay(source_vertices))` from the SOURCE vertices
    (SMPL mapped joints + object reference points), per frame; `target_lap =
    calculate_laplacian_coordinates(source_vertices, adj)`.
  - `L = calculate_laplacian_matrix(v, adj)`, `J_L = kron(L, I3) @ J_V`, `J_V` stacked
    point Jacobians (active columns). Residual (direct substitution, no aux variable):
    `sqrt(lambda_lap) (J_L[:, active] dqa + (lap0 - target_lap).reshape(-1))`.
  - Reuse `src/holosoma/interaction_mesh.py`: `calculate_laplacian_matrix`,
    `calculate_laplacian_coordinates`, `get_adjacency_list`, `create_tetrahedra` (or its
    Delaunay).

## Parity verification

`tests/test_parity_test_socp_vs_holosoma.py`:
- Run TEST in Holosoma mode (`activate_pos_tracking=False, activate_rot_tracking=False,
  activate_wlap/nominal/qdiag/smooth=True`, Holosoma weights) on sub3_largebox_003.
- Run the native Holosoma retargeter on the same clip (or load its golden
  `tests/golden/holosoma_vanilla_qpos.npz` / `baseline_qpos.npz`).
- Report `max|Δqpos|`. Assert a tolerance to be set empirically after the first
  measurement (start by printing, then pin a realistic atol).

**Honest scope note:** TEST and Holosoma use different Jacobian backends (pinocchio
active-tangent vs Holosoma's `_calc_manipulator_jacobians`) and separate SQP loops
(n_iter, step_size, base handling). Term-faithful porting may not give bit-exact
parity without also aligning the solve loop; the test measures the gap, and loop
alignment is a follow-up if the gap is too large. We do not assert 1e-12 a priori.

## Decomposition / order

1. Smoothness, Q_diag, Nominal (simple direct residuals) — fast, each verified by an
   FD/structure unit test; measure parity contribution incrementally.
2. Laplacian (interaction mesh + Delaunay + J_L) — the large piece; FD-verify J_L
   against the finite difference of `L v(q)`.
3. Holosoma-mode parity test against the native retargeter; measure and pin tolerance.

## Out of scope
- Aligning the SQP solve loop to Holosoma for bit-exact parity (follow-up if needed).
- GMR-SOCP-mode parity test (TEST already has tracking; add a thin test if the gap to
  GMR-SOCP is of interest, but the GMR path is unchanged by this work).
