# Brick 3 design: Pelvis-relative Style (replaces world-frame tracking)

Part of [the TEST-SOCP paper-formulation meta-spec](2026-06-13-test-socp-paper-formulation-meta-spec.md).
The brick that replaces the current core objective (world-frame position +
orientation tracking via the IK match tables) with the paper's Style component.

## Scope

Replace world-frame tracking with pelvis-relative joint-orientation tracking
plus a roll/pitch-only pelvis anchor; keep a weak pelvis translation anchor as a
transitional scaffold (removed in brick 4). Decision variable `dqa`.

## The terms

- **Pelvis-relative joint orientation:** `S_k = (1/σ_R²) ‖ R̃_k^ref ⊖ R̃_k ‖²`,
  `R̃_k = R_B⁻¹ R_k`. Reference `R̃_k^ref = (R_B^ref)⁻¹ R_k^ref` (frozen per
  frame). Linearization: the relative angular Jacobian
  `J̃_k = R_B⁻¹ (J_ωk − J_ωB)` is formed from the world angular Jacobians of link
  `k` and the base (already available); the error is the rotvec
  `log(R̃_k⁻¹ R̃_k^ref)`.
- **Roll/pitch-only pelvis anchor:** `S_B = (1/σ_R²) ‖ (R_B^ref)ᵀ ẑ − R_Bᵀ ẑ ‖²`.
  `R_Bᵀ ẑ` is world-up expressed in the pelvis frame → depends on tilt only,
  invariant to yaw. Linearized against the base angular increment.
- `W^s = Σ_k ω_k^s S_k + ω^B S_B`, with `Σ_k ω_k^s + ω^B = 1`, `ω ≥ 0`.

## Replacement logic

Per-joint pelvis-relative orientations **determine the pose shape** by forward
kinematics. Therefore the world-frame **position and orientation** tracking is
removed; body positions **emerge** from the joint orientations + base pose +
brick-1 contacts. The one remaining unconstrained DOF is the **pelvis
translation**, held by a weak translation anchor (the meta-spec scaffold) until
brick 4 (centroidal) removes it. Pelvis **yaw becomes free** (left to the
interactions), since `S_B` is yaw-invariant.

## References

`R_k^ref`, `R_B^ref` come from the source motion (per-body ground quaternions;
confirm availability of per-joint reference orientations in `from_config`). The
pelvis-relative references are precomputed per frame.

## Config / weights

Add to `TestSocpRetargeterConfig`: `lambda_s`, normalized intra-style weights
`omega_k^s` / `omega_B` (sum to 1), `sigma_R`, and a transitional
`lambda_pelvis_anchor` (weak translation scaffold, removed in brick 4).

## Acceptance metric (validation gate)

1. Pelvis roll/pitch tracking error (`S_B` residual) within tolerance.
2. Yaw demonstrably **freed** — pelvis yaw no longer tracks the reference, drifts
   to satisfy contacts.
3. Joint-orientation fidelity (`S_k` residual) comparable to or better than the
   world-orientation tracking it replaces.
4. No pose collapse — the translation scaffold holds the pelvis; the solve stays
   feasible and finite.

Plus re-baselined regression snapshot + "runs without error" smoke on a full
OMOMO clip.

## Risks

1. **Most invasive brick** — removing the strong world tracking. Mitigations:
   (a) the weak pelvis translation scaffold, (b) brick-1 contacts anchoring the
   limbs, (c) careful weight tuning gated by the metric. If the pose collapses,
   temporarily strengthen the scaffold.
2. **Reference per-joint orientations**: verify the source motion provides
   per-joint `R_k^ref` (not only a few tracked bodies) so every styled joint has
   a reference. The plan's first task confirms data availability.
3. **Relative-Jacobian sign/frame conventions** (`R_B⁻¹ (J_ωk − J_ωB)`): unit-test
   against finite differences before trusting the linearization.
