# Contacts Metric — Design Spec

Date: 2026-06-15
Status: approved (requirements from brainstorming), implementation pending
Parent: retargeting scoreboard (sub-project 2 of 3)

## Goal

Measure whether contacts are made **at the right time, in the right place, without
slipping** — the three facets the user chose:
- **timing**: do robot contacts occur on the same frames as the source contacts?
- **placement**: when in contact, is the robot contact point on the target surface?
- **no-slip**: does a sustained contact point stay put (no tangential drift)?

## Principle (same as the rest of the scoreboard)

A **pure aggregator** consumes pre-extracted per-frame / per-contact-point arrays;
extraction (contact detection, surface distance, slip) stays at the call site, which
owns the geometry. This keeps the metric backend-agnostic and unit-testable, and lets
floor contacts (analytic z=0) be scored offline while object contacts (mesh) reuse the
same function once their arrays are extracted.

## Pure function — evaluation/metrics/contacts.py

```
compute_contacts(robot_contact, ref_contact, placement_dist, slip) -> dict[str, float]
```
Inputs, all `(T, C)` over T frames and C contact points:
- `robot_contact` (bool): robot contact point in contact this frame.
- `ref_contact` (bool): source (SMPL) contact point in contact this frame (ground truth).
- `placement_dist` (float, m): robot contact point distance to its target surface.
- `slip` (float, m): per-frame tangential displacement of the robot contact point
  (caller supplies 0 for the first frame / non-contact frames).

Outputs:
- **Timing** — per (frame, point) binary classification, positive = `robot_contact`,
  ground truth = `ref_contact`:
  - `contact_precision = TP / (TP + FP)`
  - `contact_recall    = TP / (TP + FN)`
  - `contact_f1        = 2PR / (P + R)`
  with `TP = robot & ref`, `FP = robot & ~ref`, `FN = ~robot & ref`. Each guarded to
  `0.0` when its denominator is 0.
- **Placement** — `contact_place_err` = mean `placement_dist` over frames where both
  `robot_contact & ref_contact` (when both agree there is contact, how far off is it).
  `0.0` if that mask is empty.
- **No-slip** — `contact_slip_mean` = mean `slip` over *sustained* contacts
  (`robot_contact[t] & robot_contact[t-1]`). `0.0` if no sustained contact.

## Extraction (floor channel, offline-verifiable)

Floor contacts are analytic (z = 0), so they are extractable from positions alone — no
mesh needed — which makes the metric verifiable end-to-end on a `demo_results` npz:
- contact points = the demo toe joints (`L_Toe`/`R_Toe`) mapped to robot foot links.
- `ref_contact[t,c]`  = `human_joints[t, toe_c, 2] < z_thresh` (source foot near floor).
- `robot_contact[t,c]`= robot foot-link world z (via FK) `< z_thresh`.
- `placement_dist[t,c]` = `|robot foot z|` (distance to the z=0 plane).
- `slip[t,c]` = `|| p_xy[t] - p_xy[t-1] ||` of the robot foot link (tangential drift).

`z_thresh` = 0.02 m (matches the existing eval contact threshold).

A helper `_floor_contacts(qpos, human_joints, foot_links, toe_idx)` on the evaluator (or
a small standalone function reusing FK) builds the four arrays; object contacts are a
follow-up that fills the same arrays from mesh distances and feeds the same function.

## Testing

- `tests/test_metrics_contacts.py` (pure, synthetic):
  - perfect agreement (`robot == ref`), zero placement, zero slip
    -> precision = recall = f1 = 1, place_err = 0, slip = 0.
  - one false positive + one false negative -> exact precision/recall/f1 by hand.
  - sustained contact with known per-frame slip -> `contact_slip_mean` equals the mean
    over sustained frames only (non-contact frames excluded).
- Integration: extend `tests/test_scoreboard_eval.py` (or a new test) to build the
  floor-contact arrays from a real `demo_results` robot_only npz and assert all contact
  keys are finite and in range (`0 <= f1 <= 1`, place_err >= 0).

## Out of scope (this sub-project)
- Object-surface contact extraction (mesh distance) — follow-up; the pure function and
  array contract already support it.
- Root/object pose sanity metrics — sub-project 3.
