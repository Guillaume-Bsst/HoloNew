# Per-frame signal export for PlotJuggler — design

**Date:** 2026-06-17
**Status:** Design — pending implementation plan
**Scope:** `modules/01_retargeting/HoloNew` (the HoloNew package)
**Builds on:** the TEST-SOCP retargeter (`src/test_socp/test_socp.py`), its
`RetargetResult` diagnostics (`src/retarget_result.py`), and the existing scoreboard
metrics (`evaluation/metrics/`).

## Goal

Give the user a **simple, frame-by-frame and global view of how a single retargeting
run behaved**, by exporting per-frame signals to a CSV that
[PlotJuggler](https://plotjuggler.io/) reads natively, plus a JSON sidecar of the
reduced (global) scalars.

A "run" is **one TEST-SOCP solve** of one motion clip. The rich per-frame story lives
in `RetargetResult`; today every metric collapses its per-frame array to a single mean
via the scoreboard, throwing the temporal profile away. This work exposes that profile.

This spec covers **v1 only**: a deliberately minimal, fully-isolated export package
fed by signals that **already exist** in `RetargetResult`. It establishes the complete
pipeline (collect → flatten → write) so that later increments add signals by
registering one more producer — never by touching the flattener or writer.

## Decisions (from brainstorming)

1. **Scope = one TEST-SOCP solve**, not the `eval_retargeting` batch. The batch is a
   separate aggregate world; this is the per-run deep view.
2. **Format = one wide CSV per run** + a JSON summary sidecar. CSV is read natively by
   PlotJuggler (DataLoad CSV, with a `time` column), needs zero extra dependency, and
   replays offline. MCAP and live UDP streaming were rejected for v1 (dependency /
   not-replayable).
3. **Global view = JSON sidecar**, not constant series injected into the CSV. The
   summary holds the reduced scalars; PlotJuggler computes its own stats over the
   series. Keeps the CSV clean.
4. **Start simple, grow by increments.** v1 exports only the `(T,…)` arrays already
   collected by `RetargetResult._fill_diagnostics` — **no metric refactor, no upstream
   change to the solver**. The architecture is the final one; it is just fed few
   signals.
5. **Producer registry is the extension point.** Adding a signal = adding one registry
   entry. The flattener and writer are signal-agnostic.
6. **CLI = solve-from-config only** for v1. Loading an existing `.npz` is a documented
   future extension (it cannot recover the per-frame solver cost without a live solve).

## Signals exported in v1

All are already produced by `RetargetResult` when `collect_diagnostics=True`; the
export performs **no new computation**.

| Channel (CSV column prefix)        | Source field (`RetargetResult`)            | Shape   |
|------------------------------------|--------------------------------------------|---------|
| `dynamics/com/{x,y,z}`             | `com`                                      | `(T,3)` |
| `dynamics/com_ref/{x,y,z}`         | `com_ref`                                  | `(T,3)` |
| `dynamics/ang_momentum/{x,y,z}`    | `angular_momentum`                         | `(T,3)` |
| `dynamics/ang_momentum_ref/{x,y,z}`| `angular_momentum_ref`                     | `(T,3)` |
| `diag/foot_slip`                   | `foot_slip`                                | `(T,)`  |
| `diag/human_flr_dist/<probe>`      | `human_flr_dist`                           | `(T,N)` |
| `diag/human_obj_dist/<probe>`      | `human_obj_dist`                           | `(T,N)` |

Each channel is **optional**: a producer that finds its source field `None` (e.g.
`human_obj_dist` on a floor-only run, or `com_ref` when diagnostics targets are absent)
emits nothing. The CSV contains exactly the columns whose sources were present.

`<probe>` indices come from the probe-point order in `RetargetResult.human_probe_pts`;
v1 names them positionally (`probe_000`, `probe_001`, …). A later increment can map
them to body-part labels.

## Architecture

```
config ─► TestSocpRetargeter.from_config
            └─► retarget_motion(collect_diagnostics=True)
                  └─► RetargetResult  (T-aligned per-frame diagnostics)
                          │
                          ▼
                  collect.RunSignals(result, fps)
                   ── runs the registered producers, drops absent sources
                   ── returns: time (T,) + {channel_name: (T,) | (T,K)}
                          │
                          ▼
                  flatten.to_columns(signals)
                   ── (T,K) → K columns "prefix/<name>" ("/" => PlotJuggler tree)
                   ── "time" is column 0
                          │
            ┌─────────────┴──────────────┐
            ▼                             ▼
   csv_writer.write_csv          summary.write_summary
   <out>/<task>_signals.csv      <out>/<task>_summary.json
```

### Components

**`evaluation/export/producers.py` (pure).**
A producer is `(name, fn)` where `fn(result, ctx) -> dict[str, np.ndarray] | None`.
Each returns named channels of shape `(T,)` or `(T,K)`, or `None`/`{}` when its source
field is absent. `PRODUCERS` is the ordered registry (the single place to extend).
v1 ships the producers for the table above. `ctx` is a small namespace carrying naming
hints (joint names, probe labels) so later producers stay pure.

**`evaluation/export/collect.py`.**
`RunSignals(result, fps)`: iterates `PRODUCERS`, merges their channel dicts, asserts
every channel's leading axis equals `T = result.qpos.shape[0]`, builds
`time = arange(T) / fps`. Returns `(time, channels)`. The only component that knows
`RetargetResult`.

**`evaluation/export/flatten.py` (pure).**
`to_columns(time, channels) -> (header: list[str], table: np.ndarray (T, 1+Σwidth))`.
`(T,)` → one column named exactly the channel name; `(T,K)` → K columns
`"<channel>/<leaf>"`, leaf from the channel's declared names (e.g. `x/y/z`,
`probe_000…`). Column names must be unique; `time` is first.

**`evaluation/export/csv_writer.py` (pure).**
`write_csv(path, header, table)` — header row + rows. No dependency beyond numpy.

**`evaluation/export/summary.py` (pure).**
`write_summary(path, channels)` — reduces each channel to simple global scalars
(`mean`, `rms`, `min`, `max`) and writes JSON. This is the v1 "global view"; it becomes
the real 7-family scoreboard once the `*_series` refactor lands in a later increment.

**`evaluation/export/export_signals.py` (CLI).**
`python -m HoloNew.evaluation.export_signals --task <name> --task-type <...>
--data-format <...> --out <dir>`. Builds the retargeter via
`TestSocpRetargeter.from_config`, sets `collect_diagnostics=True`, solves, collects,
writes `<out>/<task>_signals.csv` and `<out>/<task>_summary.json`. Prints the two
paths.

## Data flow / contracts

- **Time alignment:** every channel is frame-aligned to `result.qpos` (length `T`).
  The collector enforces this; a producer returning a wrong leading axis is a bug, not
  a silent truncation.
- **Absence is normal:** object-channel and reference-target signals are `None` on runs
  that don't have them. Producers skip; no error, no empty columns.
- **PlotJuggler ingestion:** load `<task>_signals.csv`, pick `time` as the index
  column. The `/`-separated names render as a collapsible signal tree.

## Error handling

- Missing `collect_diagnostics` (all sources `None`) → CSV with only the `time`
  column and a clear stderr warning ("no diagnostics collected; run with
  collect_diagnostics=True"). Not a crash.
- `fps` absent → default 30.0 (matches `RetargetResult`/eval defaults), warn.
- Output dir created if missing.

## Testing

- **flatten (pure):** `(T,)` and `(T,3)` channels → correct column count and unique,
  `/`-named headers; `time` first and monotonically increasing.
- **collect:** synthetic `RetargetResult` with a subset of fields populated → only the
  present channels appear; a deliberately mis-shaped field raises.
- **absence:** a `RetargetResult` with all diagnostic fields `None` → `time`-only CSV +
  warning, no exception.
- **csv round-trip:** write then read back (numpy/`csv`) → shape and header match.
- **summary:** known channel → expected `mean`/`rms`/`min`/`max`.
- **CLI smoke** (if a golden clip + env are available; otherwise skipped): run on the
  demo task, assert both files exist and the CSV has > 1 column and `T` rows.

## Increment roadmap (post-v1)

Each step is "register one more producer" unless noted.

- **v2 — `solver/cost` `(T,)`:** small upstream change — the solve loop already computes
  `prob.value` per frame but discards it (`RetargetResult(cost=0.0)`). Record it into a
  new `per_frame_cost (T,)` field, then add its producer. Highest-value untapped trace.
- **v3+ — `*_series` refactor of the 7 scoreboard families, one family per increment**
  (tracking, smoothness, effort, dynamics, style, contacts, roots). Each metric splits
  into a per-frame `*_series` + a thin reducer, so `compute_*` becomes
  `reduce(*_series(...))` (single source of truth; the scoreboard scalar is the mean of
  the exported series — a parity test pins this). As families land, `summary.py` swaps
  its generic reductions for the canonical scoreboard.
- **later:** per-frame penetration depth (MuJoCo collision, expensive), per-term cost
  breakdown, derived balance signals (CoM velocity/height, base tilt vs gravity),
  probe-point → body-part labels, and an optional `.npz`-load CLI mode.

## New layout

```
HoloNew/evaluation/export/
  __init__.py
  producers.py       # registry + v1 producers (pure)
  collect.py         # RunSignals: RetargetResult -> (time, channels)
  flatten.py         # channels -> wide table (pure)
  csv_writer.py      # table -> CSV (pure)
  summary.py         # channels -> summary JSON (pure)
  export_signals.py  # CLI: solve-from-config -> CSV + summary
```

Nothing outside this package is modified in v1.
