# Per-frame Signal Export (v1) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Export one TEST-SOCP run's per-frame diagnostics to a PlotJuggler-readable wide CSV plus a JSON summary of global scalars.

**Architecture:** A new isolated package `HoloNew/evaluation/export/` with four pure components (`producers`, `flatten`, `csv_writer`, `summary`), a `collect.RunSignals` glue object that turns a `RetargetResult` into time + named channels, and a `export_signals.py` CLI that solves from config and writes both files. Producers emit flat `(T,)` channels keyed by hierarchical `/`-names; a `vec_channels` helper expands vector fields (`(T,3)`, `(T,N)`) into per-component channels. Adding a future signal = adding one entry to the `PRODUCERS` registry — the flattener and writer never change.

**Tech Stack:** Python, numpy (only), tyro (CLI, already used across the repo). Tests with pytest.

**Conventions (from project memory):**
- Run tests with `cd HoloNew && /home/gbesset/.holonew_deps/miniconda3/envs/holonew/bin/python -m pytest …`. Paths below assume cwd = the inner `HoloNew/HoloNew/` package dir.
- Commits must NOT include any `Co-Authored-By: Claude` / Claude trailer.

**Channel contract (used by every task):** a *channel* is a 1-D `np.ndarray` of shape `(T,)`. A *channels* mapping is `dict[str, np.ndarray]`, each value `(T,)`, keys hierarchical with `/` (e.g. `dynamics/com/x`). PlotJuggler renders the `/` as a signal tree.

---

### Task 1: Package skeleton + `flatten`

**Files:**
- Create: `evaluation/export/__init__.py`
- Create: `evaluation/export/flatten.py`
- Test: `tests/test_export_flatten.py`

- [ ] **Step 1: Write the failing test**

`tests/test_export_flatten.py`:
```python
import numpy as np
import pytest
from HoloNew.evaluation.export.flatten import to_columns


def test_orders_columns_with_time_first():
    time = np.array([0.0, 0.1, 0.2])
    channels = {"dynamics/com/x": np.array([1.0, 2.0, 3.0]),
                "diag/foot_slip": np.array([0.0, 0.5, 0.0])}
    header, table = to_columns(time, channels)
    assert header == ["time", "dynamics/com/x", "diag/foot_slip"]
    assert table.shape == (3, 3)
    np.testing.assert_allclose(table[:, 0], time)
    np.testing.assert_allclose(table[:, 1], [1.0, 2.0, 3.0])


def test_empty_channels_gives_time_only():
    time = np.array([0.0, 0.1])
    header, table = to_columns(time, {})
    assert header == ["time"]
    assert table.shape == (2, 1)


def test_rejects_wrong_length_or_2d():
    time = np.array([0.0, 0.1, 0.2])
    with pytest.raises(ValueError):
        to_columns(time, {"bad": np.array([1.0, 2.0])})
    with pytest.raises(ValueError):
        to_columns(time, {"bad2d": np.zeros((3, 2))})
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd HoloNew && /home/gbesset/.holonew_deps/miniconda3/envs/holonew/bin/python -m pytest tests/test_export_flatten.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'HoloNew.evaluation.export'`.

- [ ] **Step 3: Write minimal implementation**

`evaluation/export/__init__.py`:
```python
"""Per-frame run signal export (PlotJuggler-friendly CSV + summary JSON)."""
```

`evaluation/export/flatten.py`:
```python
"""Turn a channels mapping into a wide table: time column first, one column per channel."""
from __future__ import annotations

import numpy as np


def to_columns(time: np.ndarray, channels: dict[str, np.ndarray]):
    """(time, channels) -> (header, table).

    header: ["time", *channel names]. table: (T, 1 + len(channels)) float array.
    Every channel must be 1-D of length T == len(time); raises ValueError otherwise.
    """
    time = np.asarray(time, dtype=float).ravel()
    T = time.shape[0]
    names = list(channels.keys())
    cols = [time]
    for n in names:
        a = np.asarray(channels[n], dtype=float)
        if a.ndim != 1 or a.shape[0] != T:
            raise ValueError(f"channel {n!r} must be 1-D length {T}, got shape {a.shape}")
        cols.append(a)
    header = ["time"] + names
    table = np.column_stack(cols)
    return header, table
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd HoloNew && /home/gbesset/.holonew_deps/miniconda3/envs/holonew/bin/python -m pytest tests/test_export_flatten.py -q`
Expected: PASS (3 passed).

- [ ] **Step 5: Commit**

```bash
git add HoloNew/evaluation/export/__init__.py HoloNew/evaluation/export/flatten.py HoloNew/tests/test_export_flatten.py
git commit -m "feat(export): channels-to-wide-table flattener for signal export"
```

---

### Task 2: `csv_writer`

**Files:**
- Create: `evaluation/export/csv_writer.py`
- Test: `tests/test_export_csv_writer.py`

- [ ] **Step 1: Write the failing test**

`tests/test_export_csv_writer.py`:
```python
import numpy as np
from HoloNew.evaluation.export.csv_writer import write_csv


def test_writes_header_and_rows_roundtrip(tmp_path):
    header = ["time", "dynamics/com/x"]
    table = np.array([[0.0, 1.0], [0.1, 2.0], [0.2, 3.0]])
    path = tmp_path / "nested" / "run_signals.csv"
    write_csv(path, header, table)

    text = path.read_text().splitlines()
    assert text[0] == "time,dynamics/com/x"
    assert len(text) == 4  # header + 3 rows

    read = np.loadtxt(path, delimiter=",", skiprows=1)
    np.testing.assert_allclose(read, table)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd HoloNew && /home/gbesset/.holonew_deps/miniconda3/envs/holonew/bin/python -m pytest tests/test_export_csv_writer.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'HoloNew.evaluation.export.csv_writer'`.

- [ ] **Step 3: Write minimal implementation**

`evaluation/export/csv_writer.py`:
```python
"""Write a (header, table) pair as a CSV PlotJuggler reads natively (numpy only)."""
from __future__ import annotations

from pathlib import Path

import numpy as np


def write_csv(path, header: list[str], table: np.ndarray) -> Path:
    """Write CSV: first line = comma-joined header, then one row per frame."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(path, np.asarray(table, dtype=float), delimiter=",",
               header=",".join(header), comments="", fmt="%.9g")
    return path
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd HoloNew && /home/gbesset/.holonew_deps/miniconda3/envs/holonew/bin/python -m pytest tests/test_export_csv_writer.py -q`
Expected: PASS (1 passed).

- [ ] **Step 5: Commit**

```bash
git add HoloNew/evaluation/export/csv_writer.py HoloNew/tests/test_export_csv_writer.py
git commit -m "feat(export): dependency-free CSV writer for signal tables"
```

---

### Task 3: `summary`

**Files:**
- Create: `evaluation/export/summary.py`
- Test: `tests/test_export_summary.py`

- [ ] **Step 1: Write the failing test**

`tests/test_export_summary.py`:
```python
import json
import numpy as np
from HoloNew.evaluation.export.summary import reduce_channel, write_summary


def test_reduce_channel_scalars():
    r = reduce_channel(np.array([0.0, 3.0, 4.0]))
    assert r["mean"] == 7.0 / 3.0
    assert r["min"] == 0.0
    assert r["max"] == 4.0
    np.testing.assert_allclose(r["rms"], np.sqrt((0 + 9 + 16) / 3))


def test_reduce_channel_empty_is_zeros():
    r = reduce_channel(np.array([]))
    assert r == {"mean": 0.0, "rms": 0.0, "min": 0.0, "max": 0.0}


def test_write_summary_json(tmp_path):
    path = tmp_path / "run_summary.json"
    write_summary(path, {"diag/foot_slip": np.array([0.0, 1.0])})
    data = json.loads(path.read_text())
    assert data["diag/foot_slip"]["max"] == 1.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd HoloNew && /home/gbesset/.holonew_deps/miniconda3/envs/holonew/bin/python -m pytest tests/test_export_summary.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'HoloNew.evaluation.export.summary'`.

- [ ] **Step 3: Write minimal implementation**

`evaluation/export/summary.py`:
```python
"""Reduce each per-frame channel to global scalars and write a JSON summary.

v1 uses generic reductions (mean / rms / min / max). Later increments replace these
with the canonical 7-family scoreboard as the *_series metrics land.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def reduce_channel(a: np.ndarray) -> dict[str, float]:
    a = np.asarray(a, dtype=float).ravel()
    if a.size == 0:
        return {"mean": 0.0, "rms": 0.0, "min": 0.0, "max": 0.0}
    return {
        "mean": float(np.mean(a)),
        "rms": float(np.sqrt(np.mean(np.square(a)))),
        "min": float(np.min(a)),
        "max": float(np.max(a)),
    }


def write_summary(path, channels: dict[str, np.ndarray]) -> dict[str, dict[str, float]]:
    summary = {name: reduce_channel(arr) for name, arr in channels.items()}
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, indent=2))
    return summary
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd HoloNew && /home/gbesset/.holonew_deps/miniconda3/envs/holonew/bin/python -m pytest tests/test_export_summary.py -q`
Expected: PASS (3 passed).

- [ ] **Step 5: Commit**

```bash
git add HoloNew/evaluation/export/summary.py HoloNew/tests/test_export_summary.py
git commit -m "feat(export): global-scalar summary JSON for run signals"
```

---

### Task 4: `producers` registry + v1 producers

**Files:**
- Create: `evaluation/export/producers.py`
- Test: `tests/test_export_producers.py`

Each producer is `fn(result) -> dict[str, np.ndarray]`, reading `(T,…)` fields off a
`RetargetResult` and returning flat `(T,)` channels (empty dict when its source is
absent). `vec_channels` expands a `(T, K)` field into K named `(T,)` channels.

- [ ] **Step 1: Write the failing test**

`tests/test_export_producers.py`:
```python
from types import SimpleNamespace

import numpy as np
import pytest

from HoloNew.evaluation.export.producers import PRODUCERS, vec_channels, run_all


def _fake_result(**over):
    base = dict(qpos=np.zeros((4, 43)), com=None, com_ref=None,
                angular_momentum=None, angular_momentum_ref=None,
                foot_slip=None, human_flr_dist=None, human_obj_dist=None)
    base.update(over)
    return SimpleNamespace(**base)


def test_vec_channels_expands_columns():
    arr = np.arange(6.0).reshape(3, 2)
    ch = vec_channels("dynamics/com", arr, ("x", "y"))
    assert set(ch) == {"dynamics/com/x", "dynamics/com/y"}
    np.testing.assert_allclose(ch["dynamics/com/y"], arr[:, 1])


def test_com_producer_emits_xyz_and_ref():
    res = _fake_result(com=np.ones((4, 3)), com_ref=np.zeros((4, 3)))
    ch = run_all(res)
    for leaf in ("x", "y", "z"):
        assert f"dynamics/com/{leaf}" in ch
        assert f"dynamics/com_ref/{leaf}" in ch
    assert ch["dynamics/com/x"].shape == (4,)


def test_absent_sources_emit_nothing():
    ch = run_all(_fake_result())
    assert ch == {}


def test_dist_channels_named_positionally():
    res = _fake_result(human_flr_dist=np.zeros((4, 2)))
    ch = run_all(res)
    assert "diag/human_flr_dist/probe_000" in ch
    assert "diag/human_flr_dist/probe_001" in ch


def test_foot_slip_scalar_channel():
    res = _fake_result(foot_slip=np.array([0.0, 1.0, 2.0, 3.0]))
    ch = run_all(res)
    np.testing.assert_allclose(ch["diag/foot_slip"], [0.0, 1.0, 2.0, 3.0])


def test_registry_is_list_of_named_callables():
    assert PRODUCERS and all(callable(fn) for _, fn in PRODUCERS)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd HoloNew && /home/gbesset/.holonew_deps/miniconda3/envs/holonew/bin/python -m pytest tests/test_export_producers.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'HoloNew.evaluation.export.producers'`.

- [ ] **Step 3: Write minimal implementation**

`evaluation/export/producers.py`:
```python
"""Signal producers: read (T,...) fields off a RetargetResult, emit flat (T,) channels.

A producer is fn(result) -> dict[str, np.ndarray] (each value 1-D length T), returning
{} when its source field is absent. PRODUCERS is the ordered registry — the single
place to extend when adding a new signal. ``run_all`` merges every producer's output.
"""
from __future__ import annotations

import numpy as np

_XYZ = ("x", "y", "z")


def vec_channels(prefix: str, arr: np.ndarray, leaves) -> dict[str, np.ndarray]:
    """Expand a (T, K) field into {f"{prefix}/{leaf}": (T,)} for each leaf."""
    arr = np.asarray(arr)
    return {f"{prefix}/{leaf}": arr[:, i] for i, leaf in enumerate(leaves)}


def _probe_leaves(n: int) -> list[str]:
    return [f"probe_{i:03d}" for i in range(n)]


def _com(result) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    if getattr(result, "com", None) is not None:
        out |= vec_channels("dynamics/com", result.com, _XYZ)
    if getattr(result, "com_ref", None) is not None:
        out |= vec_channels("dynamics/com_ref", result.com_ref, _XYZ)
    return out


def _ang_momentum(result) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    if getattr(result, "angular_momentum", None) is not None:
        out |= vec_channels("dynamics/ang_momentum", result.angular_momentum, _XYZ)
    if getattr(result, "angular_momentum_ref", None) is not None:
        out |= vec_channels("dynamics/ang_momentum_ref", result.angular_momentum_ref, _XYZ)
    return out


def _foot_slip(result) -> dict[str, np.ndarray]:
    fs = getattr(result, "foot_slip", None)
    return {"diag/foot_slip": np.asarray(fs)} if fs is not None else {}


def _distances(result) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    flr = getattr(result, "human_flr_dist", None)
    if flr is not None:
        flr = np.asarray(flr)
        out |= vec_channels("diag/human_flr_dist", flr, _probe_leaves(flr.shape[1]))
    obj = getattr(result, "human_obj_dist", None)
    if obj is not None:
        obj = np.asarray(obj)
        out |= vec_channels("diag/human_obj_dist", obj, _probe_leaves(obj.shape[1]))
    return out


PRODUCERS = [
    ("com", _com),
    ("ang_momentum", _ang_momentum),
    ("foot_slip", _foot_slip),
    ("distances", _distances),
]


def run_all(result) -> dict[str, np.ndarray]:
    """Merge every producer's channels (later producers cannot overwrite earlier keys)."""
    channels: dict[str, np.ndarray] = {}
    for _name, fn in PRODUCERS:
        for cname, arr in (fn(result) or {}).items():
            channels[cname] = arr
    return channels
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd HoloNew && /home/gbesset/.holonew_deps/miniconda3/envs/holonew/bin/python -m pytest tests/test_export_producers.py -q`
Expected: PASS (6 passed).

- [ ] **Step 5: Commit**

```bash
git add HoloNew/evaluation/export/producers.py HoloNew/tests/test_export_producers.py
git commit -m "feat(export): producer registry + v1 RetargetResult diagnostic producers"
```

---

### Task 5: `collect.RunSignals`

**Files:**
- Create: `evaluation/export/collect.py`
- Test: `tests/test_export_collect.py`

- [ ] **Step 1: Write the failing test**

`tests/test_export_collect.py`:
```python
from types import SimpleNamespace

import numpy as np
import pytest

from HoloNew.evaluation.export.collect import RunSignals


def _result(T=5, **over):
    base = dict(qpos=np.zeros((T, 43)), com=None, com_ref=None,
                angular_momentum=None, angular_momentum_ref=None,
                foot_slip=None, human_flr_dist=None, human_obj_dist=None)
    base.update(over)
    return SimpleNamespace(**base)


def test_time_axis_from_fps():
    sig = RunSignals(_result(T=4), fps=10.0)
    np.testing.assert_allclose(sig.time, [0.0, 0.1, 0.2, 0.3])


def test_only_present_channels_appear():
    sig = RunSignals(_result(T=4, com=np.ones((4, 3))), fps=30.0)
    assert set(sig.channels) == {"dynamics/com/x", "dynamics/com/y", "dynamics/com/z"}


def test_no_diagnostics_gives_empty_channels():
    assert RunSignals(_result(), fps=30.0).channels == {}


def test_mismatched_leading_axis_raises():
    # foot_slip length 3 but qpos T=5
    with pytest.raises(ValueError):
        RunSignals(_result(T=5, foot_slip=np.zeros(3)), fps=30.0)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd HoloNew && /home/gbesset/.holonew_deps/miniconda3/envs/holonew/bin/python -m pytest tests/test_export_collect.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'HoloNew.evaluation.export.collect'`.

- [ ] **Step 3: Write minimal implementation**

`evaluation/export/collect.py`:
```python
"""RunSignals: turn a RetargetResult into a frame-aligned (time, channels) bundle.

The only component that knows RetargetResult. Runs the producer registry, enforces
that every emitted channel is frame-aligned to qpos (length T), and builds the time
axis from fps.
"""
from __future__ import annotations

import numpy as np

from .producers import run_all


class RunSignals:
    def __init__(self, result, fps: float = 30.0):
        self.T = int(np.asarray(result.qpos).shape[0])
        self.time = np.arange(self.T, dtype=float) / float(fps)
        self.channels: dict[str, np.ndarray] = {}
        for name, arr in run_all(result).items():
            arr = np.asarray(arr)
            if arr.shape[0] != self.T:
                raise ValueError(
                    f"channel {name!r}: leading axis {arr.shape[0]} != T={self.T}")
            self.channels[name] = arr
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd HoloNew && /home/gbesset/.holonew_deps/miniconda3/envs/holonew/bin/python -m pytest tests/test_export_collect.py -q`
Expected: PASS (4 passed).

- [ ] **Step 5: Commit**

```bash
git add HoloNew/evaluation/export/collect.py HoloNew/tests/test_export_collect.py
git commit -m "feat(export): RunSignals collector (RetargetResult -> time + channels)"
```

---

### Task 6: `export_signals` CLI + end-to-end test

**Files:**
- Create: `evaluation/export/export_signals.py`
- Test: `tests/test_export_cli.py`

The CLI solves from config; the heavy solve needs the full env. The end-to-end test
instead exercises the **pure write path** (`to_columns` → `write_csv` + `write_summary`
on a synthetic `RunSignals`), so it runs without a MuJoCo/cvxpy solve. A manual smoke
command is documented for the real solve.

- [ ] **Step 1: Write the failing test**

`tests/test_export_cli.py`:
```python
from types import SimpleNamespace

import numpy as np

from HoloNew.evaluation.export.collect import RunSignals
from HoloNew.evaluation.export.export_signals import write_run


def _result(T=3, **over):
    base = dict(qpos=np.zeros((T, 43)), com=np.ones((T, 3)), com_ref=None,
                angular_momentum=None, angular_momentum_ref=None,
                foot_slip=np.arange(float(T)), human_flr_dist=None, human_obj_dist=None)
    base.update(over)
    return SimpleNamespace(**base)


def test_write_run_emits_both_files(tmp_path):
    sig = RunSignals(_result(T=3), fps=30.0)
    csv_path, json_path = write_run(sig, tmp_path, "demo_task")

    assert csv_path == tmp_path / "demo_task_signals.csv"
    assert json_path == tmp_path / "demo_task_summary.json"

    lines = csv_path.read_text().splitlines()
    assert lines[0].split(",")[0] == "time"
    assert len(lines) == 1 + 3  # header + T rows
    assert "diag/foot_slip" in lines[0]

    import json
    summary = json.loads(json_path.read_text())
    assert "diag/foot_slip" in summary
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd HoloNew && /home/gbesset/.holonew_deps/miniconda3/envs/holonew/bin/python -m pytest tests/test_export_cli.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'HoloNew.evaluation.export.export_signals'`.

- [ ] **Step 3: Write minimal implementation**

`evaluation/export/export_signals.py`:
```python
"""CLI: solve one TEST-SOCP run from config and export per-frame signals.

  python -m HoloNew.evaluation.export.export_signals --task-name <name> --out <dir>

Writes <out>/<task>_signals.csv (PlotJuggler: load as CSV, index column = "time")
and <out>/<task>_summary.json (global scalars).
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import tyro

from HoloNew.evaluation.export.collect import RunSignals
from HoloNew.evaluation.export.csv_writer import write_csv
from HoloNew.evaluation.export.flatten import to_columns
from HoloNew.evaluation.export.summary import write_summary


def write_run(sig: RunSignals, out_dir, task_name: str):
    """Write the CSV + summary for a collected RunSignals. Returns (csv_path, json_path)."""
    out_dir = Path(out_dir)
    csv_path = out_dir / f"{task_name}_signals.csv"
    json_path = out_dir / f"{task_name}_summary.json"
    header, table = to_columns(sig.time, sig.channels)
    write_csv(csv_path, header, table)
    write_summary(json_path, sig.channels)
    return csv_path, json_path


@dataclass
class Args:
    task_name: str = "sub3_largebox_003"
    task_type: str = "object_interaction"
    data_format: str | None = None
    out: Path = Path("export_out")
    max_frames: int | None = None


def main(cfg: Args) -> None:
    from HoloNew.config_types.retargeting import RetargetingConfig
    from HoloNew.src.test_socp.test_socp import TestSocpRetargeter

    rc = RetargetingConfig(task_type=cfg.task_type, task_name=cfg.task_name,
                           data_format=cfg.data_format)
    rt = TestSocpRetargeter.from_config(rc)
    rt.collect_diagnostics = True
    res = rt.retarget(max_frames=cfg.max_frames)

    dt = getattr(rt, "_dt", None)
    fps = (1.0 / float(dt)) if dt else 30.0
    sig = RunSignals(res, fps=fps)
    if not sig.channels:
        print("WARNING: no diagnostics collected (collect_diagnostics off or empty); "
              "CSV will contain only the 'time' column.", file=sys.stderr)

    csv_path, json_path = write_run(sig, cfg.out, cfg.task_name)
    print(csv_path)
    print(json_path)


if __name__ == "__main__":
    main(tyro.cli(Args))
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd HoloNew && /home/gbesset/.holonew_deps/miniconda3/envs/holonew/bin/python -m pytest tests/test_export_cli.py -q`
Expected: PASS (1 passed).

- [ ] **Step 5: Run the full export suite**

Run: `cd HoloNew && /home/gbesset/.holonew_deps/miniconda3/envs/holonew/bin/python -m pytest tests/test_export_*.py -q`
Expected: PASS (all export tests green).

- [ ] **Step 6: Commit**

```bash
git add HoloNew/evaluation/export/export_signals.py HoloNew/tests/test_export_cli.py
git commit -m "feat(export): export_signals CLI (solve-from-config -> CSV + summary)"
```

- [ ] **Step 7: Manual smoke (real solve — optional, needs the demo data + env)**

Run:
```bash
cd HoloNew && /home/gbesset/.holonew_deps/miniconda3/envs/holonew/bin/python \
  -m HoloNew.evaluation.export.export_signals --task-name sub3_largebox_003 \
  --task-type object_interaction --out export_out --max-frames 20
```
Expected: prints `export_out/sub3_largebox_003_signals.csv` and `…_summary.json`; the
CSV has a `time` column plus `dynamics/com/*`, `diag/foot_slip`, and `diag/human_*_dist/*`
columns; open it in PlotJuggler (DataLoad CSV, index column = `time`) to confirm the
`/`-tree renders.

---

## Notes for the implementer

- **`__init__.py` exports:** keep `evaluation/export/__init__.py` minimal (docstring
  only); modules are imported by full path, matching `evaluation/metrics/` usage.
- **No upstream changes in v1.** `solver/cost` and the `*_series` metric refactor are
  later increments (see the design spec). Do not touch `test_socp.py` /
  `RetargetResult` in this plan.
- **Why `run_all` ignores key collisions safely:** v1 producers emit disjoint prefixes
  (`dynamics/…`, `diag/…`), so no overwrite occurs; the merge order in `PRODUCERS` is
  the column order in the CSV.
