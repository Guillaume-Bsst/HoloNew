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
