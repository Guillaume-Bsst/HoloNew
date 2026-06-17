"""CLI: solve one TEST-SOCP run from config and export per-frame signals.

  python -m HoloNew.evaluation.export.export_signals --task-name <name> --out <dir>

Writes <out>/<task>_signals.csv (PlotJuggler: load as CSV, index column = "time")
and <out>/<task>_summary.json (global scalars).
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import tyro

from HoloNew.evaluation.export.collect import RunSignals
from HoloNew.evaluation.export.context import SignalContext
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


def _limited_joint_ranges(model, dof: int):
    """(cols, lower, upper, names) of limited actuated joints, aligned to qpos[:, 7:7+dof].

    Mirrors RetargetingEvaluator._limited_joint_ranges so the effort margins line up with
    the dof joint block; also returns each limited joint's name for the export labels.
    """
    import mujoco

    cols, lo, hi, names = [], [], [], []
    for j in range(model.njnt):
        if model.jnt_type[j] not in (mujoco.mjtJoint.mjJNT_HINGE, mujoco.mjtJoint.mjJNT_SLIDE):
            continue
        if not model.jnt_limited[j]:
            continue
        col = int(model.jnt_qposadr[j]) - 7
        if 0 <= col < dof:
            cols.append(col)
            lo.append(float(model.jnt_range[j, 0]))
            hi.append(float(model.jnt_range[j, 1]))
            names.append(mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j) or f"joint_{col:03d}")
    return np.array(cols, dtype=int), np.array(lo), np.array(hi), names


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
    dof = getattr(getattr(rt, "task_constants", None), "ROBOT_DOF", None)
    dof = int(dof) if dof is not None else None
    ctx = SignalContext(dt=(1.0 / fps), dof=dof)
    model = getattr(rt, "robot_model", None)
    if dof is not None and model is not None:
        cols, lo, hi, names = _limited_joint_ranges(model, dof)
        if cols.size:
            ctx.joint_limit_cols, ctx.joint_limit_lower = cols, lo
            ctx.joint_limit_upper, ctx.joint_limit_names = hi, names

    # Reference/FK families (tracking + per-link orientation). Degrade gracefully if the
    # reference machinery is unavailable for this run rather than failing the export.
    try:
        from HoloNew.evaluation.reference_context import ReferenceContext
        from HoloNew.evaluation.export.reference_signals import tracking_channels
        ref_ctx = ReferenceContext.from_rt(rt)
        ctx.extra_channels = tracking_channels(ref_ctx, res.qpos)
    except Exception as exc:  # noqa: BLE001 - tracking is optional, never crash the export
        print(f"WARNING: tracking channels unavailable ({exc}); skipping.", file=sys.stderr)

    sig = RunSignals(res, fps=fps, ctx=ctx)
    if not sig.channels:
        print("WARNING: no diagnostics collected (collect_diagnostics off or empty); "
              "CSV will contain only the 'time' column.", file=sys.stderr)

    csv_path, json_path = write_run(sig, cfg.out, cfg.task_name)
    print(csv_path)
    print(json_path)


if __name__ == "__main__":
    main(tyro.cli(Args))
