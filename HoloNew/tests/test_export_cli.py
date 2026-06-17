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
