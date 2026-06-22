"""view_stages must persist each GMR/TEST solve as a demo_results .npz in the same
format robot_retarget/holosoma write (qpos, human_joints, fps, cost), so the viewer
produces a reusable result and not just an on-screen render."""
import numpy as np

from HoloNew.examples.view_stages import save_view_result


def test_save_view_result_default_path_and_keys(tmp_path):
    qpos = np.zeros((5, 36), np.float32)
    human_joints = np.zeros((5, 22, 3), np.float32)
    dest = save_view_result(
        base_dir=tmp_path, robot="g1", task_type="robot_only", dataset="sfu",
        task_name="0008_ChaCha001_stageii", method="test_socp",
        qpos=qpos, human_joints=human_joints, cost=1.25)

    assert dest == tmp_path / "g1" / "robot_only" / "sfu" / "0008_ChaCha001_stageii_test_socp.npz"
    assert dest.exists()
    with np.load(dest) as d:
        assert {"qpos", "human_joints", "fps", "cost"}.issubset(set(d.files))
        assert d["qpos"].shape == (5, 36)
        assert d["human_joints"].shape == (5, 22, 3)
        assert int(d["fps"]) == 30
        assert float(d["cost"]) == 1.25


def test_save_view_result_explicit_save_dir_overrides(tmp_path):
    # An explicit --save-dir is used verbatim (mirrors robot_retarget's save_dir).
    out = tmp_path / "custom"
    dest = save_view_result(
        base_dir=tmp_path, robot="g1", task_type="robot_only", dataset="sfu",
        task_name="clip", method="gmr_socp", qpos=np.zeros((2, 36), np.float32),
        human_joints=np.zeros((2, 22, 3), np.float32), cost=0.0, save_dir=out)
    assert dest == out / "clip_gmr_socp.npz"
    assert dest.exists()
