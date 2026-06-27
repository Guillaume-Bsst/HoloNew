"""Integration test for the OT correspondence rebuild. Skips when the SMPL-X model / G1 URDF are
absent. Validates anatomical consistency (every robot point's segment == its human sample's segment)
and determinism (rebuild twice -> identical, so the cache is sound)."""
from pathlib import Path

import numpy as np
import pytest

from src.prepare.contracts import RobotSpec
from src.prepare.config import PrepareConfig
from src.prepare.point_cloud.correspondence import build_correspondence
from src.prepare.point_cloud.correspondence.segments import (
    link_to_segment, point_segments, seg_index)

_DATA = Path("/home/vboxuser/Documents/wbt_rl/data/00_raw_datasets")
_SMPLX = _DATA / "models" / "models_smplx_v1_1" / "models" / "smplx"
_URDF = Path(__file__).resolve().parent.parent / "models" / "g1" / "g1_29dof.urdf"


@pytest.mark.skipif(not (_SMPLX.is_dir() and _URDF.exists()), reason="SMPL-X model / G1 URDF absent")
def test_rebuild_segment_consistency_and_determinism():
    from src.prepare.load.smpl import rest_body_model

    spec = RobotSpec(name="g1", urdf_path=_URDF, link_names=(), dof=29, height=1.3)
    body = rest_body_model(np.zeros(10, np.float32), "neutral", _SMPLX)
    table, sampling = build_correspondence(PrepareConfig(), body, spec)

    assert table.n_points > 1000 and len(table.link_names) > 20
    assert int(table.smpl_idx.max()) < sampling.n_points
    assert table.smpl_sampling_id == sampling.sampling_id           # cloud<->corr binding

    # anatomical consistency: each robot point's segment == its driving human sample's segment.
    human_seg = point_segments(body.lbs_weights, body.faces, sampling.tri_idx, sampling.bary)
    robot_seg = np.array([seg_index(link_to_segment(table.link_names[int(li)]))
                          for li in table.link_idx])
    assert (robot_seg == human_seg[table.smpl_idx]).all()

    # determinism (cache soundness): a rebuild must reproduce the asset exactly.
    table2, sampling2 = build_correspondence(PrepareConfig(), body, spec)
    assert np.array_equal(table.smpl_idx, table2.smpl_idx)
    assert np.array_equal(table.link_idx, table2.link_idx)
    assert np.allclose(table.offset_local, table2.offset_local)
    assert sampling.sampling_id == sampling2.sampling_id
