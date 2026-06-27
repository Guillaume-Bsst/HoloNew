"""Adapts the cached correspondence (``corr_neutral.npz``, shipped in ``cache/``) into the V2
contract and checks the cloud<->correspondence binding the runner relies on."""
from pathlib import Path

import numpy as np

from src.prepare.point_cloud import sampling_id
from src.prepare.point_cloud.correspondence import load_correspondence

_CORR = Path(__file__).resolve().parent.parent / "cache" / "correspondence" / "corr_neutral.npz"


def test_load_correspondence_schema_and_binding():
    table, sampling = load_correspondence(_CORR)
    m = table.n_points
    assert table.smpl_idx.shape == (m,)
    assert table.link_idx.shape == (m,)
    assert table.offset_local.shape == (m, 3)
    assert len(table.link_names) == 52
    assert sampling.bary.shape == (sampling.n_points, 3)

    # binding guard (what the runner asserts): smpl_idx indexes the sampling, ids agree.
    assert int(table.smpl_idx.max()) < sampling.n_points
    assert table.smpl_sampling_id == sampling.sampling_id
    # sampling_id is a pure function of the sampling (reproducible).
    assert sampling_id(sampling.tri_idx, sampling.bary) == sampling.sampling_id
