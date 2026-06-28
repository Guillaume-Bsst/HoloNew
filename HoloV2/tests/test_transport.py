"""transport unit test (synthetic, torch-free): the human MultiChannelField gather onto the M robot
correspondence points (by ``smpl_idx``), the (C,M) / (C,M,3) shapes, the carried channel names, and
that EVERY field is gathered as-is (transport is a pure, frame-agnostic gather — no scaling here; the
human->robot scale is a scene-placement concern handled in the runner)."""
import numpy as np

from src.prepare.contracts import CorrespondenceTable
from src.targets.contracts import MultiChannelField
from src.targets.interaction import transport


def _human_field() -> MultiChannelField:
    """C=2 channels over P_human=5 points, with recognizable per-(channel,point) values so a wrong
    gather is obvious. distance[c,p] = 10*c + p ; witness/direction carry the same tag in xyz."""
    C, P = 2, 5
    cc, pp = np.meshgrid(np.arange(C), np.arange(P), indexing="ij")   # (C, P) each
    tag = (10.0 * cc + pp)                                            # (C, P) distinct per cell
    distance = tag.astype(np.float64)                                # (C, P)
    witness = np.stack([tag, tag + 0.1, tag + 0.2], axis=-1)         # (C, P, 3)
    # distinct, non-unit raw directions; transport must gather them verbatim (no renormalising).
    direction = np.stack([tag, -tag, 2.0 * tag], axis=-1)            # (C, P, 3)
    active = ((cc + pp) % 2 == 0)                                    # (C, P) bool checkerboard
    return MultiChannelField(distance=distance, direction=direction, witness=witness,
                             active=active, channels=("ground", "obj0"))


def _correspondence(smpl_idx) -> CorrespondenceTable:
    m = len(smpl_idx)
    return CorrespondenceTable(
        smpl_idx=np.asarray(smpl_idx),
        link_idx=np.zeros(m, dtype=int),
        offset_local=np.zeros((m, 3)),
        link_names=("root",),
        smpl_sampling_id="test")


def test_transport_gathers_rows_and_shapes():
    hf = _human_field()
    smpl_idx = [3, 0, 4]            # M=3, picks human points 3, 0, 4 (out of order)
    out = transport(hf, _correspondence(smpl_idx))

    # shapes: (C, M) and (C, M, 3)
    assert out.distance.shape == (2, 3)
    assert out.direction.shape == (2, 3, 3)
    assert out.witness.shape == (2, 3, 3)
    assert out.active.shape == (2, 3)
    assert out.channels == ("ground", "obj0")    # names carried over

    # every field gathered verbatim (pure gather, no scaling, no renormalising).
    assert np.allclose(out.distance, hf.distance[:, smpl_idx])
    assert np.allclose(out.witness, hf.witness[:, smpl_idx])
    assert np.allclose(out.direction, hf.direction[:, smpl_idx])
    assert np.array_equal(out.active, hf.active[:, smpl_idx])
    assert out.active.dtype == bool
    for j, p in enumerate(smpl_idx):
        for c in range(2):
            assert out.distance[c, j] == 10.0 * c + p


def test_transport_pure_no_input_mutation_and_frozen_output():
    hf = _human_field()
    dist_before = hf.distance.copy()
    wit_before = hf.witness.copy()
    out = transport(hf, _correspondence([1, 2]))
    # inputs untouched
    assert np.array_equal(hf.distance, dist_before)
    assert np.array_equal(hf.witness, wit_before)
    # output buffers are read-only (frozen by convention)
    assert not out.distance.flags.writeable
    assert not out.witness.flags.writeable
    assert not out.direction.flags.writeable
    assert not out.active.flags.writeable


def test_transport_handles_repeated_indices():
    hf = _human_field()
    smpl_idx = [4, 4, 0, 2]        # repeats allowed: two robot points map to the same human point
    out = transport(hf, _correspondence(smpl_idx))
    assert np.allclose(out.distance, hf.distance[:, smpl_idx])
    assert np.allclose(out.witness, hf.witness[:, smpl_idx])
    # the two points mapped to human idx 4 are identical
    assert np.allclose(out.distance[:, 0], out.distance[:, 1])
