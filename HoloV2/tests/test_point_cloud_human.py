"""Integration test for the human cloud bake. Skips when HODome data / the SMPL-X model are absent.

Validates the sparse-skinning bake: posing the cloud (mesh-free) must track the true posed SMPL
surface (the full forward) within a few mm, and the builder must be deterministic + cache
round-trippable. The cloud is sampled at the NEUTRAL correspondence's sampling but skinned on the
SUBJECT's rest mesh — exactly the cross-subject reuse the binding relies on."""
from pathlib import Path

import numpy as np
import pytest

from holov2.contracts import CloudConfig, RobotSpec, SceneSpec
from holov2.prepare.point_cloud import HumanCloudBuilder, build_human_cloud
from holov2.prepare.point_cloud.correspondence import load_correspondence
from holov2.targets.interaction import pose_cloud

_DATA = Path("/home/vboxuser/Documents/wbt_rl/data/00_raw_datasets")
_HODOME = _DATA / "HODome"
_SMPLX = _DATA / "models" / "models_smplx_v1_1" / "models" / "smplx"
_CORR = Path(__file__).resolve().parent.parent / "cache" / "correspondence" / "corr_neutral.npz"


def _pick() -> Path | None:
    sm, ob = _HODOME / "smplx", _HODOME / "object"
    if not (sm.is_dir() and ob.is_dir() and _SMPLX.is_dir()):
        return None
    shared = {p.stem for p in sm.glob("*.npz")} & {p.stem for p in ob.glob("*.npz")}
    return sm / f"{sorted(shared)[0]}.npz" if shared else None


_SEQ = _pick()


def _posed_surface_ref(body, params, t, tri_idx, bary):
    """Reference: the cached samples carried onto the FULL posed SMPL surface (full forward)."""
    verts = body.posed_vertices(params, t)                           # (V,3) world Z-up, full LBS
    tri_v = body.faces[tri_idx]                                      # (N,3)
    return np.einsum("nij,ni->nj", verts[tri_v], np.asarray(bary, np.float64))


@pytest.mark.skipif(_SEQ is None, reason="HODome data / SMPL-X model not available")
def test_human_cloud_parity_determinism_roundtrip(tmp_path):
    from holov2.prepare.load import load
    from holov2.prepare.load.smpl import build_body_model

    spec = SceneSpec(
        dataset="hodome", motion_path=_SEQ,
        robot=RobotSpec(name="g1", urdf_path=Path("g1.urdf"), link_names=("a",), dof=29, height=1.3),
        smpl_model_dir=_SMPLX,
    )
    raw = load(spec)
    params = raw.smpl_params
    body = build_body_model(params, _SMPLX)
    _, sampling = load_correspondence(_CORR)

    cloud = build_human_cloud(body, sampling, CloudConfig())
    assert cloud.n_points == sampling.n_points
    assert cloud.parts.shape == (sampling.n_points, CloudConfig().k_influences)
    assert np.allclose(cloud.weights.sum(axis=1), 1.0, atol=1e-5)    # partition of unity
    assert int(cloud.parts.max()) < body.n_bones
    assert cloud.sampling_id == sampling.sampling_id                  # binds to the correspondence

    # parity of the mesh-free posed cloud vs the true posed surface: few-mm median, cm-scale tail.
    for t in (0, raw.n_frames // 2, raw.n_frames - 1):
        posed = pose_cloud(cloud, *body.bone_transforms(params, t))   # (N,3)
        ref = _posed_surface_ref(body, params, t, sampling.tri_idx, sampling.bary)
        err = np.linalg.norm(posed - ref, axis=1)
        assert np.median(err) < 0.01, f"frame {t}: median {np.median(err):.4f} m"
        assert np.percentile(err, 95) < 0.03, f"frame {t}: p95 {np.percentile(err, 95):.4f} m"

    # determinism: same inputs -> identical cloud.
    builder = HumanCloudBuilder()
    again = builder.build(CloudConfig(), body, sampling)
    assert np.array_equal(cloud.parts, again.parts)
    assert np.array_equal(cloud.weights, again.weights)
    assert np.array_equal(cloud.offsets, again.offsets)

    # cache round-trip: save -> load == build.
    path = tmp_path / "human_cloud.npz"
    builder.save(cloud, path)
    loaded = builder.load(path)
    assert np.array_equal(cloud.parts, loaded.parts)
    assert np.array_equal(cloud.weights, loaded.weights)
    assert np.array_equal(cloud.offsets, loaded.offsets)
    assert cloud.sampling_id == loaded.sampling_id
