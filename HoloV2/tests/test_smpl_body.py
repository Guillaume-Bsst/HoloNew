"""Integration test for the SMPL-X BodyModel. Skips when HODome data / the SMPL-X model are
absent. Validates that the pure-FK bone transforms reproduce a real SMPL-X forward's joints."""
from pathlib import Path

import numpy as np
import pytest

from src.prepare.contracts import RobotSpec, SceneSpec

_DATA = Path("/home/vboxuser/Documents/wbt_rl/data/00_raw_datasets")
_HODOME = _DATA / "HODome"
_SMPLX = _DATA / "models" / "models_smplx_v1_1" / "models" / "smplx"
_YUP_TO_ZUP = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]])


def _pick() -> Path | None:
    sm, ob = _HODOME / "smplx", _HODOME / "object"
    if not (sm.is_dir() and ob.is_dir() and _SMPLX.is_dir()):
        return None
    shared = {p.stem for p in sm.glob("*.npz")} & {p.stem for p in ob.glob("*.npz")}
    return sm / f"{sorted(shared)[0]}.npz" if shared else None


_SEQ = _pick()


def _forward_joints_zup(params, t):
    """Reference: a real SMPL-X forward at frame t -> 22 body joints in Z-up."""
    import smplx
    import torch

    betas = np.asarray(params.betas, np.float32).reshape(1, -1)
    model = smplx.SMPLX(model_path=str(_SMPLX), gender=params.gender, ext="npz",
                        num_betas=betas.shape[1],
                        num_expression_coeffs=int(np.asarray(params.expression).shape[-1]), use_pca=False)
    kw = {k: torch.from_numpy(np.asarray(getattr(params, k)[t: t + 1], np.float32))
          for k in ("global_orient", "body_pose", "transl", "left_hand_pose", "right_hand_pose",
                    "jaw_pose", "leye_pose", "reye_pose", "expression")}
    with torch.no_grad():
        out = model(betas=torch.from_numpy(betas), **kw)
    return (out.joints[0, :22].numpy() @ _YUP_TO_ZUP.T)


@pytest.mark.skipif(_SEQ is None, reason="HODome data / SMPL-X model not available")
def test_bone_transforms_match_forward():
    from src.prepare.load import load
    from src.prepare.load.smpl import build_body_model

    spec = SceneSpec(
        dataset="hodome", motion_path=_SEQ,
        robot=RobotSpec(name="g1", urdf_path=Path("g1.urdf"), link_names=("a",), dof=29, height=1.3),
        smpl_model_dir=_SMPLX,
    )
    raw = load(spec)
    params = raw.smpl_params
    body = build_body_model(params, _SMPLX)

    assert body.n_bones == 55
    assert body.faces.ndim == 2 and body.faces.shape[1] == 3
    assert body.rest_vertices(params).shape[1] == 3
    assert body.posed_vertices(params, 0).shape[1] == 3

    for t in (0, raw.n_frames // 2, raw.n_frames - 1):
        rot, pos = body.bone_transforms(params, t)
        assert rot.shape == (55, 3, 3) and pos.shape == (55, 3)
        # pure-FK bone positions must match the SMPL-X forward's joints (joints are blendshape-free)
        ref = _forward_joints_zup(params, t)
        assert np.allclose(pos[:22], ref, atol=1e-4), f"frame {t}: max err {np.abs(pos[:22]-ref).max():.2e}"
        # loader's demo joints == body FK (same source)
        assert np.allclose(raw.joint_pos[t], pos[:22], atol=1e-5)
