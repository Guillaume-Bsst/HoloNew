"""Test d'intégration du loader PA-HOI. Skip quand le dataset PA-HOI ou le modèle SMPL-X n'est pas
dispo localement (donnée machine-spécifique)."""
from pathlib import Path

import numpy as np
import pytest

from src.prepare.contracts import RobotSpec, SceneSpec
from datapaths import PAHOI as _PAHOI, SMPLX_MODELS as _SMPLX


def _pick_sequence() -> Path | None:
    """Une séquence ayant à la fois <seq>.npz et cap_res_fbx/<seq>_o.fbx (donc un objet), sinon None."""
    if _PAHOI is None or not (_PAHOI.is_dir() and _SMPLX.is_dir()):
        return None
    fbx_dir = _PAHOI / "cap_res_fbx"
    for sub in ("cap_res_bvh_s1", "cap_res_bvh_s2"):
        d = _PAHOI / sub
        if not d.is_dir():
            continue
        for seq in sorted(p.name for p in d.iterdir() if p.is_dir()):
            npz = d / seq / f"{seq}.npz"
            if npz.exists() and (fbx_dir / f"{seq}_o.fbx").exists():
                return npz
    return None


_SEQ = _pick_sequence()


@pytest.mark.skipif(_SEQ is None, reason="PA-HOI data / SMPL-X model not available")
def test_pahoi_load_contract():
    from src.prepare.load import load  # lazy : importe + enregistre le loader pahoi

    spec = SceneSpec(
        dataset="pahoi", motion_path=_SEQ,
        robot=RobotSpec(name="g1", urdf_path=Path("g1.urdf"), link_names=("a",), dof=29, height=1.3),
        smpl_model_dir=_SMPLX,
    )
    raw = load(spec)
    T = raw.n_frames

    assert raw.is_parametric and T > 0
    assert raw.source_format == "pahoi"
    assert raw.joint_pos.shape == (T, 22, 3)
    assert len(raw.joint_names) == 22
    p = raw.smpl_params
    assert p.model_type == "smplx"
    assert p.body_pose.shape == (T, 63)
    assert p.left_hand_pose.shape == (T, 45) and p.right_hand_pose.shape == (T, 45)
    # un objet : poses (T,7) pos-first wxyz, mesh proxy résolu sur disque
    assert len(raw.object_poses_raw) == 1 and len(raw.object_mesh_paths) == 1
    poses = raw.object_poses_raw[0]
    assert poses.shape == (T, 7)
    assert np.allclose(np.linalg.norm(poses[:, 3:], axis=1), 1.0, atol=1e-4)   # quats normés
    assert raw.object_mesh_paths[0].exists()
    # Z-up sanity : le corps a une étendue verticale plausible (debout-ish).
    assert np.ptp(raw.joint_pos[:, :, 2]) > 0.3


@pytest.mark.skipif(_SEQ is None, reason="PA-HOI data / SMPL-X model not available")
def test_pahoi_object_reaches_a_hand():
    """Garde d'alignement (objet & humain dans le MÊME monde) : au fil d'une interaction, l'objet
    passe près d'un poignet. Valide échelle + axes de la trajectoire objet (pas seulement les formes)."""
    from src.prepare.load import load

    spec = SceneSpec(
        dataset="pahoi", motion_path=_SEQ,
        robot=RobotSpec(name="g1", urdf_path=Path("g1.urdf"), link_names=("a",), dof=29, height=1.3),
        smpl_model_dir=_SMPLX,
    )
    raw = load(spec)
    obj = raw.object_poses_raw[0][:, :3]          # (T,3) monde Z-up
    l_wrist = raw.joint_pos[:, 20]                 # L_Wrist
    r_wrist = raw.joint_pos[:, 21]                 # R_Wrist
    d = np.minimum(np.linalg.norm(obj - l_wrist, axis=1), np.linalg.norm(obj - r_wrist, axis=1))
    assert d.min() < 0.6, f"objet jamais proche d'une main (min={d.min():.2f} m) — désalignement ?"
