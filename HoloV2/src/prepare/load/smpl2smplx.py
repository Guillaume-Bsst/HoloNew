"""SMPL -> SMPL-X betas (shape) transfer via deformation transfer.

HOI-M3 ships EasyMocap SMPL (10 betas); the rest of the V2 pipeline is SMPL-X. SMPL and SMPL-X
have DIFFERENT shape spaces, so betas cannot be reused directly (a few-cm body-shape error). We
map the subject's SMPL shape to SMPL-X ONCE (shape is time-invariant):

  1. build the SMPL rest mesh ``v_template + shapedirs @ betas`` (zero pose -> no LBS/pose blend);
     the body template comes from the SMPL-H neutral model (SMPL-H shares SMPL's body template);
  2. push it through the SMPL->SMPL-X deformation-transfer matrix from vchoutas/smplx
     ``transfer_model`` (the ``mtx`` in ``smpl2smplx_deftrafo_setup.pkl``, a sparse map of the
     normals-stacked source ``[verts ; verts+normals]`` -> SMPL-X-topology vertices);
  3. fit SMPL-X betas to that rest mesh. At rest the SMPL-X mesh is LINEAR in betas, so this is a
     single least-squares solve. Shape is translation-free, so both meshes are centred first (the
     SMPL and SMPL-X templates sit at different origins).

Pure numpy, no per-frame optimisation. Halves the shape error vs naive betas reuse (median ~6mm
vs ~12mm, p95 ~20mm vs ~39mm on HOI-M3 subjects)."""
from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np


def _smpl_rest_mesh(smplh_npz: Path, betas10: np.ndarray):
    """SMPL subject rest mesh (V, 3) + faces from the SMPL-H neutral model (its body == SMPL)."""
    d = np.load(str(smplh_npz), allow_pickle=True)
    vt = np.asarray(d["v_template"], np.float64)                 # (6890, 3)
    sd = np.asarray(d["shapedirs"], np.float64)[:, :, :10]       # (6890, 3, 10)
    faces = np.asarray(d["f"], np.int64)
    return vt + sd @ np.asarray(betas10, np.float64)[:10], faces


def smpl_rest_pelvis(betas10: np.ndarray, smplh_npz: Path) -> np.ndarray:
    """(3,) SMPL rest pelvis (native frame) for ``betas10`` = ``J_regressor @ rest_mesh``, row 0.

    EasyMocap calibrates its ``Th`` against the SMPL pelvis; converting to SMPL-X must re-place the
    SMPL-X pelvis at the SMPL one (their rest heights differ ~14cm), so the loader needs this."""
    verts, _ = _smpl_rest_mesh(Path(smplh_npz), betas10)
    jreg = np.asarray(np.load(str(smplh_npz), allow_pickle=True)["J_regressor"], np.float64)
    return (jreg @ verts)[0]


def smpl_betas_to_smplx(betas_smpl: np.ndarray, smplh_npz: Path, smplx_npz: Path,
                        deftrafo_pkl: Path, n_out: int = 16) -> np.ndarray:
    """Map a subject's SMPL 10-betas to SMPL-X ``n_out``-betas. Returns ``(n_out,)`` float32."""
    import trimesh

    verts, faces = _smpl_rest_mesh(Path(smplh_npz), betas_smpl)
    normals = np.asarray(trimesh.Trimesh(verts, faces, process=False).vertex_normals, np.float64)
    stacked = np.concatenate([verts, verts + normals], axis=0)  # (2*6890, 3) normals-stacked

    with open(deftrafo_pkl, "rb") as f:
        mtx = pickle.load(f, encoding="latin1")["mtx"].tocsr()  # (10475, 13780) sparse
    target = np.asarray(mtx @ stacked)                          # (10475, 3) SMPL-X-topology rest

    sx = np.load(str(smplx_npz), allow_pickle=True)
    vt_x = np.asarray(sx["v_template"], np.float64)             # (10475, 3)
    sd_x = np.asarray(sx["shapedirs"], np.float64)[:, :, :n_out]
    # Centre both (shape is translation-free), then betas is one least-squares solve.
    rhs = ((target - target.mean(0)) - (vt_x - vt_x.mean(0))).reshape(-1)
    betas_x, *_ = np.linalg.lstsq(sd_x.reshape(-1, n_out), rhs, rcond=None)
    return betas_x.astype(np.float32)
