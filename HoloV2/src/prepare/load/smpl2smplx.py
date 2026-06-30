"""Transfert SMPL -> SMPL-X betas (shape) via deformation transfer.

HOI-M3 propose EasyMocap SMPL (10 betas) ; le reste du pipeline V2 est SMPL-X. SMPL et SMPL-X
ont des espaces de shape DIFFÉRENTS, donc les betas ne peuvent pas être réutilisées directement
(erreur de forme du corps de quelques cm). Nous cartographions la shape SMPL du sujet à SMPL-X
UNE FOIS (la shape est temps-invariant) :

  1. construire le mesh rest SMPL ``v_template + shapedirs @ betas`` (pose zéro -> pas de blend LBS/pose) ;
     le template du corps vient du modèle neutre SMPL-H (SMPL-H partage le template du corps SMPL) ;
  2. le pousser à travers la matrice deformation-transfer SMPL->SMPL-X de vchoutas/smplx
     ``transfer_model`` (le ``mtx`` dans ``smpl2smplx_deftrafo_setup.pkl``, une carte clairsemée de
     la source empilée-normales ``[verts ; verts+normals]`` -> sommets topologie SMPL-X) ;
  3. adapter les betas SMPL-X à ce mesh rest. Au rest, le mesh SMPL-X est LINÉAIRE en betas, donc c'est
     une seule résolution des moindres carrés. La shape est sans translation, donc les deux meshes sont
     d'abord centrés (les templates SMPL et SMPL-X sont à des origines différentes).

Numpy pur, pas d'optimisation par frame. Réduit de moitié l'erreur de shape vs réutilisation naïve de betas
(médiane ~6mm vs ~12mm, p95 ~20mm vs ~39mm sur les sujets HOI-M3)."""
from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np


def _smpl_rest_mesh(smplh_npz: Path, betas10: np.ndarray):
    """Mesh rest sujet SMPL (V, 3) + faces du modèle neutre SMPL-H (son body == SMPL)."""
    d = np.load(str(smplh_npz), allow_pickle=True)
    vt = np.asarray(d["v_template"], np.float64)                 # (6890, 3)
    sd = np.asarray(d["shapedirs"], np.float64)[:, :, :10]       # (6890, 3, 10)
    faces = np.asarray(d["f"], np.int64)
    return vt + sd @ np.asarray(betas10, np.float64)[:10], faces


def smpl_rest_pelvis(betas10: np.ndarray, smplh_npz: Path) -> np.ndarray:
    """(3,) bassin rest SMPL (frame natif) pour ``betas10`` = ``J_regressor @ rest_mesh``, ligne 0.

    EasyMocap calibre son ``Th`` contre le bassin SMPL ; convertir en SMPL-X doit replacer le bassin
    SMPL-X au SMPL (leurs hauteurs rest diffèrent ~14cm), donc le chargeur en a besoin."""
    verts, _ = _smpl_rest_mesh(Path(smplh_npz), betas10)
    jreg = np.asarray(np.load(str(smplh_npz), allow_pickle=True)["J_regressor"], np.float64)
    return (jreg @ verts)[0]


def smpl_betas_to_smplx(betas_smpl: np.ndarray, smplh_npz: Path, smplx_npz: Path,
                        deftrafo_pkl: Path, n_out: int = 16) -> np.ndarray:
    """Mapper les 10-betas SMPL d'un sujet aux ``n_out``-betas SMPL-X. Retourne ``(n_out,)`` float32."""
    import trimesh

    verts, faces = _smpl_rest_mesh(Path(smplh_npz), betas_smpl)
    normals = np.asarray(trimesh.Trimesh(verts, faces, process=False).vertex_normals, np.float64)
    stacked = np.concatenate([verts, verts + normals], axis=0)  # (2*6890, 3) empilé-normales

    with open(deftrafo_pkl, "rb") as f:
        mtx = pickle.load(f, encoding="latin1")["mtx"].tocsr()  # (10475, 13780) clairsemé
    target = np.asarray(mtx @ stacked)                          # (10475, 3) rest topologie SMPL-X

    sx = np.load(str(smplx_npz), allow_pickle=True)
    vt_x = np.asarray(sx["v_template"], np.float64)             # (10475, 3)
    sd_x = np.asarray(sx["shapedirs"], np.float64)[:, :, :n_out]
    # Centrer les deux (la shape est sans translation), puis betas est une seule résolution des moindres carrés.
    rhs = ((target - target.mean(0)) - (vt_x - vt_x.mean(0))).reshape(-1)
    betas_x, *_ = np.linalg.lstsq(sd_x.reshape(-1, n_out), rhs, rcond=None)
    return betas_x.astype(np.float32)
