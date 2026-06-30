"""Transport optimal entropique par segment entre le nuage source humain et le nuage cible du robot.

Indexé par les points du robot : pour chaque point de surface du robot, il retourne le point humain
qui le pilote. Chaque colonne d'appairage souple est transformée en une source humaine unique via
une image barycentrique attachée au plus proche échantillon humain dans le MÊME segment. L'assignation
est fonctionnelle, pas injective — deux points du robot peuvent partager un point humain (le champ
humain est simplement lu à la même localisation corporelle deux fois), ce qui convient au pilotage de
chaque point du robot sans forcer le nuage du robot plus creux que l'humain.

Les deux nuages sont dans une T-pose correspondante et le même repère mondial, donc les segments
correspondants partagent leur orientation : le coût par segment est centre + échelle isotrope +
distance au carré (haut/avant/gauche restent cohérents — pas de retournements bout à bout, pas
de miroir gauche<->droit).
"""
from __future__ import annotations

import numpy as np


def couple(human_pts: np.ndarray, human_seg: np.ndarray, robot_pts: np.ndarray,
           robot_seg: np.ndarray, reg: float) -> np.ndarray:
    """``smpl_idx (M,)`` : pour chaque point du robot, l'indice du point humain qui le pilote
    (dans l'ordre des points du nuage humain). ``reg`` = régularisation entropique de Sinkhorn sur
    les coordonnées normalisées par segment. Les entrées indexent dans ``human_pts`` et peuvent se
    répéter (fonctionnel, pas injectif)."""
    import ot
    from scipy.spatial import cKDTree

    m = robot_pts.shape[0]
    smpl_idx = np.full(m, -1, dtype=np.int64)
    for s in np.unique(robot_seg):
        ir = np.flatnonzero(robot_seg == s)
        ih = np.flatnonzero(human_seg == s)
        if ih.size == 0:
            raise ValueError(f"segment {int(s)} has robot points but no human source points")
        xh = human_pts[ih].astype(np.float64)
        xr = robot_pts[ir].astype(np.float64)

        # Centre + échelle isotrope chaque segment localement pour que le coût porte sur la forme relative.
        hn = (xh - xh.mean(0)) / (xh.std() + 1e-8)
        rn = (xr - xr.mean(0)) / (xr.std() + 1e-8)

        cost = ot.dist(hn, rn, metric="sqeuclidean")
        cost /= (cost.max() + 1e-12)
        plan = ot.sinkhorn(np.full(ih.size, 1.0 / ih.size),
                           np.full(ir.size, 1.0 / ir.size), cost, reg)   # (n_h, n_r)

        # Image barycentrique de chaque point du robot en coordonnées humaines, attachée au plus proche
        # échantillon humain du segment (fonctionnel : plusieurs points du robot peuvent atterrir sur
        # le même point humain).
        image = (plan.T @ xh) / (plan.sum(0)[:, None] + 1e-12)            # (n_r, 3)
        _, nn = cKDTree(xh).query(image, k=1)
        smpl_idx[ir] = ih[nn]

    if (smpl_idx < 0).any():
        raise ValueError("some robot points were left unassigned by the OT coupling")
    return smpl_idx
