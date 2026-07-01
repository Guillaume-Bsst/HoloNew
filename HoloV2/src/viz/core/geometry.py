"""Helpers géométrie partagés par les couches viz — numpy-only, sans viser.

Ce module centralise les calculs de coordonnées grille→monde consommés par plusieurs couches
(sdf_iso en Phase C, debug sdf-viewer en Phase D) afin d'éviter la duplication. Règle : aucun
import viser ni torch — importable partout sans écran (testable headless).
"""
from __future__ import annotations

import numpy as np

from ...prepare.contracts import SDF


def node_coords(sdf: SDF) -> np.ndarray:
    """(Nx, Ny, Nz, 3) coordonnées locales de chaque nœud de la grille SDF.

    Convertit les indices grille discrets en positions spatiales en repère LOCAL du SDF
    (même repère que ``SDF.witness`` et ``SDF.grid``). Formule :
        coords[i, j, k] = origin + spacing * (i, j, k)

    Paramètre
    ---------
    sdf :
        Grille distance-signée portant ``origin``, ``spacing`` et la forme de ``grid`` (Nx, Ny, Nz).

    Retourne
    --------
    np.ndarray
        Tableau (Nx, Ny, Nz, 3) float64 — une position 3-D par nœud grille.
    """
    nx, ny, nz = sdf.grid.shape
    xs = sdf.origin[0] + sdf.spacing * np.arange(nx)   # (Nx,)
    ys = sdf.origin[1] + sdf.spacing * np.arange(ny)   # (Ny,)
    zs = sdf.origin[2] + sdf.spacing * np.arange(nz)   # (Nz,)
    gx, gy, gz = np.meshgrid(xs, ys, zs, indexing="ij")
    return np.stack([gx, gy, gz], axis=-1)              # (Nx, Ny, Nz, 3)
