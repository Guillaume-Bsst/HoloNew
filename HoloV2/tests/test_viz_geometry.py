"""Tests unitaires pour ``src.viz.core.geometry`` — numpy-only, pas de viser.

Vérifie que ``node_coords`` calcule correctement les coordonnées monde d'une grille SDF sur des cas
petits et analytiquement connus : origine nulle, origine décalée, plusieurs espacements."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from src.prepare.contracts import SDF
from src.viz.core.geometry import node_coords


def _sdf(nx: int, ny: int, nz: int, *, origin=(0.0, 0.0, 0.0), spacing: float = 1.0) -> SDF:
    """Construit un SDF minimal de taille (nx, ny, nz) avec zéros (géométrie fictive)."""
    return SDF(
        grid=np.zeros((nx, ny, nz), np.float32),
        witness=np.zeros((nx, ny, nz, 3), np.float32),
        origin=np.array(origin, dtype=np.float64),
        spacing=spacing,
        name="test",
    )


def test_node_coords_shape():
    """La forme de sortie est bien (Nx, Ny, Nz, 3)."""
    sdf = _sdf(3, 4, 5)
    coords = node_coords(sdf)
    assert coords.shape == (3, 4, 5, 3), f"attendu (3,4,5,3), obtenu {coords.shape}"


def test_node_coords_origin_zero_spacing_one():
    """Avec origine (0,0,0) et espacement 1, les coordonnées coïncident avec les indices."""
    sdf = _sdf(2, 2, 2, origin=(0.0, 0.0, 0.0), spacing=1.0)
    coords = node_coords(sdf)
    # nœud (0,0,0) -> (0,0,0)
    np.testing.assert_allclose(coords[0, 0, 0], [0.0, 0.0, 0.0])
    # nœud (1,0,0) -> (1,0,0)
    np.testing.assert_allclose(coords[1, 0, 0], [1.0, 0.0, 0.0])
    # nœud (0,1,0) -> (0,1,0)
    np.testing.assert_allclose(coords[0, 1, 0], [0.0, 1.0, 0.0])
    # nœud (0,0,1) -> (0,0,1)
    np.testing.assert_allclose(coords[0, 0, 1], [0.0, 0.0, 1.0])
    # nœud (1,1,1) -> (1,1,1)
    np.testing.assert_allclose(coords[1, 1, 1], [1.0, 1.0, 1.0])


def test_node_coords_nonzero_origin():
    """L'origine décalée est correctement répercutée sur toutes les coordonnées."""
    origin = (2.0, -1.0, 0.5)
    sdf = _sdf(2, 2, 2, origin=origin, spacing=0.5)
    coords = node_coords(sdf)
    # nœud (0,0,0) = origin
    np.testing.assert_allclose(coords[0, 0, 0], list(origin), atol=1e-12)
    # nœud (1,0,0) = origin + (0.5, 0, 0)
    np.testing.assert_allclose(coords[1, 0, 0], [2.5, -1.0, 0.5], atol=1e-12)
    # nœud (1,1,1) = origin + (0.5, 0.5, 0.5)
    np.testing.assert_allclose(coords[1, 1, 1], [2.5, -0.5, 1.0], atol=1e-12)


def test_node_coords_arbitrary_spacing():
    """L'espacement est appliqué uniformément sur les trois axes."""
    sdf = _sdf(3, 1, 1, origin=(0.0, 0.0, 0.0), spacing=0.1)
    coords = node_coords(sdf)
    # x-axis : 0, 0.1, 0.2
    np.testing.assert_allclose(coords[:, 0, 0, 0], [0.0, 0.1, 0.2], atol=1e-12)


def test_node_coords_import_numpy_only():
    """Vérifie qu'importer le module ne tire PAS viser ni torch (numpy-only).

    Testé dans un sous-processus isolé pour garantir un sys.modules frais,
    robuste à l'ordre des tests : même si un test antérieur a importé viser,
    ce test procède à une vérification en processus enfant indépendant."""
    # Racine HoloV2/ (cwd pour le sous-processus et pour importer src.*)
    holov2 = Path(__file__).resolve().parent.parent

    script = (
        "import sys; "
        "import src.viz.core.geometry; "
        "viser_present = 'viser' in sys.modules; "
        "torch_present = 'torch' in sys.modules; "
        "assert not viser_present, "
        "'geometry.py ne doit pas importer viser'; "
        "assert not torch_present, "
        "'geometry.py ne doit pas importer torch'"
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        cwd=str(holov2),
    )
    assert result.returncode == 0, (
        f"import src.viz.core.geometry a chargé viser ou torch :\n{result.stderr}"
    )
