"""Tests du lecteur FBX binaire (``prepare/load/fbx.read_object_fbx``) — op pure.

Gate sur la présence du dataset PA-HOI (le vrai ``1_001_o.fbx`` sert de fixture : petit, régulier,
valeurs connues). Skip propre quand la donnée est absente (comme les autres tests data-gated).
"""
from __future__ import annotations

import numpy as np
import pytest

from datapaths import PAHOI
from src.prepare.load.fbx import read_object_fbx

_FBX = (PAHOI / "cap_res_fbx" / "1_001_o.fbx") if PAHOI else None
pytestmark = pytest.mark.skipif(_FBX is None or not _FBX.exists(),
                                reason="PA-HOI 1_001_o.fbx indisponible")


@pytest.fixture(scope="module")
def ofbx():
    return read_object_fbx(_FBX)


def test_frame_count_and_fps(ofbx):
    # 1_001 = 208 frames @ 30 fps (cf. frames.txt "1_001:208").
    assert ofbx.transl_native.shape == (208, 3)
    assert ofbx.rot_native.shape == (208, 3, 3)
    assert ofbx.fps == pytest.approx(30.0, abs=1e-6)


def test_object_name(ofbx):
    # Nœud "01_milkbox" -> nom d'objet sans le préfixe d'index.
    assert ofbx.name == "milkbox"


def test_mesh_nonempty_and_faces_valid(ofbx):
    assert ofbx.vertices.ndim == 2 and ofbx.vertices.shape[1] == 3
    assert ofbx.vertices.shape[0] >= 8              # proxy boîte
    assert ofbx.faces.ndim == 2 and ofbx.faces.shape[1] == 3
    assert ofbx.faces.min() >= 0
    assert ofbx.faces.max() < ofbx.vertices.shape[0]


def test_units_meters_and_yup_lift(ofbx):
    # Translation en mètres, repère natif Y-up : sur ce "grab from ground", la hauteur (Y) monte
    # d'~0.16 m (sol) à ~1 m. Étendue verticale nettement > étendues X/Z ici.
    t = ofbx.transl_native
    y = t[:, 1]
    assert 0.05 < y.min() < 0.4          # départ près du sol
    assert 0.8 < y.max() < 1.2           # levé à hauteur de prise
    assert np.abs(t).max() < 5.0         # mètres, pas centimètres


def test_rotations_orthonormal(ofbx):
    R = ofbx.rot_native
    eye = np.einsum("tij,tkj->tik", R, R)          # R Rᵀ = I
    assert np.allclose(eye, np.eye(3)[None], atol=1e-4)
    assert np.allclose(np.linalg.det(R), 1.0, atol=1e-4)
