"""VizContext porte désormais les assets statiques d'interaction (channels + correspondance) que les
couches contacts/correspondence/sdf_iso/geodesic consomment. Deux niveaux :
  - introspection (toujours exécutée) : les deux champs EXISTENT sur la dataclass ;
  - déterminisme/forme (gated data) : BakeSource les peuple identiquement build-vs-build, alignés sur
    channel_names / correspondence.n_points.

NOTE d'adaptation A/B : ``demo_scene_spec()`` n'existe pas dans le module _scene_args actuel — on
utilise directement le chemin de données HODome standard (même pattern que test_viz_bake_source.py).
"""
from __future__ import annotations

import dataclasses
import shutil

import numpy as np
import pytest

from src.viz.model import VizContext
from datapaths import CORR_NEUTRAL as _CORR, G1_URDF as _URDF


def test_vizcontext_declares_interaction_fields():
    """Garde dur : les deux champs channels + correspondence EXISTENT sur la dataclass VizContext."""
    names = {f.name for f in dataclasses.fields(VizContext)}
    assert "channels" in names, "VizContext must carry the prepare Channels (sdf+geodesic+object_idx)"
    assert "correspondence" in names, "VizContext must carry the SMPL<->robot CorrespondenceTable"


# --- déterminisme/forme via BakeSource sur la scène démo (gated data, max_frames bas) ---

# Chemins HoloV2-internes — skip propre si données absentes (sources via datapaths).


def _pick_hodome():
    """Retourne un chemin de séquence HODome valide ou None si les données sont absentes."""
    try:
        from datapaths import HODOME, SMPLX_MODELS
    except ImportError:
        return None, None
    sm, ob = HODOME / "smplx", HODOME / "object"
    if not (sm.is_dir() and ob.is_dir() and SMPLX_MODELS.is_dir() and _CORR.exists()):
        return None, None
    shared = {p.stem for p in sm.glob("*.npz")} & {p.stem for p in ob.glob("*.npz")}
    if not shared:
        return None, None
    return sm / f"{sorted(shared)[0]}.npz", SMPLX_MODELS


_SEQ, _SMPLX = _pick_hodome()
_SKIP = pytest.mark.skipif(
    _SEQ is None,
    reason="HODome data / SMPL-X model / corr_neutral.npz absent -> bake gated",
)


@pytest.fixture(scope="module")
def _cache(tmp_path_factory):
    """Cache temporaire avec la correspondance neutrali copiée (évite de polluer le cache projet)."""
    c = tmp_path_factory.mktemp("viz_interaction_cache")
    (c / "correspondence").mkdir(parents=True, exist_ok=True)
    shutil.copy(_CORR, c / "correspondence" / "corr_neutral.npz")
    return c


def _spec(cache):
    """SceneSpec HODome minimal, adaptée au cache temporaire."""
    from src.prepare.contracts import RobotSpec, SceneSpec
    robot = RobotSpec(name="g1", urdf_path=_URDF, link_names=("pelvis",), dof=29, height=1.3)
    return SceneSpec(dataset="hodome", motion_path=_SEQ, robot=robot,
                     smpl_model_dir=_SMPLX, cache_dir=cache)


@_SKIP
def test_bakesource_populates_interaction_context_deterministic(_cache):
    """Déterminisme/forme : BakeSource peuple channels + correspondence identiquement build×2.

    - channels alignés avec channel_names (même ordre, même noms) ;
    - correspondence.n_points cohérent avec smpl_idx.shape[0] ;
    - deux builds indépendants donnent les mêmes noms de canaux et le même appariement.
    """
    from src.prepare.config import PrepareConfig
    from src.viz.sources import BakeSource

    s1 = BakeSource(_spec(_cache), PrepareConfig(), solve=False, frame_step=8, max_frames=2)
    s2 = BakeSource(_spec(_cache), PrepareConfig(), solve=False, frame_step=8, max_frames=2)
    c1, c2 = s1.context, s2.context

    # alignement : channels <-> channel_names ; correspondence M-points cohérent
    assert tuple(ch.name for ch in c1.channels) == tuple(c1.channel_names)
    assert c1.correspondence.n_points == c1.correspondence.smpl_idx.shape[0]

    # déterminisme : mêmes noms de canaux, même appariement
    assert tuple(ch.name for ch in c1.channels) == tuple(ch.name for ch in c2.channels)
    assert np.array_equal(c1.correspondence.smpl_idx, c2.correspondence.smpl_idx)
    assert np.array_equal(c1.correspondence.link_idx, c2.correspondence.link_idx)
