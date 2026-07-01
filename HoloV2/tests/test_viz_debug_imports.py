# tests/test_viz_debug_imports.py
"""Smokes d'import pour les viewers de débogage réécrits : chaque module s'importe, expose son
point d'entrée, et référence les symboles ``core/`` canoniques (la réécriture consomme vraiment le
socle). Nécessite Phase A (viz/core) fusionnée. Aucun écran requis — viser s'importe en headless.

Logique du test ``test_scene_imports`` : toute la logique pure (calcul du point le plus bas, de
l'erreur de parité, conversion quaternion→R) est déjà couverte par ``debug._geometry`` et
``core.viser_ops`` (leurs propres tests unitaires). Le test ici est STRUCTUREL : on vérifie que
``scene`` s'importe, expose ses points d'entrée ``view_scene`` et ``main``, ET que ``play_loop``
provient bien de ``core.player`` (pas re-roulé dans le viewer). Aucune nouvelle op pure n'est
ajoutée dans ``scene.py``, donc un smoke structurel suffit."""
from src.viz.core.colors import diverging, parity  # noqa: F401  (utilisés dans les tâches 4–6)
from src.viz.core.player import play_loop


def test_scene_imports():
    """Vérifie que ``src.viz.debug.scene`` s'importe, expose ``view_scene``/``main`` et consomme
    ``core.player.play_loop`` (assertion structurelle : la boucle de lecture vient bien du socle core)."""
    from src.viz.debug import scene
    assert callable(scene.view_scene) and callable(scene.main)
    assert scene.play_loop is play_loop              # consomme core/play_loop (pas de player ré-roulé)


def test_cloud_imports():
    """Vérifie que ``src.viz.debug.cloud`` s'importe, expose ``view_cloud``/``main`` et consomme
    ``core.player.play_loop`` + ``core.colors.parity`` (assertion structurelle : socle core bien câblé)."""
    from src.viz.debug import cloud
    assert callable(cloud.view_cloud) and callable(cloud.main)
    assert cloud.play_loop is play_loop and cloud.parity is parity   # consomme core/play_loop + core/colors.parity


def test_sdf_imports():
    """Vérifie que ``src.viz.debug.sdf`` s'importe, expose ``view_sdf``/``main`` et consomme
    ``core.colors.diverging`` + ``core.geometry.node_coords`` (assertions structurelles : le viewer
    est sans axe temps et délègue bien aux helpers core partagés, sans re-rouler la logique)."""
    from src.viz.debug import sdf
    assert callable(sdf.view_sdf) and callable(sdf.main)
    assert sdf.diverging is diverging                       # consomme core/colors.diverging
    assert callable(sdf.node_coords)                        # consomme le helper pur core.geometry
