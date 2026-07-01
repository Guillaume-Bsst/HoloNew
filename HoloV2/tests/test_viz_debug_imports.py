# tests/test_viz_debug_imports.py
"""Smokes d'import pour les viewers de débogage réécrits : chaque module s'importe, expose son
point d'entrée, et référence les symboles ``core/`` canoniques (la réécriture consomme vraiment le
socle). Nécessite Phase A (viz/core) fusionnée. Aucun écran requis — viser s'importe en headless.

Logique du test ``test_scene_imports`` : toute la logique pure (calcul du point le plus bas, de
l'erreur de parité, conversion quaternion→R) est déjà couverte par ``debug._geometry`` et
``core.viser_ops`` (leurs propres tests unitaires). Le test ici est STRUCTUREL : on vérifie que
``scene`` s'importe, expose ses points d'entrée ``view_scene`` et ``main``, ET que ``Player``
provient bien de ``core.player`` (pas re-roulé dans le viewer). Aucune nouvelle op pure n'est
ajoutée dans ``scene.py``, donc un smoke structurel suffit."""
from src.viz.core.colors import diverging, parity  # noqa: F401  (utilisés dans les tâches 4–6)
from src.viz.core.player import Player


def test_scene_imports():
    """Vérifie que ``src.viz.debug.scene`` s'importe, expose ``view_scene``/``main`` et consomme
    ``core.player.Player`` (assertion structurelle : pas de player ré-roulé dans le viewer debug)."""
    from src.viz.debug import scene
    assert callable(scene.view_scene) and callable(scene.main)
    assert scene.Player is Player                        # consomme core/Player (pas de player ré-roulé)
