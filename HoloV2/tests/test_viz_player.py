"""Player — le dispatch pur (frame -> chaque layer.update) est unit-testable sans serveur viser;
le run() complet (dossiers, boucle daemon, keep-alive) est vérifié par le parity check manuel (Task 12).
play_loop est testé à la signature uniquement (sans serveur viser, headless)."""
import inspect

from src.viz.core.layer import UiState
from src.viz.core.player import Player


class _RecLayer:
    """Couche fictive qui enregistre les appels update() pour la vérification."""
    folder = "X"
    def __init__(self): self.calls = []
    def setup(self, server, gui, ctx): ...
    def update(self, frame, ui): self.calls.append((frame, ui))


class _FakeSource:
    """Source fictive pour tester le dispatch sans source réelle."""
    context = None
    n_frames = 3
    def get(self, i): return f"frame{i}"


def test_n_frames_passthrough():
    """Vérifie que Player transmet n_frames depuis la Source."""
    assert Player(_FakeSource(), []).n_frames == 3


def test_dispatch_fans_out_to_every_layer():
    """Vérifie que _dispatch envoie le frame et l'UiState à chaque couche."""
    l1, l2 = _RecLayer(), _RecLayer()
    p = Player(_FakeSource(), [l1, l2])
    ui = UiState(channel="ground", color_mode="distance", point_size=0.01)
    p._dispatch("FRAME", ui)
    assert l1.calls == [("FRAME", ui)]
    assert l2.calls == [("FRAME", ui)]


def test_play_loop_importable_et_signature():
    """Vérifie que play_loop est importable sans viser et que sa signature est correcte.

    Test headless : pas de serveur démarré. On inspecte seulement la signature de la fonction
    (paramètres positionnels + keyword-only + défaut fps_default).
    """
    from src.viz.core.player import play_loop

    sig = inspect.signature(play_loop)
    params = sig.parameters
    # Paramètre positionnel : server
    assert "server" in params
    # Keyword-only obligatoires : n_frames + render
    assert params["n_frames"].kind == inspect.Parameter.KEYWORD_ONLY
    assert params["render"].kind == inspect.Parameter.KEYWORD_ONLY
    # Keyword-only optionnel : fps_default avec valeur par défaut 20
    assert params["fps_default"].kind == inspect.Parameter.KEYWORD_ONLY
    assert params["fps_default"].default == 20
