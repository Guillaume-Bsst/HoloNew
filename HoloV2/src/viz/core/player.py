"""Player — possède la GUI Playback + Selectors, le rendu par frame qui distribue le vue-modèle
à chaque couche, la boucle daemon play/fps et le keep-alive. Factorise le boilerplate autrefois
dupliqué sur les quatre viewers. viser est importé PARESSEUSEMENT dans ``run`` pour que la logique
dispatch reste testable.

Contrat de câblage : ``render()`` est câblée au slider + aux trois sélecteurs (canal / mode couleur /
taille point); chaque couche câble SON PROPRE checkbox localement (dans setup) pour basculer la
visibilité de sa poignée. Parce que render() rafraîchit CHAQUE poignée de couche sur chaque slider/
sélecteur, une couche en pause qui bascule sa visibilité bascule le rendu de la poignée du frame courant
(jamais périmée)."""
from __future__ import annotations

import threading
import time

from ..model import VizFrame
from .layer import Layer, UiState


class Player:
    """Pilote une liste de ``Layer`` sur une ``Source``. ``run()`` construit le serveur viser + la
    GUI et sert; le pur ``_dispatch`` (frame -> chaque layer.update) est unit-testable sans viser.

    ``panels`` : panels 2D optionnels (ex. ``CostDashboard``) dont le ``setup(server, gui, frames)``
    est appelé une seule fois dans ``run()``, après les couches. Les panels lisent toute la séquence
    (fournie via ``source.frames`` si disponible, sinon reconstituée via ``source.get``)."""

    def __init__(self, source, layers: list[Layer], *, port: int = 8080,
                 panels: list | None = None) -> None:
        """Initialise le joueur avec une source, une liste de couches et des panels optionnels.

        Args:
            source: Source (duck-typed: .context, .n_frames, .get(i), optionnel .frames).
            layers: Liste de Layer à piloter.
            port: Port viser (défaut 8080).
            panels: Panels 2D additionnels (ex. CostDashboard) setup une fois avec toute la séquence.
        """
        self.source = source
        self.layers = list(layers)
        self.port = port
        self.panels = list(panels) if panels else []

    @property
    def n_frames(self) -> int:
        """Nombre de frames dans la source."""
        return self.source.n_frames

    def _dispatch(self, frame: VizFrame, ui: UiState) -> None:
        """Envoie un (frame, ui) à chaque couche. Pas de viser, pas d'état — pur fan-out
        (testable sans écran)."""
        for layer in self.layers:
            layer.update(frame, ui)

    def run(self) -> None:
        """Construit le serveur viser, les dossiers GUI (Playback, Selectors), wire les callbacks,
        puis lance la boucle daemon play/fps et le keep-alive principal."""
        import viser

        ctx = self.source.context
        srv = viser.ViserServer(port=self.port)
        srv.scene.add_grid("/grid", width=4.0, height=4.0)

        # Dossier Playback : slider frame, play, fps
        with srv.gui.add_folder("Playback"):
            sld = srv.gui.add_slider("frame", 0, max(self.n_frames - 1, 1), 1, 0)
            play = srv.gui.add_checkbox("play", False)
            fps = srv.gui.add_number("fps", 20, min=1, max=120, step=1)

        # Dossier Selectors : canal, mode couleur, taille point
        with srv.gui.add_folder("Selectors"):
            channel = srv.gui.add_dropdown("channel", ctx.channel_names,
                                           initial_value=ctx.channel_names[0])
            color_mode = srv.gui.add_dropdown("colour mode", ("uniform", "distance", "active"),
                                              initial_value="distance")
            size = srv.gui.add_number("point size", 0.012, min=0.002, max=0.05, step=0.002)
        info = srv.gui.add_markdown("")

        # Initialise chaque couche
        for layer in self.layers:
            layer.setup(srv, srv.gui, ctx)

        # Initialise les panels (une seule fois avec toute la séquence)
        if self.panels:
            # Récupère la séquence complète : .frames si la source l'expose (BakeSource),
            # sinon reconstituée via get() (source quelconque).
            all_frames = (
                self.source.frames if hasattr(self.source, "frames")
                else [self.source.get(i) for i in range(self.n_frames)]
            )
            for panel in self.panels:
                panel.setup(srv, srv.gui, all_frames)

        def render(_=None):
            """Callback de rendu : récupère le frame courant, assemble l'UiState, dispatche aux
            couches, met à jour le markdown info."""
            i = int(sld.value)
            frame = self.source.get(i)
            ui = UiState(channel=channel.value, color_mode=color_mode.value,
                         point_size=float(size.value))
            self._dispatch(frame, ui)
            info.content = (f"**frame {i + 1}/{self.n_frames}** · channel **{channel.value}** · "
                            f"colour **{color_mode.value}** · margin {ctx.margin:.3f} m")

        # Câble render() aux contrôles slider + sélecteurs
        for h in (sld, channel, color_mode, size):
            h.on_update(render)
        render()

        # Boucle daemon play/fps : avance le slider quand play=True
        def loop():
            while True:
                if play.value:
                    sld.value = (int(sld.value) + 1) % self.n_frames
                    render()
                time.sleep(1.0 / float(fps.value))
        threading.Thread(target=loop, daemon=True).start()

        print(f"viser ready -> http://localhost:{self.port}")

        # Keep-alive principal
        while True:
            time.sleep(1)


def play_loop(server, *, n_frames: int, render, fps_default: int = 20) -> None:
    """Boucle de lecture générique pour les viewers debug (sans Source ni VizFrame).

    Construit un dossier GUI « Playback » (slider frame / case à cocher play / fps) sur le serveur
    viser fourni, câble ``render(i)`` sur le slider + déclenche un rendu initial, puis lance la
    boucle daemon play/fps et le keep-alive principal.

    ``render`` est un callable prenant l'index de frame (int) ; les viewers debug passent leur propre
    ``render(i)`` (qui pilote les internes de l'étage qu'ils visualisent). Le Player prod conserve
    sa propre logique (``Player.run()``), cette fonction est réservée aux viewers sans Source.

    ``import viser`` est local à cette fonction : le module ``player`` reste importable sans viser.

    Args:
        server: Serveur viser sur lequel construire la GUI (``viser.ViserServer``).
        n_frames: Nombre total de frames (borne du slider : 0 .. n_frames-1).
        render: Callable ``(int) -> None`` — appelé à chaque changement de slider et au démarrage.
        fps_default: Valeur initiale du contrôle fps (défaut 20).
    """
    import threading

    # Dossier Playback : slider frame, play, fps — même structure que Player.run()
    with server.gui.add_folder("Playback"):
        sld = server.gui.add_slider("frame", 0, max(n_frames - 1, 1), 1, 0)
        play = server.gui.add_checkbox("play", False)
        fps = server.gui.add_number("fps", fps_default, min=1, max=120, step=1)

    def _on_update(_=None) -> None:
        """Callback déclenché par le slider : appelle render avec l'index courant."""
        render(int(sld.value))

    sld.on_update(_on_update)
    # Rendu initial
    render(0)

    # Boucle daemon play/fps : avance le slider quand play=True
    def _loop() -> None:
        while True:
            if play.value:
                sld.value = (int(sld.value) + 1) % n_frames
                render(int(sld.value))
            time.sleep(1.0 / float(fps.value))

    threading.Thread(target=_loop, daemon=True).start()

    # Keep-alive principal
    while True:
        time.sleep(1)
