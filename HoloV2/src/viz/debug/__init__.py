"""``viz/debug`` — viewers de DEBUG par étage, réécrits sur ``viz/core``.

Chaque viewer est un module exécutable autonome (``python -m src.viz.debug.<x>``) qui PILOTE les
internes de l'étage qu'il visualise (load / calibration / sdf / point_cloud) pour exposer des
intermédiaires non-contractuels — l'exception debug-viewer de l'ARCHITECTURE.md. Ils consomment le
socle ``viz/core`` partagé (Player, colors, viser_ops) et confinent viser à ``core/viser_ops`` + le
module viewer. Le chemin viewer PROD vit ailleurs (``viz/app.py``) ; ceux-ci sont des outils de
débogage uniquement."""
