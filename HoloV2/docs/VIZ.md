# HoloV2 — `viz/` (visualiseur)

Module **top-level indépendant**, **consommateur pur** : il LIT des artefacts typés et les
affiche. **Zéro hook** dans le calcul, jamais d'`if visualize` dans `prepare`/`targets`.
`viz/` dépend de `contracts.py` uniquement ; jamais l'inverse.

## Le seam : `trace_frame` -> `FrameTrace`

`targets/pipeline.py` expose DEUX entrées, mêmes fonctions pures :
- `process_frame(prepared, f) -> FrameTargets`   (lean, prod)
- `trace_frame(prepared, f)   -> FrameTrace`      (instrumenté : garde chaque intermédiaire)

`FrameTrace` (dans `contracts.py`) bundle tous les artefacts d'une frame :
`pose (FramePose)` · `human_cloud_world` · `object_clouds_world` · `human_field`
(pré-transport) · `targets (FrameTargets)`. → intermédiaires explicites et typés, pas
d'effets de bord.

## Design retenu : UN gros viewer viser, plein de toggles (pas un registry)

`viz/viewer.py` = un seul `Viewer` viser avec beaucoup de paramètres activables/désactivables.
Interne rangé proprement : **une méthode `_draw_<couche>` par couche**, gardée par sa checkbox.
C'est le SEUL endroit qui connaît viser (effets confinés).

### Entrées du viewer
- assets STATIQUES de `prepare/` : `GroundedScene`, `InteractionContext` (channels, nuages-rest,
  correspondance), `Calibration`.
- un fournisseur de frames `get(f) -> FrameTrace` :
  - **bake** : précalcule tous les `FrameTrace` -> playback fluide (offline),
  - **live** : `trace_frame(f)` à la volée (debug téléop).
  Même viewer, on branche l'un ou l'autre.

### Couches (toggles), groupées
- **Playback** : slider frame · play · fps
- **Statique** : sol (SDF de plan) · mesh SMPL (ghost) · meshes objets · SDF (iso ≈ surface) ·
  correspondance (lignes SMPL↔G1) · nuages-rest
- **Style** : points cibles par lien · poids
- **Interaction – humain** : nuage humain posé · witnesses · normales · masque actif
- **Interaction – objets** : nuages objets posés · champs
- **Interaction – robot** : points G1 (champ transporté)
- **Sélecteurs globaux** : canal (dropdown : ground/obj0/…) · mode couleur (uni / heatmap
  distance / actif) · taille des points

### Statique vs dynamique (perf)
- statique : ajouté UNE fois (handles gardés).
- dynamique : mis à jour au slider depuis `FrameTrace` (handles `.points/.colors/.visible`).

## Visualiseurs de debug incrémentaux (par étape)

Avant le gros `viewer.py` (FrameTrace), des viewers viser **focalisés** valident chaque étape tôt —
mêmes règles (consommateurs purs, viser confiné, toggles par couche) :
- `viz/scene.py` : étape **load/grounding** (mesh SMPL posé, squelette, objets, sol, debug grounding).
- `viz/cloud.py` : bake **`point_cloud`** — nuage humain posé par `pose_cloud` (coloré par l'écart de
  parité vs la surface SMPL pleine) + nuages objets posés en monde (rigides). Runnable
  `python -m holov2.viz.cloud --motion-path … --model-dir …`.

Ils ne remplacent pas le viewer `FrameTrace` (qui viendra avec `targets/`) ; ce sont des outils de
debug **par étape**, livrés au fil de l'implémentation.

## Anti-spaghetti
- viewer = 1 responsabilité (afficher), 1 méthode par couche, effets confinés ici.
- lit `FrameTrace` + assets `prepare` via `contracts.py` ; n'appelle aucune logique de calcul
  (sauf le fournisseur `get(f)`, qui est juste `trace_frame` ou un bake).
