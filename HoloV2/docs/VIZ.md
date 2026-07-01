# HoloV2 — `viz/` (visualiseur)

Module **top-level indépendant**, **consommateur pur** (règle d'or 6) : il LIT des artefacts typés et
les affiche. **Zéro hook** dans le calcul, jamais d'`if visualize` dans `prepare`/`targets`/`solve`.
viser est confiné à `core/viser_ops`, aux `layers/`, au `Player` et à `app.py`.

## Fil rouge : Source → VizFrame → Layers (le Player orchestre)

`viz/` possède SON view-model par frame et le construit depuis la sortie publique du pipeline
(`prepare` + `targets` (+ `solve`)). Les couches ne lisent QUE le view-model, jamais les contrats
pipeline.

- **`model.py`** (numpy-only, sans viser/torch) :
  - `VizContext` — assets STATIQUES par scène, fournis à chaque couche au `setup` (noms de canaux,
    `margin`, links de style, faces/parents SMPL, `n_objects`, URDF robot, `has_solve`, SDF du sol).
  - `VizFrame` — UNE frame (`Source.get(i)`) : `pose`, `smpl_verts_world`, `human_cloud_world`,
    `object_clouds_world`, `human_field`, `targets`, `solved | None`.
  - `SolvedFrame` — bundle post-solve (`q`, `object_poses`, points robot FK, transforms de liens,
    `style_achieved`/`contact_achieved`, `cost`/`cost_by_term`/`n_iters`/`status`). `None` tant que
    non résolu → les couches solve se masquent. **Optionnel de bout en bout.**

- **`sources.py`** (numpy-only) — protocole `Source` (`get(i)->VizFrame`, `n_frames`, `context`) :
  - `BakeSource(spec, config, *, solve=False, frame_step, max_frames)` : `prepare` 1×, puis bake
    `trace_frame` + `smpl_verts_world` par frame montrée (playback fluide offline). `solve=False`
    laisse `solved=None` ; `solve=True` (bake `SolveTrajectory` + `Evaluator`) = **phase B**.
  - `LiveSource` : `trace_frame` (+ solve) à la volée — **différé** (même interface plus tard).

- **`core/`** (socle partagé prod + debug) :
  - `colors.py` — `heat_distance` · `diverging` · `parity` · `active_mask` · `AXIS_COLORS` (colormaps
    uniques, ex-dup ×3).
  - `viser_ops.py` — `quat_wxyz_to_R` · `hide` (tue le hack du triangle dégénéré) · wrappers add_*.
  - `layer.py` — protocole `Layer` (`folder`, `setup(server, gui, ctx)`, `update(frame, ui)`) +
    `UiState` (sélecteurs partagés : canal · mode couleur · taille de point).
  - `player.py` — `Player` : folder Playback (slider/play/fps) + Selectors, le `render()` qui itère
    `update` sur les couches, la boucle play/fps + le keep-alive (ex-dup ×4).

## Couches (`layers/`, 1 fichier = 1 toggle)

Portées de l'ancien `viewer.py` : `ground` (lit le **SDF du canal sol** — plan/terrain réel, plus de
box plat), `ghost` (mesh SMPL), `skeleton`, `human_cloud` (coloré par champ), `objects` (nuages +
champ env), `fields` (witness + normales), `style` (points + frames + labels). Chaque couche garde
ses handles persistants (créés au `setup`) et n'affecte que `.points/.colors/.visible` au `update` ;
sa checkbox bascule la visibilité localement.

**Roadmap (phase B+)** : `robot` (G1 résolu, ViserUrdf, lit `solved.q`), `cost_dashboard` (panel 2D),
`contacts`, `correspondence`, `sdf_iso`, `geodesic`. Ajouter une couche = 1 fichier `layers/x.py` +
1 ligne dans la liste de `app.py`.

## Entrée prod : `app.py`

`run_app(spec, *, port, frame_step, max_frames, solve=False)` câble `BakeSource → Player → 7 couches →
run()`. CLI : `python -m src.viz.app --motion-path … --model-dir … [--dataset … --max-frames …]`
(flags partagés via `_scene_args`). `viz.run_app` / `viz.main` sont re-exportés.

## Viewers de debug par étape

`viz/scene.py` · `viz/cloud.py` · `viz/sdf.py` · `viz/hoim3_multiperson.py` — consommateurs purs
focalisés par étape (load / point_cloud / sdf / multi-personne). Leur réécriture sur `core/` est la
**phase 6** de la migration (indépendante des couches prod).

## Tests

`viz` = effets aux extrémités ; on ne teste pas le rendu viser. Helpers purs (`core/colors`,
`viser_ops.quat_wxyz_to_R`/`hide`, `ground_surface_mesh`) → entrée connue/uint8 connu. `BakeSource` →
déterminisme + formes/dtypes + chemin `solved=None` (sur données démo, `max_frames` très bas). Couches
minces → conformité structurelle (`isinstance(layer, Layer)` + `folder`) ; parité visuelle = check
manuel. Tests dans `HoloV2/tests/`, python de l'env `holonew`.

## Anti-spaghetti
- Source → VizFrame → Layers, acyclique : `viz/` importe la sortie publique de prepare/targets/solve ;
  rien n'importe `viz/`.
- viser confiné (core/viser_ops + layers + Player + app) ; `model.py`/`sources.py` numpy-only.
- `targets`/`solve` jamais modifiés (consommateur pur) ; `solved` optionnel partout.
