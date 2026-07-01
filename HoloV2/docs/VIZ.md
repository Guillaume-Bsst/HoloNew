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

## Visualiseurs de debug incrémentaux (par étape) — réécrits sur `viz/core`

Des viewers focalisés valident chaque étape (consommateurs purs, viser confiné à `core/viser_ops` +
le viewer, socle `core/` partagé : Player/colors/viser_ops). Ils gardent le droit de **piloter les
internes de l'étage** qu'ils visualisent (exception ARCHITECTURE.md) :

- `viz/debug/scene.py` : étape **load/grounding** (mesh SMPL posé, squelette, objets, sol, debug
  grounding : offsets, slider foot-percentile, marqueurs point-bas).
  `python -m src.viz.debug.scene`.
- `viz/debug/cloud.py` : bake **`point_cloud`** — nuage humain posé par `pose_cloud` (coloré par la
  parité vs surface SMPL pleine, `core.colors.parity`) + nuages objets rigides.
  `python -m src.viz.debug.cloud`.
- `viz/debug/sdf.py` : build **`prepare/sdf`** — tranche/bande/witness, `core.colors.diverging`
  (parser propre `--mesh`/`--plane`, pas de frames donc pas de Player).
  `python -m src.viz.debug.sdf`.
- `viz/debug/hoim3.py` : **load multi-personnes** HOI-M3 (toutes les personnes + objets).
  `python -m src.viz.debug.hoim3`.
- `viz/debug/_args.py` : glue CLI partagée (ex-`_scene_args.py`) ; `viz/debug/_geometry.py` :
  helpers géométriques purs testés (point-bas, surface barycentrique de parité, node_coords).

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

## Seam solve : couche `robot` + panel `cost_dashboard` (phase B)

Le viz consomme aussi `solve/` (optionnel). `BakeSource(spec, config, solve=True)` exécute
`solve.runner.solve` une fois sur la séquence montrée, construit un `targets.Evaluator` une fois, et
remplit `VizFrame.solved` par frame via `build_solved_frame(traj, ev, ctx, f)` (pur, numpy-only) :

- `q` / `object_poses` / `cost`/`cost_by_term`/`n_iters`/`status` : lus directement dans
  `SolveTrajectory[f]` (+ `FrameInfo[f]`).
- `style_achieved = ev.style(q)`, `contact_achieved = ev.contacts(q, object_rot, object_pos)` :
  l'« atteint » réutilise l'`Evaluator` (recompute prouvé dans `test_solve_runner`), aucune
  nouvelle logique de retargeting. `object_poses` (wxyz) → matrices via scipy.
- `robot_points_world = pose_cloud(robot_cloud, R, t)` et `link_transforms (L,4,4)` : FK @ q via
  `ctx.robot`.

Sans `solve` (`solve=False`), `VizFrame.solved is None` et tout le pré-solve marche ; les
consommateurs solve se masquent. Couches/panels concernés :

- **`layers/robot.py` (`RobotLayer`)** : le G1 résolu en meshes complets via
  `viser.extras.ViserUrdf` (yourdfpy). `update_cfg(q[7:7+dof])` + pose de base `q[:3]` /
  `q[[6,3,4,5]]` (le `q` pinocchio porte le quat en xyzw sur `q[3:7]` ⇒ réordonné en wxyz pour
  viser). No-op/masquée quand `frame.solved is None`. URDF depuis `VizContext.robot_urdf_path`.
- **`panels/cost_dashboard.py` (`CostDashboard`)** : panel 2D (matplotlib `Agg` → image
  viser, car plotly absent de l'env) — `cost_by_term` empilé + `cost` total sur toutes les frames,
  + marqueurs/tableau des frames non convergées (`status`/`n_iters`). Lit `solved.*` sur toute la
  séquence (fourni par la Source au `setup`).

`app.py --solve` câble `BakeSource(solve=True)` + `RobotLayer` + `CostDashboard`. Invariant tenu :
`viz` n'importe que les surfaces publiques (`solve.runner`/`solve.contracts`, `targets.Evaluator`) ;
`targets`/`solve` restent inchangés.

## Couches d'interaction (roadmap #3 / #4 / #6 / #7)

Couches composables ajoutées au viewer prod (`app.py`), une par fichier `layers/`, lisant le SEUL
view-model. Géométrie d'affichage pure (testée en unitaire) + classe `Layer` mince (handles viser
persistants). Les couches contact/correspondance sont **solve-gated** (no-op si `frame.solved is None`).

| Couche | Fichier | Donnée (view-model) | Solve-gated |
|---|---|---|---|
| **Contacts robot** (#3) | `layers/contacts.py` · `ContactsLayer` | cible `targets.robot_interaction.field` vs atteint `solved.contact_achieved.field` sur `solved.robot_points_world` **+ witness cible/atteint** (lignes point→surface, mappées via la pose objet **résolue** du canal — les deux) | oui |
| **Contacts objets** | `layers/object_contacts.py` · `ObjectContactsLayer` | cible `targets.env_interaction.per_object[k]` ET atteint `solved.contact_achieved.env[k].field`, **toutes deux sur le cloud RÉSOLU** (`object_cloud_solved`) — seul le CHAMP diffère ; **witness** via la pose **résolue** de l'objet du **canal** (pas l'objet-probe) | oui |
| **Correspondance SMPL↔G1** (#4) | `layers/correspondence.py` · `CorrespondenceLayer` | `human_cloud_world[ctx.correspondence.smpl_idx]` → `solved.robot_points_world` | oui |
| **SDF iso (surface)** (#6) | `layers/sdf_iso.py` · `SdfIsoLayer` | bande `|d|<band` des `ctx.channels[c].sdf`, posée par `frame.pose` | non |
| **Champ géodésique** (#7) | `layers/geodesic.py` · `GeodesicLayer` | `ctx.channels[c].geodesic` (points/normales + heat mono-source `geo[src]`), posée par `frame.pose` | non |

Assets statiques consommés : `VizContext.channels` (chaque `Channel` porte `sdf` + `geodesic` +
`object_idx`) et `VizContext.correspondence` — ajoutés au view-model et peuplés par `BakeSource` depuis
l'`InteractionContext` de `prepare`.

**Debug solve — état source vs résolu (les objets sont des variables de l'optimiseur).** La couche
`objects` gagne un sous-toggle **« cloud objet résolu »** (posé à `solved.object_poses`, vert vs source
orange, via `object_cloud_solved`) → on voit de combien l'optimiseur a déplacé chaque objet. La vue
montre la **scène résolue** : `contacts` (robot) et `object_contacts` affichent **cible ET atteint sur
la même géométrie résolue** (seul le CHAMP diffère : cible = champ cible, atteint = champ atteint) — une
ligne witness part toujours d'un point du nuage affiché (résolu). Helpers purs partagés :
`layers/_contact_ops.py` (`object_cloud_solved`, `witness_segments`). Aucune donnée nouvelle plombée —
`solved.contact_achieved.env[k]` (contact objet atteint) était déjà dans le view-model.

**Roadmap #5 « activité des contraintes » : hors périmètre / BLOQUÉ.** Afficher le slack par-contrainte
exige que `solve/` l'exporte dans `Step`/`FrameInfo` (changement de contrat `solve`, pas une tâche viz) ;
tant que `SolvedFrame` ne porte pas ce slack, #5 ne peut pas être une couche.
