# Spec — Module `viz/` : redesign architecture (couches composables + seam solve)

**Date** : 2026-06-30 · **Module** : `viz/` (consommateur, hors pipeline) · **Statut** : conçu

Redesign **complet** du module de visualisation. Motivation : l'audit du 2026-06-30 (cf. mémoire
`viz-audit-roadmap`) a montré que le viz actuel couvre proprement le côté **humain/cibles
pré-solve** mais (1) **n'affiche nulle part le robot G1 résolu** ni aucune sortie de `solve/`
(le produit final du pipeline), et (2) duplique massivement entre le viewer prod et les 4 viewers
debug (player ×4, colormaps ×3, conversion quat ×3, hack de masquage). VIZ.md, écrit avant
l'existence de `solve`, est en partie périmé.

But du redesign : un **socle partagé** + des **couches composables** lisant un **view-model
possédé par viz**, où le robot résolu et les diagnostics solve deviennent des couches comme les
autres, et où ajouter une couche = un fichier. On conçoit **à neuf** dans l'idiome
prepare/targets/solve — la V1 (`../HoloNew/src/viewer.py`, classe-dieu ~1000 l. avec ViserUrdf,
`draw_q`, `_draw_g1_points`, centroidal) reste une **réf. pour porter la LOGIQUE de rendu**, jamais
la structure.

## Décisions cadrées (entrées de ce design)

| Décision | Choix |
|---|---|
| Périmètre | **Redesign complet** : unifier viewer prod + viewers debug sous un même framework, accueillir toute la roadmap de couches. |
| Seam solve | **Source unifiée** : viz consomme `prepare`+`targets`+`solve` et construit SON view-model par frame (`VizFrame` bundle `{trace, solved\|None}`). Les couches ne lisent QUE le view-model, jamais les contrats pipeline. Solve **optionnel**. (Rejeté : étendre `FrameTrace` → violerait la dépendance amont `targets`→`solve`.) |
| Viewers debug | **Socle partagé, entry points distincts** : le framework unifié sert le chemin PROD ; les viewers debug restent des runnables séparés mais **réécrits sur `core/`**, gardant le droit de piloter les internes d'étage pour montrer des intermédiaires hors-contrat. |
| Mécanisme de couche | **Couches = objets stateful** (`Layer` protocol : `setup` une fois + `update` par frame). (Rejeté : registry fonctionnel — état en closures fragile ; drawables déclaratifs — YAGNI, viser fait déjà l'overwrite par nom.) |
| Rendu robot | **`ViserUrdf`/yourdfpy** (`update_cfg(q)` + pose base) — meshes complets, code minimal, comme V1. |
| Cost dashboard | **Panel 2D** séparé des couches 3D (concept `panels/`). |
| Règle d'or | `viz` reste un **consommateur pur** (golden rule 6) : viser confiné à `core/viser_ops` + couches ; aucune logique de calcul du pipeline ici (sauf la `Source`, qui orchestre prepare/targets/solve). `targets`/`solve` jamais modifiés. |

## Arborescence cible

```
viz/
  __init__.py        # re-exports : entrée prod (run_app / view) + sources
  core/
    player.py        # Player : folder Playback (slider/play/fps) + boucle render + thread keep-alive   [tue dup ×4]
    colors.py        # colormaps uniques : heat_distance, diverging, parity, active_mask, axis           [tue dup ×3]
    viser_ops.py     # helpers viser confinés : hide(handle), line_segments, quat wxyz→R, label, cloud   [tue hack triangle]
    layer.py         # protocole Layer : setup(server, gui, ctx) / update(frame, ui)
  model.py           # VizContext (statique) · VizFrame (par-frame) · SolvedFrame (post-solve)
  sources.py         # protocole Source : get(i)->VizFrame, n_frames, context ;  BakeSource / LiveSource
  layers/            # couches prod composables (1 fichier, 1 toggle chacune)
    ground · ghost · skeleton · human_cloud · objects · fields · style        (portées de viewer.py)
    robot · contacts · correspondence · sdf_iso · geodesic                     (roadmap, une à une)
  panels/
    cost_dashboard.py  # panneau 2D : cost_by_term empilé + status/n_iters — pas une couche 3D
  app.py             # viewer PROD unifié : Player + Source + couches sélectionnées + panels
  debug/             # viewers debug par étape, RÉÉCRITS sur core/
    scene · cloud · sdf · hoim3 · _args.py     (_args.py = ex-_scene_args.py)
```

Fil rouge : **Source → VizFrame → Layers**, le `Player` orchestre, `core/` est partagé par prod
**et** debug. Acyclique : `viz/` importe la sortie publique de prepare/targets/solve ; rien
n'importe `viz/`.

## Contrat du view-model (la seam)

Deux niveaux — encode la règle « statique vs dynamique » (et supprime le smell perf des couches
lignes re-`add` à chaque tick : le statique est posé une fois au `setup`).

**`VizContext`** (statique, fourni par la `Source` au `setup` des couches) :
```
channel_names: tuple[str,...] · margin: float · style_link_names: tuple[str,...]
smpl_faces: (F,3) · smpl_parents: (J,) · n_objects: int
robot_urdf_path: Path · has_solve: bool
```

**`VizFrame`** (par-frame, `Source.get(i)`) — `frozen`, numpy-only :
```
pose: FramePose                      # bone_pos/rot, object_rot/pos              (de FrameTrace)
smpl_verts_world: (V,3) | None       # mesh SMPL posé pour la couche ghost (None si non paramétrique)
human_cloud_world: (N,3)
object_clouds_world: tuple[(P,3),...]
human_field: MultiChannelField
targets: FrameTargets                # style, robot_interaction, env_interaction
solved: SolvedFrame | None           # None tant que non résolu → couches solve no-op
```

(`smpl_verts_world` = ce que `viewer.py` précalcule aujourd'hui via `body.posed_vertices` ; la
`BakeSource` le calcule dans le même passage que `trace_frame`. Les `smpl_faces` sont statiques →
`VizContext`.)

**`SolvedFrame`** (bundle post-solve, construit par `BakeSource` depuis `SolveTrajectory` +
`targets.Evaluator`) :
```
q: (nq,)                             # config robot résolue                (SolveTrajectory.qpos[f])
object_poses: (N,7)                  # objets résolus                      (SolveTrajectory.object_poses[f])
robot_points_world: (M,3)            # points de correspondance G1 placés par FK @ q
link_transforms: (L,4,4)             # placements des liens (FK) — pour correspondance/contact
style_achieved: StyleEval            # atteint @ q              (ev.style(q))
contact_achieved: ContactEval        # contact atteint @ q,poses (ev.contacts(q, poses))
cost: float · cost_by_term: dict[str,float] · n_iters: int · status: str    (FrameInfo[f])
```

Point clé : `solved` est **optionnel**. `BakeSource(spec, solve=False)` produit des `VizFrame` à
`solved=None` → tout le pré-solve marche sans solver, les couches robot/contacts/correspondence se
masquent. Le recompute « atteint » est exactement celui prouvé dans `tests/test_solve_runner.py:69`
(`ev.style(traj.qpos[f]).position`).

**Espace géométrique** : `VizFrame` documente son espace. Quand `solved` est présent, la scène est
en **espace solve/scene-scaled** (cohérente avec ce que le solver reçoit) ; la `BakeSource` fixe ce
choix en un seul endroit — résout le smell « objets colorés par un champ scaled mais dessinés
non-scalés » de l'audit.

## Flux de données

```
Source ──get(i)──> VizFrame ──> [Layer.update(frame, ui)]*  +  [Panel.update(...)]*
  │   (+ VizContext statique au setup des couches)                ▲
  ├── BakeSource : prepare 1× → pour chaque frame montrée :       │
  │     trace_frame(grounded, ctx, robot, f)             → côté trace
  │     [si solve] q,poses,info = SolveTrajectory[f] ;            │
  │                ev.style(q)/ev.contacts(q,poses) ; FK  → SolvedFrame
  │     bundle → VizFrame                                          │  (offline, playback fluide)
  └── LiveSource : trace_frame (+ solve) à la volée — même interface  (plus tard)
```

`app.py` câble : construit la `Source`, le `Player`, instancie la liste de couches + panels,
`run()`. Ajouter une couche roadmap = 1 fichier `layers/x.py` + 1 ligne dans la liste de `app.py`.

## Protocole de couche

```python
class Layer(Protocol):
    folder: str                                              # son groupe GUI
    def setup(self, server, gui, ctx: VizContext) -> None:   # handles statiques + SA checkbox, 1×
        ...
    def update(self, frame: VizFrame, ui: UiState) -> None:  # maj handles depuis le view-model
        ...
```

- `UiState` = sélecteurs partagés en lecture seule (`channel · color_mode · point_size`), passés
  par le `Player` ; les couches ne se touchent jamais entre elles.
- `Player` possède le folder Playback + le `render()` qui itère `update` sur couches/panels + la
  boucle play/fps + le keep-alive.
- Couche sans donnée solve (`frame.solved is None`) → no-op/masquée. Solve optionnel de bout en bout.
- Une couche garde ses handles persistants (créés au `setup`) et ne fait qu'affecter
  `.points/.colors/.visible` en `update` — pas de re-`add` par tick (sauf cas justifié, ex. ghost
  dont les verts changent).

## Couches & panels (roadmap, par priorité)

| Couche/panel | Source de données (`VizFrame`) | Priorité |
|---|---|---|
| `ground` (plan/terrain réel via `ctx`/SDF) | `pose`, SDF du canal ground | port |
| `ghost` (mesh SMPL) | verts SMPL (context/trace) | port |
| `skeleton` | `pose.bone_pos` | port |
| `human_cloud` (coloré par champ) | `human_cloud_world`, `human_field` | port |
| `objects` (clouds + champ env) | `object_clouds_world`, `targets.env_interaction` | port |
| `fields` (witness + normales) | `human_field` (+ `pose.object_*` pour local→world) | port |
| `style` (points + frames + labels) | `targets.style` | port |
| **`robot`** (G1 résolu, ViserUrdf) | `solved.q` / `object_poses` | **roadmap #1** |
| **`cost_dashboard`** (panel 2D) | `solved.cost_by_term/cost/status/n_iters` sur toutes les frames | roadmap #2 |
| **`contacts`** (atteint vs cible sur points robot) | `targets.robot_interaction.field` vs `solved.contact_achieved` | roadmap #3 |
| **`correspondence`** (lignes SMPL↔G1 @ q) | `CorrespondenceTable` (context) + `solved.robot_points_world` | roadmap #4 |
| `sdf_iso` (iso-surface dans le viewer prod) | SDF des canaux (context) | roadmap #6 |
| `geodesic` (champ géodésique) | `GeodesicTable` (context) | roadmap #7 |

(Roadmap #5 « activité des contraintes » = **bloqué** : nécessite d'abord d'exporter le slack
par-contrainte dans `solve` (`Step`/`FrameInfo`) — changement de contrat hors de ce redesign viz.)

## Plan de migration (incrémental — chaque étape reste runnable)

| # | Étape | Détail | Résultat |
|---|---|---|---|
| 1 | `core/` | extraire player + colors + viser_ops + layer des bouts dupliqués ; zéro changement de comportement | socle + tests unitaires colors |
| 2 | `model.py` + `sources.py` | `VizContext`/`VizFrame`/`SolvedFrame` ; `BakeSource(solve=False)` d'abord | seam posée, pré-solve seul |
| 3 | porter couches prod | `ground/ghost/skeleton/human_cloud/objects/fields/style` ← `viewer.py` → `layers/`, lisant `VizFrame` ; câbler `app.py` | `app.py` ≡ `viewer.py` actuel → **supprimer `viewer.py`** |
| 4 | solve + couche `robot` | `BakeSource` branche `SolveTrajectory` + `Evaluator` → `SolvedFrame` ; couche `robot` (ViserUrdf) | **1ʳᵉ valeur neuve : robot résolu superposé** |
| 5 | couches roadmap | `cost_dashboard` → `contacts` → `correspondence` → `sdf_iso` → `geodesic`, une par une | roadmap déroulée |
| 6 | réécrire debug | `scene/cloud/sdf/hoim3` → `debug/` sur `core/` ; `_scene_args.py` → `debug/_args.py` | dette debug purgée (indépendant de 3-5, peut suivre 1) |

**Dette résorbée** : colormaps ×3 → `core/colors` ; playback ×4 + keep-alive → `core/player` ;
quat wxyz→R ×3 → `core/viser_ops` ; hack triangle-dégénéré → `viser_ops.hide()` ; sol box plat →
couche `ground` lit le SDF du canal (plan/terrain réel) ; couleur/échelle objets → espace fixé par
`BakeSource` ; couches lignes re-`add` → handles persistants. Puis **VIZ.md réécrit** pour refléter
la nouvelle archi + la section viz solve.

## Tests

`viz` = effets aux extrémités ; on ne teste pas le rendu viser. Stratégie :
- **Helpers purs unitaires** : `core/colors` (entrée connue → uint8 connu — porte la logique
  `_heat_distance`/`_diverging`/parité existante), conversion quat (quat connu → R connue),
  `hide()` renvoie une géométrie dégénérée.
- **Déterminisme des sources** : `BakeSource.get(i)` build ×2 identique ; formes/dtypes des champs
  `VizFrame` ; **les deux chemins** `solved=None` vs `solved` présents. Sur la scène démo avec
  `max_frames` très bas (cf. mémoire `run-tests-low-max-frames`).
- **Couches minces** : `update()` = quasi que de l'affectation de handles depuis `VizFrame`, zéro
  calcul → rien à tester côté couche ; le calcul d'« atteint » vit dans `Evaluator` (déjà testé) +
  `BakeSource`. On garde les couches assez fines pour ne pas avoir à les tester.
- Règles CLAUDE.md : tests dans `HoloV2/tests/`, python de l'env `holonew`, `max_frames` bas.

## Invariants / garde-fous

- `viz/` n'importe que des **sorties publiques** (`prepare.runner`, `*.contracts`,
  `targets.pipeline.trace_frame`, `targets.Evaluator`, `solve.runner`/`solve.contracts`) ; jamais un
  sous-module interne du pipeline (sauf les viewers `debug/`, qui pilotent délibérément les internes
  de l'étage qu'ils visualisent — exception ARCHITECTURE.md).
- viser est importé **uniquement** dans `core/viser_ops`, les `layers/`, `panels/`, `app.py` et
  `debug/` ; jamais dans `model.py`/`sources.py` (numpy-only, importables/testables sans écran).
- `targets` et `solve` **inchangés** par ce redesign (consommateur pur).
- `SolvedFrame` ne recalcule rien que le pipeline produit déjà : il **lit** `SolveTrajectory` et
  **réutilise** `targets.Evaluator` pour l'« atteint » — pas de nouvelle logique de retargeting.
