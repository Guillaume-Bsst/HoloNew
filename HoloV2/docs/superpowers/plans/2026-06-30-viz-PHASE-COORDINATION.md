# Viz redesign — PHASE COORDINATION (contrat inter-plans, autoritaire)

> **Autorité** : ce document fige le contrat partagé entre les 4 plans de phase
> (`viz-foundation`, `viz-solve-robot`, `viz-interaction-layers`, `viz-debug-rewrite`).
> **En cas de conflit entre un plan de phase et ce contrat, CE CONTRAT GAGNE.** Les plans de phase
> ont été rédigés en parallèle ; ce doc réconcilie leurs hypothèses croisées. À lire AVANT
> d'exécuter une phase, et entre chaque phase.

Spec de référence : `docs/superpowers/specs/2026-06-30-viz-architecture-design.md`.

## 1. Ordre d'exécution & dépendances

Implémentation **séquencée** (pas en parallèle : B/C/D éditent tous `app.py` + `layers/`) :

```
A (foundation)  →  B (solve+robot)  →  C (couches interaction)  →  D (debug rewrite)
   prérequis de tout      qpos/SolvedFrame      contacts/corr (besoin B)   ne dépend que de A (core/)
```

- **A** doit être mergé+vert avant toute autre phase (définit `core/`, `model`, `sources`, `app`).
- **B** dépend de A. **C** dépend de A **et** B (ses couches `contacts`/`correspondence` lisent
  `SolvedFrame`). **D** ne dépend que de A (`core/`), mais on le place en dernier pour éviter les
  conflits d'édition sur `app.py`.
- Gate de revue entre chaque phase (subagent-driven : un sous-agent par tâche, revue entre tâches).

## 2. API canonique de la foundation (ce que B/C/D CONSOMMENT — ne pas redéfinir)

Référence d'implémentation : `viz-foundation`. Signatures gelées :

```python
# viz/model.py  (frozen, numpy-only — aucun import viser/torch)
VizContext(channel_names, margin, style_link_names, smpl_faces, smpl_parents, n_objects,
           robot_urdf_path, has_solve, ground_sdf,
           channels, correspondence)          # ← channels + correspondence AJOUTÉS (voir §3 Phase A)
VizFrame(pose, smpl_verts_world, human_cloud_world, object_clouds_world, human_field, targets, solved)
SolvedFrame(q, object_poses, robot_points_world, link_transforms, style_achieved, contact_achieved,
            cost, cost_by_term, n_iters, status)     # solved=None tant que non résolu

# viz/core/layer.py
UiState(channel: str, color_mode: str, point_size: float)              # frozen
class Layer(Protocol): folder: str; setup(server, gui, ctx: VizContext); update(frame, ui: UiState)

# viz/core/colors.py
heat_distance(dist, margin) · active_mask(active) · diverging(signed, vmax) · parity(err, vmax) · AXIS_COLORS

# viz/core/viser_ops.py
quat_wxyz_to_R(quat) · hide(handle)
# + AJOUTS §3 Phase A : line_segments(...) · point_cloud(...) · label(...) (wrappers minces, anti-dup)

# viz/core/player.py
class Player(source, layers: list[Layer], *, port=8080)   # PROD : possède Playback+Selectors, render(), play loop
play_loop(server, *, n_frames, render, fps_default=20)    # ← AJOUT §3 Phase A : shell générique pour les viewers debug

# viz/sources.py
class Source(Protocol): get(i) -> VizFrame ; n_frames: int ; context: VizContext
class BakeSource(spec, config, *, solve=False, frame_step=2, max_frames=200)  # .get(i)/.n_frames/.context
# accès « toutes les frames résolues » (cost dashboard) = [src.get(i).solved for i in range(src.n_frames)] — pas d'accesseur spécial

# viz/app.py
run_app(spec, *, port=8080, frame_step=2, max_frames=200, solve=False)  # B ajoute le câblage solve
```

## 3. Deltas par phase (corrections à appliquer pendant l'exécution)

### Phase A — foundation (ajouts au plan tel qu'écrit)
1. **`VizContext` : ajouter deux champs** consommés par C/B :
   - `channels: tuple[Channel, ...]` — chaque `Channel` porte `sdf`+`geodesic`+`object_idx`
     (couvre `sdf_iso` #6, `geodesic` #7, et le lift local→monde des canaux). **Remplace
     fonctionnellement `ground_sdf`** (= `channels[0].sdf`) ; garder `ground_sdf` est toléré mais
     redondant — la couche `ground` peut lire `channels[0].sdf`.
   - `correspondence: CorrespondenceTable` — `smpl_idx`/`link_idx` pour les lignes SMPL↔G1 (#4).
   - `BakeSource` les remplit direct depuis `InteractionContext` (zéro recompute) ; ajouter un test
     de forme/déterminisme.
2. **`core/player.py` : exposer `play_loop(server, *, n_frames, render, fps_default=20)`** — le shell
   générique (folder Playback + slider/play/fps + keep-alive + dispatch `render(i)`) que **D** réutilise.
   `Player` (prod) peut l'utiliser en interne ou rester tel quel ; l'essentiel est que le shell
   générique existe et soit importable par `debug/`.
3. **`core/viser_ops.py` : ajouter `line_segments(...)`, `point_cloud(...)`, `label(...)`** (wrappers
   minces sur viser) consommés par les couches prod ET les viewers debug — sinon la dup revient par
   la bande. (`quat_wxyz_to_R`/`hide` déjà prévus.)
4. **`core/geometry.py` (nouveau, numpy-only) : `node_coords(sdf)`** — la conversion grille→coords
   portée de `sdf.py`. Partagée par la couche prod `sdf_iso` (C) ET `debug/sdf.py` (D), pour éviter
   le doublon `iso_band_points`(C)/`node_coords`(D). (Si trivial, le mettre dans `viser_ops` convient.)
5. Convention de toggle (déjà dans le plan A, à généraliser) : **chaque couche câble SA checkbox en
   local dans `setup` → bascule `.visible` de son handle persistant**. Le `render()` du Player n'est
   relié qu'au slider + sélecteurs. Pas de re-render sur toggle (le handle persiste).

### Phase B — solve+robot
- **Evaluator** : l'API est `contacts(q, object_rot, object_pos)` (PAS `contacts(q, poses)`).
  `SolveTrajectory.object_poses` est `(N,7)` wxyz → convertir wxyz→matrices (motif `viz/scene.py`).
- **Quaternion base robot** : pinocchio `q[3:7]` = xyzw, la base viser veut wxyz → `RobotLayer`
  réordonne `q[[6,3,4,5]]`. (Risque correctness #1 — tester/vérifier visuellement.)
- **Cost dashboard** : plotly ABSENT de l'env → matplotlib `Agg` + `gui.add_image`. Itère
  `src.get(i).solved` pour toutes les frames (pas d'accesseur spécial sur `BakeSource`).
- `robot_points_world = pose_cloud(ctx.robot_cloud, *ctx.robot.link_transforms(q))` — zéro nouvelle
  logique de retargeting (lit `SolveTrajectory` + réutilise l'Evaluator).

### Phase C — couches interaction
- **SUPPRIMER la Task 0** du plan C (« ajouter channels+correspondence à VizContext ») : **A les
  fournit déjà** (§3 Phase A). Consommer `ctx.channels` / `ctx.correspondence` directement.
- **`node_coords`/iso** : consommer `core.geometry.node_coords` (§3 Phase A #4), ne PAS recopier une
  variante privée.
- `contacts` (#3) et `correspondence` (#4) restent **solve-gated** (no-op si `frame.solved is None`).
- Toggles : suivre la convention §3 Phase A #5 (checkbox locale → `.visible`), pas « Player re-render ».
- `#5 contraintes` = **hors-scope** (bloqué : exige l'export du slack par-contrainte dans
  `solve.Step`/`FrameInfo` — changement de contrat solve, pas une tâche viz).

### Phase D — debug rewrite
- **Player** : les viewers debug ne produisent PAS de `VizFrame` → utiliser **`play_loop(...)`**
  (le shell générique, §3 Phase A #2), **pas** `Player(source, layers)`. Garder leurs données
  par-frame bespoke (intermédiaires hors-contrat : grounding live, parité, slices SDF).
- **Déplacement `_args`** : `_scene_args.py → debug/_args.py` doit **repointer `app.py`** (et tout
  importeur), **pas `viewer.py`** (supprimé par A). Vérifier qu'aucun `from .._scene_args` ne subsiste.
- **`node_coords`** : consommer `core.geometry.node_coords` (partagé avec C), ne pas le garder en
  privé dans `debug/_geometry.py`. Le reste de `_geometry.py` (lowz/parité/surface) reste local à debug.
- `sdf.py` frameless : OK de ne pas avoir d'axe temps ; utiliser `play_loop` n'est pas requis pour
  lui (garder son keep-alive 2 lignes ou un helper `core` de keep-alive).
- Réutiliser `core/colors` + `core/viser_ops` (zéro player/colors/quat/hide redupliqué).

## 4. Convention globale (TOUTES les phases)

- **Commentaires + docstrings en ANGLAIS** dans le code (`src/`), conformément au CLAUDE.md mis à
  jour le 2026-06-30 (« rédigés en anglais … JAMAIS [de français] dans le code »). Le français reste
  pour ce doc, les specs/plans (prose), les messages de commit et les échanges. → **Plan C : traduire
  en anglais les commentaires/docstrings de ses blocs de code** (il avait choisi le français).
- Noms de symboles en anglais (déjà le cas).
- Contrats (`model.py`/`sources.py`) **numpy-only** : aucun import viser/torch.
- `viz` = **consommateur pur** : viser confiné à `core/viser_ops` + `layers/` + `panels/` + `app.py`
  + `debug/`. `targets`/`solve` **inchangés**.
- Tests dans `HoloV2/tests/`, lancés avec le python de l'env `holonew`, `max_frames` très bas.
- Commits conventionnels, **NE JAMAIS tagger Claude** (aucun `Co-Authored-By`, aucune mention
  Claude/Anthropic). Auteur : `Guillaume-Bsst`.
```
