# Spec — Réévaluation des champs de contact pour `solve` (package online + références)

**Date** : 2026-06-29 · **Étage** : `targets` (surface publique) + `prepare`/`targets` (contrats) · **Statut** : conçu

## Problème

Pendant le solve, la config robot `q` (et, en mode objet-variable, la pose objet `δξ`) change à
chaque itération SQP. Pour que les contacts du robot/objets restent **cohérents avec la scène de
référence** (la démo humaine), `solve` doit, à chaque itération, recalculer pour chaque point de
chaque cloud (robot + objets) la **distance signée** et le **witness** (point le plus proche) par
canal SDF, puis les comparer aux valeurs de référence.

On veut fournir ça **sans dupliquer** la logique d'évaluation déjà présente dans
`targets/interaction/` (règle #8) et **sans casser** les dépendances à sens unique (#1) ni le « pas
de noyau partagé central » (#2).

## Périmètre (ce que ce design livre)

Exactement **2 choses**, et rien de plus :

1. **Les quantités de référence** — pour chaque point de chaque cloud (robot + objets), par canal :
   `distance` + `witness`. Déjà portées par `MultiChannelField` (superset :
   `distance, direction, witness, active`).
2. **Le package de calcul online** des *mêmes* quantités — `pose_cloud` + `eval_fields` (→
   `MultiChannelField`) exposés en **API publique de `targets`**, appelés par `solve` à chaque
   itération sur le cloud robot (posé FK@q) et les clouds objets (posés à leur pose courante).

## Non-objectifs (YAGNI / à l'utilisateur)

- **Les fonctions de coût** qui consomment `(distance, witness)_réf` vs `(distance, witness)_courant`
  (assemblage des résidus type V1 `wd`/`wx` = D/X, activation `α(d_ref)`, normalisation `1/N_k`,
  composition avec les Jacobiennes) — **écrites par l'utilisateur**, hors de ce design.
- **`RobotModel.point_jacobians`** — les Jacobiennes servent les fonctions de coût, pas le package
  d'évaluation (qui n'a besoin que de `link_transforms` pour poser). À ajouter au protocol quand les
  coûts seront codés, pas ici.
- Le reste de l'étage `solve` (backend enfichable, ConstraintProvider, forme canonique du problème) —
  voir le plan SOLVE de `ARCHITECTURE.md`, séparé.
- Aucune relocalisation ni suppression de type existant.

## Idée pivot

`targets` construit le champ de **référence** ; `solve` construit le champ **courant** ; **même
noyau** ⇒ directement comparables.

```
RÉFÉRENCE (targets, online, q-indép.)            COURANT (solve, q-dép., par itération SQP)
 human_cloud ─pose(bones)─► eval ─transport─►       robot_cloud ─pose(FK@q)──► eval ─► champ courant robot
      └─► RobotInteractionTargets.field (d_ref,x_ref)        cur (d0,n0,x0)
 object_clouds ─pose(démo)─► eval ─►                 object_clouds ─pose(pose objet@δξ)─► eval ─► champ courant env
      └─► EnvironmentInteractionTargets.per_object[i]        cur_i
                        │                                                   │
                        └──────────► (coûts utilisateur : résidu réf − courant) ◄─────────┘
```

`targets` = « où sont les contacts dans la démo » ; `solve` = « où sont les contacts du robot
maintenant ». Référence robot = champ humain **transporté** sur les M points (le robot n'existe pas
dans la démo) ; référence objet = éval directe du cloud objet à la pose démo.

## Conformité aux règles d'or (`CLAUDE.md`)

- **#1 (dépendances à sens unique)** : `solve` (aval) importe la surface publique de `targets`
  (`from ..targets import pose_cloud, eval_fields, MultiChannelField`), la référence
  (`from ..targets.contracts import FrameTargets`) et les assets statiques
  (`from ..prepare.contracts import InteractionContext, …`). `targets` n'importe **jamais** `solve`.
  Aucun cycle.
- **#2 (pas de noyau central)** : `targets` **possède** le noyau d'évaluation (il produit le type
  `MultiChannelField`) ; on ne crée aucun module partagé — on **expose** l'existant. Le sous-module
  `targets/interaction/` reste interne ; l'aval n'importe que le **package** (`targets/__init__`),
  jamais un chemin sous-module.
- **#8 (homogénéité, une seule impl)** : `solve` réutilise littéralement `pose_cloud`/`eval_fields` ;
  robot (K=1, parts=liens, FK) et objet (K=1, part=corps) passent par la même op que l'humain (K~4).
- **#3 / #5** : le package reste pur (numpy, torch-free à l'import — le moteur cinématique lourd est
  caché derrière le protocol `RobotModel`, instancié dans `prepare/`). L'instrumentation (`prof.span`)
  reste dans l'orchestrateur de `solve`, pas dans les ops pures.
- **Symétrie validée** : `InteractionContext.robot: RobotModel` calque `GroundedScene.body: BodyModel`
  (moteur de posage protocol-typé, deps lourdes cachées dans l'instance).

## Changements de contrat (minimaux)

Le **livrable 1 (référence) existe déjà** dans `FrameTargets` aujourd'hui
(`robot_interaction.field` pour les M points robot, transporté ; `env_interaction.per_object[i]` pour
chaque cloud objet) ⇒ **zéro changement côté référence**. Les ajouts ci-dessous servent
**uniquement** le livrable 2 (le package online).

### `prepare/contracts.py`

- `InteractionContext` gagne 2 champs, assemblés une fois par `prepare.runner` :
  - `robot_cloud: PointCloud` — les M points de correspondance vus comme nuage **K=1** :
    `parts = link_idx_remappé[:, None]`, `weights = ones((M, 1))`, `offsets = offset_local[:, None, :]`.
    Symétrique à `human_cloud`/`object_clouds`. **Ordre des liens** : `pose_cloud` indexe
    `part_rot[cloud.parts]` où `part_rot = robot.link_transforms(q)[0]` est dans l'ordre de
    `robot.link_names`. Or `correspondence.link_idx` indexe dans `correspondence.link_names`. À
    l'assemblage on **remappe par nom** (`correspondence.link_names[k] → index dans robot.link_names`)
    pour que `robot_cloud.parts` s'aligne sur la sortie FK (V1 résolvait par nom via
    `link_placements(q, corr.link_names)` ; on fige le remap une fois, offline).
    **Invariants** : `robot_cloud.n_points == correspondence.n_points` ; tout nom de
    `correspondence.link_names` existe dans `robot.link_names`.
  - `robot: RobotModel` — le moteur cinématique q-dépendant (protocol-typé numpy-only ; instance
    pinocchio cachée). Instancié dans `prepare/load/robot.py` (règle « robot instancié uniquement
    dans prepare »). `solve` le pose via `robot.link_transforms(q)`.

  (Note d'import : `InteractionContext` reste numpy-only-importable — `RobotModel` est un protocol de
  `prepare/contracts.py`, exactement comme `BodyModel` sur `GroundedScene`.)

### `targets/contracts.py`

- `FrameTargets` gagne les **poses objets de la frame** :
  - `object_rot: np.ndarray  # (N, 3, 3)`
  - `object_pos: np.ndarray  # (N, 3)`

  C'est la part objet de `FramePose` (sans les bones, inutiles au solve). Double usage : (a) **repère
  de canal** passé à `eval_fields` pour la réévaluation robot ; (b) **réf/init** des variables de
  décision objet (la pose courante part de cette valeur et évolue par `δξ`).

  `pipeline._build_frame` remplit ces champs depuis le `FramePose` déjà calculé (zéro recompute).

### `targets/__init__.py`

- Ré-export de la surface publique du noyau : `pose_cloud`, `eval_fields`, `MultiChannelField`.
- Note dans `docs/TARGETS.md` : « ops pures **publiques**, réutilisées par `solve` ; importer depuis
  `targets` (package), jamais depuis `targets.interaction` (sous-module interne) ».

## API du package (ce que `solve` appelle)

```python
from ..targets import pose_cloud, eval_fields, MultiChannelField

# robot, à q courant
pts   = pose_cloud(ctx.robot_cloud, *ctx.robot.link_transforms(q))       # (M, 3) monde
cur   = eval_fields(pts, ctx.channels, obj_rot, obj_pos, ctx.margin)     # MultiChannelField (C, M)

# objet i, à sa pose courante (obj_rot[i], obj_pos[i] = pose courante de l'objet)
pts_i = pose_cloud(ctx.object_clouds[i], obj_rot[i][None], obj_pos[i][None])
cur_i = eval_fields(pts_i, ctx.channels, obj_rot, obj_pos, ctx.margin)   # MultiChannelField (C, P_i)
```

- `cur.distance  # (C, M)` et `cur.witness  # (C, M, 3)` = les 2 quantités, par canal, par point.
- Côté référence : `frame_targets.robot_interaction.field.{distance,witness}` (robot) et
  `frame_targets.env_interaction.per_object[i].{distance,witness}` (objet i).
- `obj_rot`/`obj_pos` initiaux = `frame_targets.object_rot`/`object_pos` ; en mode objet-variable,
  `solve` les met à jour à chaque itération.

### Garantie de contrat (le point critique pour les coûts utilisateur)

Référence et online passent par **le même `eval_fields`**, donc pour un canal donné les deux jeux
`(distance, witness)` sont dans **le même repère** : sol en **monde**, objet en **objet-local** (le
`witness` n'est jamais ramené en monde). Donc `distance_réf − distance_cur` et
`witness_réf − witness_cur` sont comparables **canal-par-canal sans conversion** côté coût.
`direction` (normale unité = `normalize(probe − witness)`) et `active` (masque dans la bande
`margin`) sont fournis en bonus.

## Flux

```
prepare.runner ─► InteractionContext{ …, robot_cloud, robot }      (build-once)
              └─► GroundedScene
targets.pipeline ─► FrameTargets{ style, robot_interaction, env_interaction, object_rot, object_pos }
solve (par itération) ─► pose_cloud + eval_fields(ctx.robot_cloud / ctx.object_clouds, q / poses courantes)
                       ─► (distance, witness) courants  ── vs ──  FrameTargets (référence)  ─► coûts utilisateur
```

## Gestion d'erreurs

- Assemblage `InteractionContext` : `raise ValueError` si `robot_cloud.n_points !=
  correspondence.n_points` ou si un `link_idx` déborde de `correspondence.link_names` (invariant de
  contrat explicite, cf. `MultiChannelField.__post_init__`).
- `FrameTargets` : `object_rot`/`object_pos` ont la même longueur `N` que
  `env_interaction.per_object` (cohérence canal objet ↔ pose).
- `eval_fields` est déjà robuste hors-grille (probes clampés → `active=False`) ; aucun changement.

## Tests (`HoloV2/tests/`, env `holonew`)

- **Unitaire `robot_cloud`** : dérivé de `correspondence` ;
  `pose_cloud(robot_cloud, *robot.link_transforms(q))` reproduit
  `body_pos[link] + R[link] @ offset_local` sur un cas connu (parité V1 `robot_control_points`).
- **Unitaire éval (clouds robot/objet)** : `eval_fields` sur un cloud K=1 posé à une pose connue
  (sol plan + une boîte) → `distance`/`witness` attendus. (Le chemin K~4 humain est déjà couvert ;
  on étend la couverture aux clouds K=1.)
- **Parité V1** : `(distance, witness)` du package sur une frame démo == `query_entities` /
  `contact_field` V1 aux mêmes points monde (tolérance documentée). Verrouille la sémantique des 2
  quantités (le test « portage V1 → V2 »).
- **Cohérence référence ↔ online** : poser le `robot_cloud` à la config qui réalise la démo et
  vérifier que le champ online retombe (à tolérance) sur la référence transportée — sanity du seam.
- **Compile/import** : après changement des `contracts.py`, `python -m py_compile` + un import dans
  l'env (les contrats restent numpy-only-importables).

## Hors périmètre / suites possibles

- `RobotModel.point_jacobians` (Jacobiennes de point) quand les fonctions de coût seront codées.
- Le ConstraintProvider d'interaction (D/X) + le backend enfichable (plan SOLVE).
- Batch vectorisé sur T du package (déjà array-oriented ; optimisation ultérieure).
