# Spec — Seam `targets → solve` : références + évaluateur (géométrie + jacobiennes)

**Date** : 2026-06-29 · **Étages** : `targets` (surface publique, réorg) + `prepare` (contrat `RobotModel`, impl FK) · **Statut** : conçu

> **Remplace** `2026-06-29-solve-field-eval-package-design.md` (socle « valeur seule », jacobiennes
> différées, sans réorg). Le présent spec **réutilise** ce socle déjà codé (kernels publics,
> `InteractionContext.robot_cloud`/`robot`, `FrameTargets.object_rot/pos`) et y **ajoute** : un
> évaluateur-objet exposé par `targets`, des **jacobiennes analytiques**, la méthode
> `RobotModel.link_jacobians`, et la **réorganisation** de `targets/` autour de la symétrie
> référence / évaluation.

## Problème

Pendant l'optimisation, les variables de décision changent à chaque itération :
**`q` du robot (base flottante incluse) et la pose `SE(3)` de chaque objet**. Le solveur a besoin, à
chaque itération, de :

1. l'**état géométrique courant** du robot et des objets (positions/rotations des links, et
   distance/witness/normale des nuages vs les canaux SDF), et
2. les **jacobiennes** de ces quantités vis-à-vis des variables de décision, pour linéariser ses
   résidus (style GMR Gauss-Newton / contacts SOCP).

`targets` doit fournir **(a)** les **références** q-indépendantes (déjà en place) et **(b)** un
**évaluateur** qui transforme une config en (état courant + jacobiennes) — **sans** soustraire la
référence ni façonner un coût. Le tout sans dupliquer les noyaux d'évaluation ni casser les
dépendances à sens unique.

## Décisions cadrées (entrées de ce design)

| Décision | Choix retenu |
|---|---|
| But | Définir l'API solveur (le seam `targets ↔ solve`) |
| Dérivées | **Jacobiennes analytiques fournies** par les packages (pas d'autodiff, pas de FD en prod) |
| Config robot | **q free-flyer pinocchio** : `q = [pelvis(7: pos+quat), angles_joints]`, tangente `v` de dim `nv = 6 + n_joints` |
| Objets | Variables **`SE(3)`** (tangente locale 6 : lin + ang), même convention que la base flottante |
| Placement / forme | **B — évaluateur exposé par `targets`** (objet `Evaluator`), appelé par `solve` |
| Réorganisation | **Réorg complète** de `targets/` (symétrie référence / évaluation) |
| Frontière éval ↔ coût | L'évaluateur rend **géométrie courante + jacobiennes**, JAMAIS un résidu/coût. Résidus, poids, cônes, activation : **dans `solve`** |

## Périmètre

**Ce que ce design livre :**

1. **Références** (q-indép., per-frame) — inchangées sur le fond, juste rangées : `StyleTargets`
   (**géométrie seule** : pos/rot cible par link ; les **poids de tracking sont déjà retirés vers
   `solve`**, commit `02ef94b`) ; contact refs (champ humain transporté + champs objets).
2. **Évaluateur** (q-dép.) — `Evaluator(ctx)`, **construit une fois** (assets statiques), exposant
   `.style(q) → StyleEval` et `.contacts(q, object_poses) → ContactEval`, chacun = **état courant +
   jacobiennes géométriques**.
3. **Contrat cinématique** : `RobotModel.link_jacobians(q)` (free-flyer) + impl pinocchio dans
   `prepare/load/robot.py`.
4. **Réorganisation** de `targets/` (ci-dessous).

**Non-objectifs (→ `solve`, écrits par l'utilisateur) :**

- Les **fonctions de coût / résidus** (`courant − référence`, activation `α(d_ref)`, poids `w_p`/`w_r`,
  D/X contact, normalisation, cônes SOCP, composition avec les jacobiennes).
- La **sémantique du résidu de contact** (matcher la distance de réf vs forcer le contact) — décision
  de modélisation côté `solve`.
- Le reste de l'étage `solve` (backend, ConstraintProvider, forme canonique du problème).

## Idée pivot — la symétrie référence / évaluation

`targets` fait **une seule** opération conceptuelle — *poser une configuration et lire son état
style + contact* — appliquée à **deux** configs :

| | Config posée | Quand | q ? | Jacobienne ? | Produit |
|---|---|---|---|---|---|
| **Référence** (source) | humain/démo (bones SMPL + nuage humain) | 1×/frame, bakée | non | non | `StyleTargets` + contact refs (witness/normale/dist transportés) |
| **Évaluation** (cible) | robot @ `q` + objets @ `SE(3)` | par itération | oui | **oui** | `StyleEval` / `ContactEval` (état courant + jacobiennes) |

Le résidu (`cible − référence`) et le coût se font **dans `solve`** à partir de ces deux sorties. Les
**mêmes noyaux purs** (`pose_cloud`, `eval_fields`, géodésique) servent les deux côtés ; seul le côté
cible porte une jacobienne.

```
RÉFÉRENCE (targets.pipeline, q-indép.)        ÉVALUATION (targets.Evaluator, q-dép., par itération)
 human_cloud ─pose(bones)─►eval─transport─►      robot_cloud ─pose(FK@q)──►eval─► field cur + point_jac
   └─►RobotInteractionTargets.field (réf)          + ∂point/∂q
 object_clouds ─pose(démo)─►eval─►                object_clouds ─pose(SE(3))─►eval─► field cur + ∂/∂SE(3)
   └─►EnvironmentInteractionTargets (réf)
                     │                                                    │
                     └────────────► solve : résidus (réf − cur), coûts ◄──┘
```

## Architecture révisée de `targets/`

On garde la règle d'or « **style et interaction ne se connaissent pas** » : chaque concern porte SON
build (réf) **et** SON eval ; seul `evaluator.py` les réunit.

```
targets/
  contracts.py     RÉFS  : ContactField, MultiChannelField, StyleTargets,
                           RobotInteractionTargets, EnvironmentInteractionTargets,
                           FrameTargets, FramePose, FrameTrace                    [existe]
                   ÉVAL  : StyleEval, ContactEval (+ ContactEnvEval)              [NOUVEAU]
  config.py        KNOBS StyleConfig/TargetsConfig + recette style robot-keyée
                   (SMPL_BODY_INDEX/style_table) — ex-style/tables.py fusionné ici  [existe, 02ef94b]
  pipeline.py      RÉFÉRENCES : process_frame / run_sequence → FrameTargets ; trace_frame   [existe]
  evaluator.py     NOUVEAU (q-dép., orchestrateur) : Evaluator(ctx)
                       .style(q)                  → StyleEval
                       .contacts(q, object_poses) → ContactEval
  style/
    build.py       (réf)  bones SMPL → StyleTargets (géométrie seule)            [existe]
    eval.py        NOUVEAU (éval) FK(q) : link (R,t) + jac → StyleEval (config-free :
                   le SCALE/OFFSET ne sert qu'à la référence, pas à l'éval du robot)
  interaction/
    pointclouds.py pose_cloud              [noyau partagé]                       [existe]
    fields.py      eval_fields             [noyau partagé — RENOMMÉ depuis eval.py]
    geodesic.py    geo_value_grad / nearest_index   [noyau partagé]             [existe]
    transport.py   transport               [réf]                                [existe]
    refs.py        robot_interaction_targets / environment_interaction_targets
                                           [réf — RENOMMÉ depuis targets.py]
    eval.py        NOUVEAU (éval) pose robot_cloud @FK(q) + objets @SE(3),
                   eval_fields → ContactEval (field courant + jacobiennes)
```

`solve` se réduit à : construire `Evaluator(ctx)` une fois, puis par frame × itération appeler
`.style(q)` / `.contacts(q, object_poses)`, et assembler résidus/coûts vs `FrameTargets`.

**`Evaluator` est construit UNE fois** (séquence-wide) : il ne dépend que des assets **statiques** de
`ctx` (`robot_cloud`, `channels`, `correspondence`, `margin`, `robot`). Les seules entrées per-frame
sont les variables de décision `(q, object_poses)`. Les références per-frame (`list[FrameTargets]`)
sont une sortie **parallèle**, pas une entrée de l'évaluateur.

## Variables de décision (rappel des conventions)

- **`q`** : pinocchio free-flyer, `q = [t_pelvis(3), quat_pelvis(4, wxyz→xyzw géré dans l'impl),
  angles_joints]`. Tangente `v` de dim `nv = 6 + n_joints` (6 = base : lin + ang).
- **`object_poses`** : `(object_rot (N,3,3), object_pos (N,3))` courants. Pour un objet **fixe**
  (non optimisé), `solve` passe la pose `grounded` ; pour un objet **variable**, l'itéré courant.
  L'évaluateur fournit la jacobienne objet pour **tous** ; `solve` ignore celle des objets fixes.

## Contrats — types ajoutés (`targets/contracts.py`)

Frozen dataclasses, numpy-only (cohérent avec les types existants). `nv = 6 + n_joints`.

```python
@dataclass(frozen=True)
class StyleEval:
    """État courant des links suivis à q (FK), + jacobiennes géométriques. Reference-free, cost-free.
    Ordre = StyleTargets.link_names (mêmes links que la référence de style)."""
    position: np.ndarray     # (L, 3)      position monde courante du link
    rotation: np.ndarray     # (L, 3, 3)   rotation monde courante du link
    jac_pos: np.ndarray      # (L, 3, nv)  ∂position/∂v   (monde)
    jac_rot: np.ndarray      # (L, 3, nv)  ∂ω/∂v          (jacobienne angulaire géométrique, monde)
    link_names: tuple[str, ...]  # (L,)

@dataclass(frozen=True)
class ContactEval:
    """Géométrie de contact courante (robot) + jacobiennes géométriques pour (q, object_poses).
    Reference-free, cost-free. Canal-first (C, M) sur les M points de contrôle robot.
    `field` suit la convention MultiChannelField (sol en monde, canal objet en objet-local)."""
    field: MultiChannelField   # (C, M) : distance/witness/direction/active courants
    point_jac: np.ndarray    # (M, 3, nv) ∂(point robot monde)/∂v
    probe_jac_obj: np.ndarray  # (C, M, 6) ∂(probe dans le frame canal)/∂(tangente de l'objet du canal) ;
                               #   lignes du canal sol = 0 ; canal c → objet channels[c].object_idx (creux)
    env: tuple["ContactEnvEval", ...]  # côté environnement, un par nuage objet

@dataclass(frozen=True)
class ContactEnvEval:
    """Côté env : nuage objet i vs canaux. Dépend des object_poses seuls (pas de q).
    Diagonale self-contact déjà neutralisée par eval_fields (self_idx)."""
    field: MultiChannelField   # (C, P_i)
    cloud_jac_self: np.ndarray # (P_i, 3, 6) ∂(point du nuage objet i, monde)/∂(tangente objet i)
    probe_jac_obj: np.ndarray  # (C, P_i, 6) ∂(probe dans le frame canal)/∂(tangente de l'objet du canal)
```

**Convention de repère** (le point critique côté coûts) : `field` (donc `direction` = normale,
`witness`) reste dans le **frame du canal** (sol = monde, objet = objet-local), **identique à la
référence** (`FrameTargets`), donc directement comparable canal-par-canal. `point_jac` /
`cloud_jac_self` sont en **monde**. Le contrat cost-agnostique : `∂distance/∂var = normaleᵀ ·
(jacobienne géométrique correspondante)` — contraction triviale faite par `solve`, qui choisit la
forme exacte du résidu (scalaire distance, witness-plane, contact dur…).

> **Note d'implémentation (raffinée dans le plan)** : la convention de tangente `SE(3)` (body/LOCAL
> vs spatial) suit pinocchio (`LOCAL` free-flyer), appliquée **uniformément** à la base robot et aux
> objets. Les formules exactes `∂x/∂δξ` (probe dans frame canal) et `point_jac` (link Jac translatée
> au point de contrôle : `J_lin − [R·offset]_× J_ang`) sont figées dans le plan.

## Changements `prepare`

### `prepare/contracts.py` — `RobotModel` (protocol)

Ajout d'une méthode (le reste inchangé) :

```python
def link_jacobians(self, q: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Pour la config free-flyer q : (rot (L,3,3), pos (L,3), jac_lin (L,3,nv), jac_ang (L,3,nv)),
    en repère monde, alignés sur link_names. nv = 6 + n_joints."""
```

`InteractionContext` reste **numpy-only-importable** : `RobotModel` est un protocol, l'instance lourde
(pinocchio) est cachée — exactement comme `BodyModel` sur `GroundedScene`. `targets` n'importe
**jamais** pinocchio ; il appelle `ctx.robot.link_jacobians(q)` à travers le protocol, comme il
appelle déjà `grounded.body.bone_transforms(...)`.

### `prepare/load/robot.py` — impl FK free-flyer + jacobiennes

L'impl actuelle `UrdfRobot` (yourdfpy, **base fixe**, sans jacobienne) ne suffit pas. Nouvelle impl
**pinocchio** (free-flyer, jacobiennes de frame analytiques), portée de V1 `test_socp/pin_model.py`
(cf. carte de portage `CLAUDE.md`). Le robot reste instancié **uniquement dans `prepare/`**.

- `link_transforms(q)` : conservé (utilisé par le posage `pose_cloud`, déjà câblé).
- `link_jacobians(q)` : nouveau (transforms + `J_lin`/`J_ang` par link, free-flyer).
- Parité `link_transforms` pinocchio vs yourdfpy (au repère base près) en test.

## Conformité aux règles d'or (`CLAUDE.md`)

- **#1 (sens unique, zéro cycle)** : `solve` importe la surface publique de `targets`
  (`Evaluator`, `StyleEval`, `ContactEval`) + les références (`FrameTargets`) + les assets
  (`InteractionContext`). `targets` n'importe jamais `solve`. Acyclique.
- **#2 (pas de noyau central)** : `targets` **possède** le noyau et l'évaluateur ; on **expose**,
  on ne crée pas de module partagé. Les sous-modules `style/`, `interaction/` restent internes ;
  `solve` n'importe que le **package** (`targets/__init__`).
- **#3 / #5 (cœur pur, obs aux seams)** : noyaux purs, numpy/torch-free à l'import (cinématique
  lourde derrière le protocol). `prof.span` dans `pipeline.py`/`evaluator.py` (orchestrateurs),
  jamais dans les ops pures.
- **#7 (critère q) — assoupli, assumé** : un évaluateur **q-dépendant** vit désormais dans `targets`
  (charte : *online = références q-indép. **+** évaluateur q-dép., cinématique injectée*). C'est le
  seul écart, conscient : il évite de dupliquer les noyaux entre `targets` et `solve` (#2/#8) et garde
  `targets` import-léger. `solve` devient purement l'optimiseur (construit le problème, itère).
- **#8 (homogénéité)** : robot (K=1, parts=liens, FK) et objets (K=1) passent par le **même**
  `pose_cloud`/`eval_fields` que l'humain (K~4) ; canaux uniformes (sol + N objets).

## Flux

```
prepare.runner ─► InteractionContext{ channels, human_cloud, object_clouds, correspondence,
                                      margin, robot_cloud, robot(+link_jacobians) }   (build-once)
              └─► GroundedScene
targets.pipeline   ─► list[FrameTargets]  (références q-indép. : style + contact refs + poses objets)
targets.Evaluator(ctx)  (construit 1×)
  par frame × itération :
    .style(q)                  → StyleEval   (pos/rot links + J_pos/J_rot)
    .contacts(q, object_poses) → ContactEval (field courant + point_jac + probe_jac_obj + env)
solve  ─► résidus (FrameTargets réf  vs  StyleEval/ContactEval courant) + coûts + cônes
```

## Gestion d'erreurs

- `InteractionContext` (assemblage) : invariants existants conservés (`robot_cloud.n_points ==
  correspondence.n_points` ; `link_idx` dans `correspondence.link_names`). `raise ValueError` explicite.
- `StyleEval` / `ContactEval` : valider les formes au `__post_init__` (cohérence `L` / `M` / `C` /
  `nv` / `N`), comme `MultiChannelField`.
- `link_jacobians` : `q` de dimension `nq` attendue (free-flyer) — `raise ValueError` si mismatch.
- `eval_fields` déjà robuste hors-grille (probes clampés → `active=False`) : inchangé.

## Tests (`HoloV2/tests/`, env `holonew`, `max_frames` bas)

- **Noyaux purs** (`pose_cloud`, `eval_fields` renommé `fields.py`, géodésique) : tests unitaires
  existants conservés (imports mis à jour après renommage).
- **FK + jacobienne** (`prepare/load/robot.py` pinocchio) :
  - parité `link_transforms` pinocchio vs yourdfpy actuel (au repère base près) sur une config connue ;
  - **jacobienne par différences finies** : `(point(q+εeᵢ) − point(q−εeᵢ))/2ε ≈ J_lin·eᵢ` (et idem
    angulaire) sur quelques `q` aléatoires, tolérance documentée. **Le test-clé.**
- **`style/eval.py`** : `StyleEval.position/rotation` == `FK` direct ; `jac_pos/jac_rot` vs FD.
- **`interaction/eval.py`** : `ContactEval.field` courant == `eval_fields` direct sur le `robot_cloud`
  posé @q ; `point_jac` vs FD ; `probe_jac_obj` vs FD sur un objet bougé.
- **`Evaluator` (intégration)** : à `q` réalisant la démo retargetée, `StyleEval.position` retombe
  (à tolérance) sur les références de style, et `ContactEval.field` sur la référence transportée —
  sanity du seam.
- **Compile/import** : après changement des `contracts.py`, `python -m py_compile` + un import dans
  l'env (contrats numpy-only-importables).

## Hors périmètre / suites

- Tout `solve/` (résidus, coûts D/X, activation, cônes, backend enfichable, forme canonique).
- Batch vectorisé sur T de l'évaluateur (déjà array-oriented ; optimisation ultérieure).
- Robots autres que G1 (entrée de données dans les tables, pas de changement de surface).
