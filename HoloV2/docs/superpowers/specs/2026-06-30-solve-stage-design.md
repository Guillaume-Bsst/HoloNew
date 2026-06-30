# Spec — Étage `solve/` : retargeting par QP linéarisé (architecture)

**Date** : 2026-06-30 · **Étage** : `solve/` (nouveau, q-dépendant) · **Statut** : conçu

L'étage final du pipeline `prepare → targets → solve`. Consomme la surface publique de `targets`
(l'`Evaluator` + les types Eval/Ref + le lecteur géodésique) et produit la **trajectoire qpos**
retargetée. Conçu **à neuf** dans l'idiome propre de prepare/targets — **pas** un portage de la
structure V1 (classe-dieu `TestSocpRetargeter`, ~7 métiers, flux opaques).

## Décisions cadrées (entrées de ce design)

| Décision | Choix |
|---|---|
| Structure | **Redesign à neuf**, idiome prepare/targets (étage possède `contracts.py` + `config.py`, modules mono-responsabilité, flux LINÉAIRE explicite). PAS de portage V1. |
| Classe de problème | **QP pur** : objectif = Σ moindres carrés (quadratique), contraintes linéaires/box. PAS de SOCP — le seul cône V1 était le trust-region L2 ; on prend un trust-region **box**. |
| Trust-region | **box `‖δv‖∞ ≤ r` par-DOF** (→ QP, ProxQP-ready). `TrustRegion.norm` garde le champ (2|∞) pour un L2 futur, mais v1 = box seul. |
| Backend | **enfichable** (Protocol `SolveBackend`). **CVXPY = le premier** (vérifier + benchmarker le retargeting ; route vers OSQP/ProxQP en QP). |
| Abstraction termes | **fonctions pures par concern** (`terms/{style,contact,object,reg}.py`), pas de Protocol `Term`. YAGNI. |
| Frontière targets/solve | **`targets` INCHANGÉ** (géométrie générale brute + lecteurs de champ ; Eval **ref-free**). Les ops complexes au **service exclusif des résidus** (contractions, `R_i`, `log`, `value−ref`) vivent dans **`solve/terms/`**. |
| Variable objet | **objet = variable de décision** en v1 (résidus CO + O + couplage `δv↔δξ`). |
| Dép lourde | `solve` reste **pinocchio/torch-free** (cinématique via le Protocol `RobotModel`) ; **cvxpy confiné à `backend/cvxpy.py`**. |

Résidus v1 (cf. mémoire) : `S-pos, S-rot, C-D, C-X(géodésique), CO-D, CO-X(géodésique), O, reg`.

## Flux de données (le point que V1 ratait — le rendre évident)

```
solve.runner.solve(grounded, ctx, frame_targets, config) -> SolveTrajectory
  evaluator = targets.Evaluator(ctx, robot_name)          # construit 1×
  pour chaque frame f :
    q, poses = warm_start(f)                               # f>0 : depuis f-1 ; f=0 : joints neutres + base à la cible pelvis
    répéter (SQP/trust-region) :
      evals    = evaluator.evaluate(q, poses)              # géométrie courante + jacobiennes (1 appel)
      problem  = assemble(evals, frame_targets[f], geo, config)  # ResidualBlocks + box constraints
      step     = backend.solve(problem)                    # (dv, dξ) — QP
      q, poses = retract(q, poses, step)                   # robot.integrate / exp SE(3) numpy
    converge si ‖dv‖ < tol  (ou n_iter atteint)
    qpos[f], object_poses[f] = q, poses
  -> SolveTrajectory(qpos, object_poses, info)
```
**Une passe linéaire** `evaluate → assemble → solve → retract`. Pas de classe-dieu, pas de flux caché.

## Arbre des modules (calqué sur prepare/targets)

```
solve/
  contracts.py   Problem (ResidualBlock + LinearConstraint + TrustRegion), Step, SolveTrajectory,
                 FrameInfo — frozen, numpy-only
  config.py      SolveConfig : poids par terme, rayon trust-region (par-DOF), n_iter (first/per-frame),
                 tol convergence, choix backend
  terms/         ops complexes AU SERVICE EXCLUSIF DES RÉSIDUS (le « dernier étage complexe ») :
    _ops.py        ops réutilisables : world_normal(R_i, n_local), dist_jac(n_world, point_jac)=∂d/∂δv,
                   geo_chain(∇geo, point_jac)=∂geo/∂δv, so3_log(R_ref, R_cur), obj_tangent helpers
    style.py       build_style  -> S-pos, S-rot           (StyleEval + StyleTargets)
    contact.py     build_contact-> C-D, C-X(géo)          (ContactEval + RobotInteractionTargets + geo)
    object.py      build_object -> CO-D, CO-X(géo), O      (ContactEval.env + EnvironmentInteractionTargets + poses)
    reg.py         build_reg    -> reg
    constraints.py limites articulaires + trust-region box (depuis RobotModel + config)
  backend/       solveur ENFICHABLE :
    base.py        Protocol SolveBackend : solve(Problem) -> Step
    cvxpy.py       CvxpyBackend  (le premier ; cvxpy importé ICI uniquement)
  assemble.py    (evals, refs, geo, config) -> Problem   [appelle terms/ + constraints]
  retract.py     q ⊕ dv (robot.integrate) ; pose_objet ⊕ dξ (exp SE(3) numpy)
  loop.py        l'itéré SQP : evaluate -> assemble -> backend.solve -> retract -> converge
  runner.py      ENTRÉE PUBLIQUE : solve(grounded, ctx, frame_targets, config) -> SolveTrajectory
  __init__.py    surface publique
```

## La frontière targets / solve (verrouillée)

| Étage | Rôle |
|---|---|
| **targets** (INCHANGÉ) | géométrie **générale** brute + dérivées (`point_jac`, `field` normale/witness, `jac_pos/rot`) + **lecteurs de champ** (`geo_value_grad` consomme la réf géodésique). L'**Eval reste strictement ref-free**. |
| **`solve/terms/`** | ops **complexes spécifiques aux résidus** (`_ops.py`) + les builders `build_*` (poids repliés) → `ResidualBlock`. |
| **`solve/` (reste)** | `config` (poids), `assemble` (terms→Problem), `backend` (QP), `retract`, `loop` (SQP), `runner`. |

Critère de tri (ce qui a tranché « pousser dans targets » vs « garder dans solve ») : **la finalité**. Les
contractions/`R_i`/`log`/`value−ref` ne servent **qu'au calcul des résidus** → elles sont **spécifiques
solve**, isolées dans `solve/terms/` (single-purpose, testées), et ne **polluent pas** targets ni ne
**rouvrent** son évaluateur validé. La seule lecture qui touche une réf de façon non-triviale (la
**géodésique**) reste dans `geo_value_grad` côté targets — l'Eval cost-free ne consomme jamais la réf.

## Contrats (`solve/contracts.py`) — frozen, numpy-only

```python
@dataclass(frozen=True)
class ResidualBlock:        # coût ‖A·dv + A_obj·dξ + c‖²  (poids DÉJÀ repliés dans A, c)
    A: np.ndarray           # (m, nv)
    c: np.ndarray           # (m,)
    A_obj: np.ndarray|None  # (m, n_obj*6) ou None  (couplage robot↔objet)
    name: str               # "C-D", "S-rot"… (diagnostic + breakdown poids)

@dataclass(frozen=True)
class LinearConstraint:     # lb ≤ A·dv (+ A_obj·dξ) ≤ ub  (None=libre ; lb==ub=égalité)
    A: np.ndarray; lb: np.ndarray|None; ub: np.ndarray|None; A_obj: np.ndarray|None; name: str

@dataclass(frozen=True)
class TrustRegion:          # ‖var‖_p ≤ radius  (par-DOF)
    var: str                # 'dv' | 'dxi'
    radius: np.ndarray      # (nv,) ou (n_obj*6,)  — rayon PAR-DOF (gère les unités m/rad/joints)
    norm: int               # -1 = ∞ (box→QP, v1) | 2 (L2→SOC, futur)

@dataclass(frozen=True)
class Problem:
    nv: int; n_obj: int
    residuals:     tuple[ResidualBlock, ...]
    constraints:   tuple[LinearConstraint, ...]
    trust_regions: tuple[TrustRegion, ...]
    # __post_init__ : valide formes (A: m×nv ; A_obj: m×n_obj*6 ; cohérence m) — style MultiChannelField

@dataclass(frozen=True)
class Step:                 # sortie d'un backend
    dv: np.ndarray          # (nv,)
    dxi: np.ndarray|None    # (n_obj, 6)
    value: float; status: str

@dataclass(frozen=True)
class FrameInfo:            # diagnostic par frame (réglage poids + benchmark)
    n_iters: int; status: str; cost: float
    cost_by_term: dict[str, float]   # norme résidu par terme (S-pos / C-D / …) — l'outil n°1 de tuning

@dataclass(frozen=True)
class SolveTrajectory:      # sortie du runner
    qpos: np.ndarray            # (T, nq)
    object_poses: np.ndarray    # (T, N, 7)  pos+quat (vide si pas d'objet)
    info: tuple[FrameInfo, ...]
```

## Config (`solve/config.py`) — frozen, stdlib-only

`SolveConfig` (knobs) : poids par terme (`w_pos, w_rot, w_cd, w_cx, w_cod, w_cox, w_obj, w_reg` — scalaires
ou par-lien/canal), activation contact `α` (gating + pondération `d_ref`), rayon trust-region par-DOF
(base_pos / base_rot / joints / objet), `n_iter_first` / `n_iter_per_frame`, `step_tol`, `backend`
(`"cvxpy"`). `TargetsConfig`-style : défaut = `SolveConfig()`, override inline.

## Les termes (`solve/terms/`) — `build_*(eval, ref, cfg) -> list[ResidualBlock]`

Rappel : `ResidualBlock = ‖A·δv + A_obj·δξ + c‖²`, **poids repliés**. Les ops complexes sont dans `_ops.py`.

**`style.py`** (← `StyleEval` + `StyleTargets`) — robot seul (`δv`) :
- **S-pos** : `A = w_pos ⊙ jac_pos` (L*3, nv) ; `c = w_pos ⊙ (pos_cur − pos_ref)`.
- **S-rot** : `A = w_rot ⊙ jac_rot` ; `c = w_rot ⊙ so3_log(R_ref, R_cur)` (R_ref via quat→R).

**`contact.py`** (← `ContactEval` + `RobotInteractionTargets` + `geo_value_grad`) — couple `δv↔δξ` (canal objet) :
- **C-D** (paires `active`) : `n_world = world_normal(R_i, direction)` ; `A = w_cd · dist_jac(n_world, point_jac)` ;
  `A_obj = w_cd · (∂d/∂δξ via probe_jac_obj)` (creux) ; `c = w_cd · (d_cur − d_ref)`. `α(d_ref)` plié dans `w_cd`.
- **C-X** (géodésique) : `(geo_value, ∇geo) = geo_value_grad(witness(q), witness_ref, channel)` ;
  `A = w_cx · geo_chain(∇geo, point_jac)` ; `c = w_cx · geo_value` (cible déjà = 0, pas de réf à soustraire).

**`object.py`** (← `ContactEval.env` + `EnvironmentInteractionTargets` + poses) — objet seul (`δξ`) :
- **CO-D**, **CO-X** : idem C-D/C-X mais `A = 0`, `A_obj` via `cloud_jac_self` + `probe_jac_obj` (consistance
  propre de l'objet : objet vs sol/autres objets).
- **O** : `A = 0`, `A_obj = w_obj · I`, `c = w_obj · e_ξ` (écart pose courante ↔ `object_ref`).

**`reg.py`** : `A = w_reg · I (nv)`, `c = 0` (damping de pas, bonne condition du QP). *Variante notée (cf.
Holosoma `q_nominal`)* : régularisation de **posture** `‖(q ⊕ δv) − q_nominal‖²` vers une pose nominale
(p.ex. neutre ou la pose de style) plutôt qu'un simple damping — incrément simple si le damping seul dérive.

**`constraints.py`** (← `RobotModel` + config) : limites articulaires (box sur les DOF de `δv`),
trust-region **box** par-DOF. [non-pénétration `d ≥ 0` = incrément suivant.]

> Couplage robot↔objet : **dans C** (points robot vs canal objet : `δv` ET `δξ`). **CO** = appui propre de
> l'objet, **O** = ancrage à la pose observée — tous deux `δξ` seul. La variable objet est tirée par C,
> retenue par CO+O ; les poids arbitrent.

## Backend (`solve/backend/`)

- `base.py` : `SolveBackend` Protocol — `solve(problem: Problem) -> Step`.
- `cvxpy.py` : `CvxpyBackend` — assemble `cp.sum_squares(A·dv + A_obj·dξ + c)` par bloc, `LinearConstraint`
  en contraintes affines, `TrustRegion` box en bornes ; résout (CVXPY route vers OSQP/ProxQP en QP).
  **cvxpy importé uniquement ici.**

## Boucle + rétraction

- `loop.py` : l'itéré SQP par frame (voir Flux). Warm-start frame `f>0` depuis `f-1` ; **`f=0` = joints
  neutres + base flottante placée à la cible pelvis de style** (`StyleTargets` pelvis : position +
  orientation) — c'est l'**init natif de Holosoma** (`q_init = [human pelvis pos, human root quat,
  joints=0]`), bien meilleur qu'une base à l'origine (seuls les joints convergent ; `n_iter_first`
  absorbe ce raffinement). Convergence `‖dv‖ < step_tol` ou `n_iter`. Trust-region **fixe** (adaptatif =
  incrément suivant). `prof.span` ici.
- `retract.py` : `q ⊕ dv` via `robot.integrate` (Protocol, pinocchio-free) ; `pose_objet ⊕ dξ` via exp SE(3)
  numpy (quaternion exp + translation). Pur.
- `runner.py` : entrée publique `solve(grounded, ctx, frame_targets, config) -> SolveTrajectory` ; construit
  l'`Evaluator` 1×, boucle les frames, collecte qpos/object_poses/info. `prof.span` (sequence).

## Conformité aux règles d'or (`CLAUDE.md`)

- **#1 (sens unique, zéro cycle)** : `solve` importe la surface publique de `targets` (`Evaluator`,
  `StyleEval`/`ContactEval`, `geo_value_grad`, `FrameTargets`) + `prepare.contracts` (via `ctx`). Jamais un
  interne de targets. Acyclique.
- **#2 (types + config co-localisés)** : `solve/contracts.py` (`Problem`…) + `solve/config.py`.
- **#3 / #5 (cœur pur, obs aux seams)** : `terms/`, `backend/`, `retract` purs (numpy, pas de mutation, pas
  d'I/O) ; `prof.span` dans `loop`/`runner` seulement.
- **#7 (critère q)** : `solve` EST l'étage q-dépendant. Reste **pinocchio/torch-free** (cinématique via le
  Protocol `RobotModel`) ; cvxpy = dép propre confinée à `backend/cvxpy.py`.
- **#8 (homogénéité)** : mêmes `_ops` (`dist_jac`, `geo_chain`, `world_normal`) pour C **et** CO ; un seul
  `ResidualBlock` pour tous les termes ; un seul backend Protocol.
- **#9 (YAGNI)** : fonctions pures (pas de Protocol `Term`) ; QP (pas SOCP) ; trust-region fixe (pas
  adaptatif) ; box seul (pas de branche L2) ; CVXPY seul (pas encore ProxQP).

## Gestion d'erreurs

- `Problem.__post_init__` : valide les formes (A m×nv, A_obj m×n_obj*6, cohérence m) → `ValueError`.
- `TrustRegion` : `radius > 0` (par-DOF), `norm ∈ {-1, 2}`.
- `backend.solve` : un statut non-optimal (`infeasible`/`unbounded`) → `Step.status` propagé ; le `loop`
  décide (arrêt + diagnostic), pas de crash silencieux.
- `assemble` : si `n_obj=0`, les termes objet ne produisent rien (pas de bloc `A_obj`).

## Tests (`HoloV2/tests/`, env `holonew`, `max_frames` bas)

- **`terms/_ops.py`** : unitaire + **FD** par op (`dist_jac`, `geo_chain`, `so3_log`, `world_normal`).
- **`terms/{style,contact,object,reg}`** : `build_*` sur eval/ref synthétiques → formes `ResidualBlock` +
  **linéarisation vs FD du résidu réel** (le `A·δv + c` ≈ différence finie du résidu non-linéaire).
- **`contracts`** : validation des formes du `Problem`.
- **`backend/cvxpy`** : mini-QP connu (moindres carrés 2-var + box) → optimum analytique.
- **`retract`** : `robot.integrate` round-trip ; exp SE(3) objet (vs une rotation connue).
- **`loop` (intégration)** : sur une frame → **convergence** (‖dv‖→0) + **coût décroît** monotone.
- **end-to-end (data-gated HODome)** : frame réelle → qpos fini, limites articulaires respectées, coût ↓,
  pelvis ≈ cible style ; (bonus) **parité/benchmark vs la sortie V1** (la motivation CVXPY).

## Hors périmètre / incréments suivants

- **Centroïdal / équilibre** (plan en attente : CoM + I·ω) — nouveau target + terme.
- **Termes TEMPORELS** (lissage vitesse/jerk inter-frames) — solve per-frame v1 → risque de jitter.
- **Self-collision** ; **non-pénétration dure** (`d ≥ 0` en contrainte) ; **trust-region adaptatif**.
- **Backend ProxQP natif** (boucle chaude warm-startée) ; **branche L2/SOCP** (si comparaison voulue).
- Batch/vectorisation multi-frame ; le CLI tyro d'entrée.
