# HoloV2 — `targets/` (ÉTAPE 2 : construction ONLINE des cibles)

Consomme les sorties de `prepare/` et produit, par frame, le `FrameTargets` que `solve/`
consomme. Per-frame : **seules les transfos changent** ; SDF, nuages-rest et correspondance
sont statiques (prepare) → coût minimal.

## Seam prepare → targets (via la sortie publique de prepare, jamais le code interne)
Les types de l'étage `targets` vivent dans **`targets/contracts.py`** (ContactField, MultiChannelField,
StyleTargets, Robot/EnvironmentInteractionTargets, FrameTargets, FramePose, FrameTrace). Ce module
**importe la sortie publique de prepare** (`from ..prepare.contracts import GroundedScene,
InteractionContext, RobotSpec`) — jamais les sous-modules internes de `prepare/`. `pipeline` prend
`(grounded, ctx, robot, f, cfg)` (pas de bundle `prepared`) : `robot` (`RobotSpec`) clé la recette de
style (`robot.name`, non atteignable via `ctx.robot` qui est un `RobotModel`) ; `cfg` (`TargetsConfig`,
défaut `TargetsConfig()`) porte les knobs de l'étage — aujourd'hui seul `cfg.style` (`StyleConfig` :
échelle morpho + hauteurs de référence) est lu, par `style.build`. Le knob per-frame `margin` reste
dans le context (sortie `prepare`). Ce que `targets` consomme en entrée :
- `GroundedScene` : motion calibrée (joints démo + params + poses objets / frame) + **`body`** (moteur
  de posage : `body.bone_transforms` → bone (R,t)) + `calibration` (provenance, via `grounded.calibration`)
- `InteractionContext` : assets statiques (channels, human_cloud, object_clouds, correspondence ; + `scale`
  quand `transport` sera codé)

## Flux per-frame

```
GroundedScene[f] ─► FramePose(f) = bone (R,t)  +  object (R,t)   [calculé UNE fois]
        ┌──────────────────────────┴───────────────────────────┐
        ▼ demo joints                                           ▼ transfos
   style.build ─► StyleTargets            interaction (flux à SENS UNIQUE) :
                                          1. pose_cloud(cloud, transfos)      -> points monde
                                          2. eval_fields(points, channels)    -> MultiChannelField  (matrice homogène)
                                          3. transport(human_field, corresp.) -> champ sur points G1
                                          -> RobotInteractionTargets + EnvironmentInteractionTargets
        └──────────────────────────┬───────────────────────────┘
                                    ▼
                               FrameTargets(style, robot_interaction, env_interaction,
                                           object_rot, object_pos)
                                   + object_rot/object_pos (poses objets de la frame)
                                   pour que solve repose les canaux objets et initialise
                                   les variables objet.
```

## 3 fonctions pures, réutilisées partout (homogénéité = anti-spaghetti)
- `pose_cloud(PointCloud, part_rot, part_pos) -> (P,3)` :
  `Σ_k w_k (R[parts_k] @ offsets_k + t[parts_k])`. `part_rot (J,3,3)` / `part_pos (J,3)` = la transfo
  monde de chaque part (humain : `BodyModel.bone_transforms` ; objet : sa pose `(R[None],t[None])` ;
  robot : FK des liens). MÊME fonction pour humain (K~4, parts=os) et objets (K=1, parts=corps).
  **Implémentée** (`targets/interaction/pointclouds.py`), vectorisée einsum, torch-free.
- `eval_fields(points, channels, object_rot, object_pos, margin, self_idx=None) -> MultiChannelField` :
  pour chaque `Channel`, transforme les points dans le frame du champ (identité si `object_idx is None`
  = sol, sinon `(object_rot[i], object_pos[i])`), puis échantillonne le SDF (trilinéaire) -> empile.
  MÊME fonction pour le nuage humain ET chaque nuage objet (matrice clouds×channels). La
  liaison canal→pose est EXPLICITE (pas d'offset N vs N+1). `self_idx` = l'index de l'objet du nuage
  évalué (`None` pour l'humain) : la **diagonale** (nuage objet `i` vs SON canal `i`) est un
  self-contact dégénéré, court-circuité à la forme close (distance 0, witness = le point lui-même)
  SANS échantillonner le SDF — le `solve` ignorera cette diagonale à moindre coût.
- `transport(human_field, correspondence) -> field` : gather des valeurs humaines sur les M
  points G1 (`smpl_idx`). Uniquement pour l'humain. (`RobotInteractionTargets` = field SEUL ;
  la liaison points↔liens, statique, est dans `InteractionContext.correspondence`.)

## Optimisations de flux / calcul intégrées
1. `FramePose` calculé UNE fois/frame, partagé par `style` ET `interaction` (même source,
   pas de recompute).
2. `object_poses[f]` réutilisé deux fois sans recalcul : poser le nuage objet ET comme frame
   du `Channel` objet pour l'éval. (Objet `static` => transfo constante, évaluable une fois.)
3. Tout statique sauf les transfos (champs/nuages/corresp. viennent de prepare) → par frame :
   FK (cheap) + gather + sample (SDF trilinéaire — tous les canaux, sol plat inclus).
4. Layout canal-first `(C,P)` + einsum → vectorisé, zéro boucle Python par point.
5. Dualité online/batch : les 3 fonctions sont array-oriented (axe points) → `pipeline` les
   appelle par frame (téléop) OU batché sur T (baking dataset), même cœur.

## Modules (1 responsabilité chacun)
```
targets/
  config.py            knobs de targets (StyleConfig échelle morpho + hauteurs / TargetsConfig) ET la
                       recette de style robot-keyed (DATA : SMPL_BODY_INDEX, ARM_BODIES, _STYLE_TABLE
                       lien->corps/poids/offsets, style_table()) — tout le style hors math en un fichier.
  contracts.py         RÉFS  : ContactField, MultiChannelField, StyleTargets, Robot/EnvironmentInteractionTargets,
                       FrameTargets, FramePose, FrameTrace.  ÉVAL : StyleEval, ContactEval, ContactEnvEval
                       (état géométrique courant + jacobiennes analytiques ; reference-free, cost-free).
  evaluator.py         ÉVALUATEUR q-dépendant (orchestrateur) : Evaluator(ctx, robot_name), construit 1×.
                         .style(q)                       -> StyleEval   (FK links suivis + J_pos/J_rot)
                         .contacts(q, object_rot, object_pos) -> ContactEval (field courant + point_jac
                                                                + probe_jac_obj + env)
  style/               OBJECTIF DE STYLE (ex-"body") — posture, ignore l'objet.
    build.py           (réf) FramePose démo -> StyleTargets (recette GMR : SCALE puis OFFSET).
    eval.py            (éval) style_eval : robot FK @ q -> StyleEval (config-free ; gather des links suivis).
  interaction/
    pointclouds.py     pose_cloud — pose tout nuage (humain K~4 / objet·robot K=1) [noyau partagé]
    fields.py          eval_fields (sample chaque Channel : SDF trilinéaire — chemin unique) [noyau, ex-eval.py]
    eval.py            (éval) contact_eval : robot_cloud @FK(q) + objets @SE(3) -> ContactEval
                       (field courant + jacobiennes : point_jac monde, probe_jac_obj/cloud_jac_self objet)
    transport.py       transport (gather via correspondence) [réf]
    refs.py            robot_interaction_targets / environment_interaction_targets [réf, ex-targets.py]
    geodesic.py        geo_value_grad / nearest_index — lecture différentiable du champ géodésique. [noyau]
  pipeline.py          RÉFÉRENCES : process_frame / trace_frame -> FrameTargets ; run_sequence.
```

## Symétrie référence / évaluation
`targets` fait UNE opération (poser une config, lire son état style + contact) sur DEUX configs :
- **Référence** (`pipeline.py`, q-indép., 1×/frame, bakée) : humain/démo -> `FrameTargets` (style +
  contact refs transportés). Pas de jacobienne.
- **Évaluation** (`evaluator.py`, q-dép., par itération) : robot @ `q` + objets @ `SE(3)` ->
  `StyleEval` / `ContactEval` (état courant + jacobiennes analytiques). Les MÊMES noyaux purs
  (`pose_cloud`, `fields.eval_fields`) servent les deux ; seul le côté cible porte une jacobienne.
Le résidu (`cur - ref`) et le coût se font **dans `solve`** à partir de ces deux sorties.
`Evaluator` est construit UNE fois (assets statiques de `ctx`) ; entrées par itération = `(q, object_poses)`.

## Règles anti-spaghetti
- `pipeline` orchestre et calcule l'état partagé (`FramePose`).
- `style` et `interaction` NE SE CONNAISSENT PAS.
- flux interaction à sens unique : pose → eval → transport → assemble.
- tout passe par les types des étages (`targets/contracts.py` pour les cibles ; les sorties amont
  via `..prepare.contracts`).

## Seam visualisation
`pipeline` expose aussi `trace_frame(grounded, ctx, f) -> FrameTrace` (mêmes ops pures que
`process_frame` via le cœur partagé `_build_frame`, intermédiaires gardés) pour `viz/`. `FramePose` et `FrameTrace` sont
définis dans `targets/contracts.py`. Détail : `docs/VIZ.md`.

## À fixer quand on code `targets/`
- `style/` est implémenté (recette GMR portée, parité V1) ; knobs dans `config.StyleConfig`.
