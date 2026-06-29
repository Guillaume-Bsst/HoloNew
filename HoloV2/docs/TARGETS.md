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
  style/               OBJECTIF DE STYLE (ex-"body") — posture, ignore l'objet.
                       FramePose -> StyleTargets (recette GMR par lien).
                       build.py = la math SEULE (SCALE puis OFFSET) ; il lit knobs + recette via config.
  interaction/
    pointclouds.py     pose_cloud — pose tout nuage (humain K~4 / objet·robot K=1) [FAIT]
    eval.py            eval_fields (sample chaque Channel : SDF trilinéaire — chemin unique, sol plat inclus)
    transport.py       transport (gather via correspondence)
    targets.py         assemble RobotInteractionTargets + EnvironmentInteractionTargets
    geodesic.py        geo_value_grad / nearest_index — lecture différentiable (MLS degré-1, valeur +
                       gradient tangent) du champ géodésique précalculé prepare.GeodesicTable à un
                       witness(q) continu, pour le résidu witness côté solve/utilisateur. nearest_index
                       snappe witness_ref sur sa source offline. Numpy-only, torch-free. Ré-exportés
                       sur la surface publique targets (targets/__init__.py, __all__) et sur
                       targets.interaction (__init__.py).
  pipeline.py          process_frame : FramePose -> style + interaction -> FrameTargets
                       run_sequence : boucle online OU batch vectorisé
```

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
