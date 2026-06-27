# HoloV2 — `targets/` (ÉTAPE 2 : construction ONLINE des cibles)

Consomme les sorties de `prepare/` et produit, par frame, le `FrameTargets` que `solve/`
consomme. Per-frame : **seules les transfos changent** ; SDF, nuages-rest et correspondance
sont statiques (prepare) → coût minimal.

## Seam prepare → targets (via les types de `contracts.py`, jamais le code)
- `GroundedScene` : motion calibrée (joints + params + poses objets / frame)
- `InteractionContext` : assets statiques (channels, human_cloud, object_clouds, correspondence)
- `Calibration`

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
                               FrameTargets(style, robot_interaction, env_interaction)
```

## 3 fonctions pures, réutilisées partout (homogénéité = anti-spaghetti)
- `pose_cloud(PointCloud, part_rot, part_pos) -> (P,3)` :
  `Σ_k w_k (R[parts_k] @ offsets_k + t[parts_k])`. `part_rot (J,3,3)` / `part_pos (J,3)` = la transfo
  monde de chaque part (humain : `BodyModel.bone_transforms` ; objet : sa pose `(R[None],t[None])` ;
  robot : FK des liens). MÊME fonction pour humain (K~4, parts=os) et objets (K=1, parts=corps).
  **Implémentée** (`targets/interaction/pointclouds.py`), vectorisée einsum, torch-free.
- `eval_fields(points, channels, object_poses, margin) -> MultiChannelField` : pour chaque
  `Channel`, transforme les points dans le frame du champ (identité si `object_idx is None`
  = sol, sinon `object_poses[object_idx]`), puis `channel.field.sample_local(...)` -> empile.
  MÊME fonction pour le nuage humain ET chaque nuage objet (matrice clouds×channels). La
  liaison canal→pose est EXPLICITE (pas d'offset N vs N+1).
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
  style/               OBJECTIF DE STYLE (ex-"body") — posture, ignore l'objet.
                       À revoir proprement (quoi/comment) ; placeholder pour l'instant.
                       FramePose -> StyleTargets (mapping articulaire / GMR)
  interaction/
    pointclouds.py     pose_cloud — pose tout nuage (humain K~4 / objet·robot K=1) [FAIT]
    eval.py            eval_fields (sample chaque Channel : SDF trilinéaire — chemin unique, sol plat inclus)
    transport.py       transport (gather via correspondence)
    targets.py         assemble RobotInteractionTargets + EnvironmentInteractionTargets
  pipeline.py          process_frame : FramePose -> style + interaction -> FrameTargets
                       run_sequence : boucle online OU batch vectorisé
```

## Règles anti-spaghetti
- `pipeline` orchestre et calcule l'état partagé (`FramePose`).
- `style` et `interaction` NE SE CONNAISSENT PAS.
- flux interaction à sens unique : pose → eval → transport → assemble.
- tout passe par les types de `contracts.py`.

## Seam visualisation
`pipeline` expose aussi `trace_frame(prepared, f) -> FrameTrace` (mêmes ops pures que
`process_frame`, intermédiaires gardés) pour `viz/`. `FramePose` et `FrameTrace` sont
définis dans `contracts.py`. Détail : `docs/VIZ.md`.

## À fixer quand on code `targets/`
- contenu réel de `style/` (l'objectif de style — à concevoir).
