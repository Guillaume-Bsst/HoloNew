# HoloV2 — Architecture (vue d'ensemble)

Carte globale. Détails par étape : `PREPARE.md`, `TARGETS.md`, `VIZ.md`, `CACHE.md`,
`OBS.md` (et `SOLVE.md`, à venir). Pas de noyau central : **chaque étage du pipeline possède
ses types ET sa config**, co-localisés.

Le pipeline est **LINÉAIRE** `prepare → targets → solve`. Chaque étage expose une **surface
publique** = son module `contracts.py` (les types de DONNÉES qui transitent — dataclasses `frozen`,
numpy-only) + son module `config.py` (les knobs — dataclasses `frozen`) + son point d'entrée public
(`prepare.runner.prepare`). L'aval importe **uniquement la sortie publique de l'amont** (`targets`
fait `from ..prepare.contracts import GroundedScene, InteractionContext`), jamais ses sous-modules
internes (`prepare/load/*`, …). Les dépendances ne vont que vers l'AVAL ⇒ graphe **acyclique par
construction**, plus besoin d'un noyau partagé central. La config de `prepare` = `prepare/config.py`
(`PrepareConfig` + sous-configs `calibration/sdf/cloud/correspondence`) : `PrepareConfig()` donne le
défaut, override inline `PrepareConfig(sdf=SdfConfig(spacing=0.005))` ; un CLI tyro s'attachera ici
avec le point d'entrée de run. Les clés du cache build-once dérivent de la config (voir `CACHE.md`).

## Règle d'or : zéro spaghetti
1. **Dépendances à sens unique** — pipeline linéaire `prepare → targets → solve` : l'aval importe
   la **sortie publique** de l'amont (`<étage>.contracts` / `.config`), jamais ses internes ⇒
   acyclique par construction, jamais de cycle.
2. **Chaque étage possède ses types + sa config**, co-localisés (`<étage>/contracts.py` +
   `<étage>/config.py`) ; pas de noyau partagé central. Une responsabilité par module, pas de
   classe-dieu (le retargeter V1 fait ~7 métiers dans une classe — on explose).
3. **Contrats typés aux frontières** (dataclasses frozen dans le `contracts.py` de chaque étage),
   pas de `SimpleNamespace`.
4. **Effets de bord aux extrémités** (`viz/`, futur `app/`). Le cœur est pur, testable sans
   disque ni écran.
5. **Data-oriented** (vitesse) : Structure-of-Arrays numpy, layout canal-first `(C,P)`,
   tout le lourd amorti offline.
6. **Visu = consommateur** (jamais de hook dans le calcul).
7. **Observabilité aux seams** : timing/logs via `obs.Profile` (spans) dans les
   orchestrateurs uniquement, no-op quand off — jamais dans les ops pures (voir `OBS.md`).

## Décisions prises
- Plus de `task_type` : une **scène** = robot + objets `[0..N]` ; le pipe s'adapte au nombre.
- Critère mécanique de découpe : **q-indépendant** (prepare + targets) vs **q-dépendant** (solve).
- **Une seule formulation** du problème de solve (on abandonne le trio V1 `gmr`/`holosoma`/`test`).
- **Solveur enfichable** (Clarabel, ProxQP, …).
- Posage des nuages : **LBS-on-cloud** (skinning creux), validé empiriquement (mesh-free, ~50×
  plus rapide que le forward complet, pas de trous aux articulations).

## Les 3 étapes
```
fichiers bruts ─► PREPARE ─► {GroundedScene, InteractionContext} ─► TARGETS ─► FrameTargets ─► SOLVE ─► qpos
                  (offline, q-indép.)                                            (online, q-indép.)          (q-dép.)
                                                                                      │
                                                                                 trace_frame
                                                                                      ▼
                                                                                    VIZ
```

## Arborescence
Entrée = `SceneSpec` (data identity) + `PrepareConfig` (knobs — `prepare/config.py`). Chaque étage
possède sa surface publique co-localisée : `contracts.py` (types de DONNÉES) + `config.py` (knobs)
+ son point d'entrée.

```
HoloV2/                  racine : CLAUDE.md · .gitignore · docs/ · cache/ · models/ · tests/
  src/             TOUT le code (importé `src.…` ; imports relatifs en interne)
    obs.py           observabilité (Profile/spans), no-op quand off

    prepare/         ÉTAPE 1 — offline ; SEUL endroit qui instancie SMPL/meshes/robot     [PREPARE.md]
      contracts.py     types de DONNÉES de prepare (SORTIE PUBLIQUE) : RobotSpec/SceneSpec · SmplParams/RawMotion ·
                       ObjectMesh/Calibration/GroundedScene · SDF/Channel · GeodesicTable · PointCloud/CorrespondenceTable ·
                       InteractionContext + protocols BodyModel/RobotModel/AssetBuilder — pas la config — numpy-only
      config.py        knobs de prepare : CalibrationConfig/SdfConfig/CloudConfig/CorrespondenceConfig/GeodesicConfig/PrepareConfig
      load/            base · datasets/ (1/dataset -> RawMotion) · smpl (-> BodyModel) · smpl2smplx · mesh (-> verts/faces) · robot (-> RobotModel) · frames
      calibration/     LIVRABLE : grounding scène (humain + objet)
      sdf/             LIVRABLE : SDF objets/terrain/sol (sol plat = SDF de plan exact, non caché)
      point_cloud/     LIVRABLE : nuages (human, objects) + correspondence (SMPL<->G1)
      geodesic/        LIVRABLE : table all-pairs géodésique par mesh (objets/terrain ; sol plat = aucune table)
      scene.py · runner.py   (prepare(scene_spec, config) : load-or-build + assemble)

    targets/         ÉTAPE 2 — construction ONLINE des cibles                          [TARGETS.md]
      contracts.py     types de targets : ContactField/MultiChannelField · StyleTargets ·
                       Robot/EnvironmentInteractionTargets · FrameTargets · FramePose · FrameTrace
                       (importe la sortie publique de prepare via `..prepare.contracts`)
      config.py        knobs de targets : StyleConfig (échelle morpho + hauteurs) / TargetsConfig
      style/           objectif de style (posture, ignore l'objet) -> StyleTargets
      interaction/     pointclouds · eval · transport · targets -> Robot/Env InteractionTargets
      pipeline.py      process_frame -> FrameTargets ; trace_frame -> FrameTrace

    viz/             VISUALISEUR — consommateur pur, zéro hook                         [VIZ.md]
      app.py           viewer prod unifié (Source → VizFrame → Layers) ; core/ + layers/ + model + sources

    solve/           ÉTAPE 3 (à venir) — contracts.py + config.py + point d'entrée     [SOLVE.md]
```

## Graphe de dépendances (acyclique par construction)
Pipeline linéaire ⇒ les imports ne vont que vers l'AVAL ; chaque étage importe la **sortie publique**
de l'amont (`<étage>.contracts` / `.config`), jamais ses internes. Plus de noyau partagé central.
```
prepare/  ──► {GroundedScene, InteractionContext}   (calib dans grounded.calibration ; exposés par prepare/contracts.py + config.py)
targets/  ──► FrameTargets        from ..prepare.contracts import GroundedScene, InteractionContext
viz/      ──► (lit FrameTrace)    from ..prepare.contracts / ..targets.contracts import …
solve/    ──► qpos                from ..targets.contracts import FrameTargets
```
Aucun cycle (deps aval seulement). **`viz` est un consommateur, pas un étage du pipeline** : la règle
« sortie publique seulement » contraint le pipeline linéaire (prepare→targets→solve), pas le visu. Le
viewer de PROD (`viz/app.py`, Source → VizFrame → Layers) respecte la seam (`runner.prepare` + `.contracts`),
mais les viewers de DEBUG *par étage* (`viz/scene`=load, `viz/cloud`=point_cloud, `viz/sdf`=sdf,
`viz/hoim3_multiperson`) pilotent **délibérément le builder interne de l'étage qu'ils visualisent**
(`load.mesh`, `load.smpl`, `sdf.build`, `point_cloud.*`, `calibration`…) pour montrer des intermédiaires
absents des contrats. SMPL/meshes ne sont donc touchés QUE dans `prepare/` **et** ces viewers de debug
(jamais dans `targets`/`solve`).

---

## Plan de l'étape SOLVE (à venir — non figé)

### Constraint Providers (tue la classe-dieu + gère le N-objets)
```python
class ConstraintProvider(Protocol):
    def constraints(self, frame: int, q: np.ndarray, kin) -> list[Constraint]: ...
```
```python
providers = [JointLimits(...), SelfCollision(...), FootLock(...)]
providers += [ObjectContact(o) for o in scene.objects]   # 0, 1 ou N
```
0 objet ⇒ tracking pur, sans aucun `if`. Ajouter une contrainte = un fichier.

### Solveur enfichable
On construit le problème UNE fois en forme canonique `(P,q,A,b,G,h,cones)`, puis on dispatch :
```python
class SolverBackend(Protocol):
    def solve(self, problem) -> Solution: ...   # ClarabelBackend, ProxQPBackend, …
```
`problem` ne connaît aucun solveur ; chaque backend ne connaît rien au retargeting.

## Décisions encore ouvertes (solve)
1. De quelle base dériver la formulation unique (`gmr_socp` ?).
2. Robots : g1 seul ou multi dès le début.
3. Test de parité qpos V2 vs V1 sur une séquence démo.
