# HoloV2 — Architecture (vue d'ensemble)

Carte globale. Détails par étape : `PREPARE.md`, `TARGETS.md`, `VIZ.md`, `CACHE.md`,
`OBS.md` (et `SOLVE.md`, à venir). Source de vérité des types : le package `src/contracts/`.

La **config** est séparée du code ET des données, en deux dossiers au TOP du repo :
**`config_types/`** = les SCHÉMAS (dataclasses `frozen` ; `prepare.py` → `PrepareConfig` + sous-configs
`calibration/sdf/cloud/correspondence`) et **`config_values/`** = la FACTORY qui en instancie les valeurs
(`default_prepare_config() -> PrepareConfig` — point d'entrée unique où des presets/CLI s'attacheront
plus tard). `src/contracts/` ne porte QUE les données qui transitent ; `targets`/`solve` ajouteront
leur module dans ces deux dossiers. Les clés du cache build-once dérivent de la config (voir `CACHE.md`).

## Règle d'or : zéro spaghetti
1. **Dépendances à sens unique**, jamais de cycle.
2. **Une responsabilité par module.** Pas de classe-dieu (le retargeter V1 fait ~7 métiers
   dans une classe — on explose).
3. **Contrats typés aux frontières** (dataclasses frozen dans le package `contracts/`), pas de
   `SimpleNamespace`.
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
fichiers bruts ─► PREPARE ─► {GroundedScene, InteractionContext, Calibration} ─► TARGETS ─► FrameTargets ─► SOLVE ─► qpos
                  (offline, q-indép.)                                            (online, q-indép.)          (q-dép.)
                                                                                      │
                                                                                 trace_frame
                                                                                      ▼
                                                                                    VIZ
```

## Arborescence
Entrée = `SceneSpec` (data identity) + `PrepareConfig` (knobs ; schéma dans `config_types/`, valeurs
via la factory de `config_values/`). Source de vérité des types de DONNÉES : le package `src/contracts/`.

```
HoloV2/                  racine : CLAUDE.md · .gitignore · docs/ · cache/ · models/ · tests/
  config_types/    SCHÉMAS de config (dataclasses frozen), 1 module/étape : prepare.py (targets/solve à venir)
  config_values/   FACTORY des valeurs, 1 module/étape : prepare.py -> default_prepare_config() -> PrepareConfig
  src/             TOUT le code (importé `src.…` ; imports relatifs en interne)
    contracts/       contrats de DONNÉES (package par domaine : protocols/inputs/motion/scene/fields/assets/targets ;
                     __init__ ré-exporte tout) — pas la config — ne dépend de rien
    obs.py           observabilité (Profile/spans), no-op quand off

    prepare/         ÉTAPE 1 — offline ; SEUL endroit qui instancie SMPL/meshes/robot     [PREPARE.md]
      load/            base · datasets/ (1/dataset -> RawMotion) · smpl (-> BodyModel) · smpl2smplx · mesh (-> verts/faces) · robot (-> RobotModel) · frames
      calibration/     LIVRABLE : grounding scène (humain + objet)
      sdf/             LIVRABLE : SDF objets/terrain/sol (sol plat = SDF de plan exact, non caché)
      point_cloud/     LIVRABLE : nuages (human, objects) + correspondence (SMPL<->G1)
      scene.py · runner.py   (prepare(scene_spec, config) : load-or-build + assemble)

    targets/         ÉTAPE 2 — construction ONLINE des cibles                          [TARGETS.md]
      style/           objectif de style (posture, ignore l'objet) -> StyleTargets
      interaction/     pointclouds · eval · transport · targets -> Robot/Env InteractionTargets
      pipeline.py      process_frame -> FrameTargets ; trace_frame -> FrameTrace

    viz/             VISUALISEUR — consommateur pur, zéro hook                         [VIZ.md]
      viewer.py        gros viewer viser, toggles ; lit FrameTrace + assets prepare

    solve/           ÉTAPE 3 (à venir)                                                 [SOLVE.md]
```

## Graphe de dépendances (acyclique)
```
contracts/   ◄── prepare/ , targets/ , viz/ , solve/
prepare/  ──► {GroundedScene, InteractionContext, Calibration}
targets/  ──► FrameTargets        (consomme les sorties prepare via leur TYPE, jamais le code)
viz/      ──► (lit FrameTrace + assets prepare)
solve/    ──► qpos                 (consomme FrameTargets)
```
SMPL/meshes ne sont jamais touchés hors `prepare/`.

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
