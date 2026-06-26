# HoloV2 — Architecture (vue d'ensemble)

Carte globale. Détails par étape : `PREPARE.md`, `TARGETS.md`, `VIZ.md`, `CACHE.md`,
`OBS.md` (et `SOLVE.md`, à venir). Source de vérité des types : `holov2/contracts.py`.

`prepare` et `targets` sont pilotés par une **`Config`** (`contracts.Config` : sous-configs
`calibration/sdf/cloud/correspondence` + `margin`). Les clés du cache build-once en dérivent
(voir `CACHE.md`).

## Règle d'or : zéro spaghetti
1. **Dépendances à sens unique**, jamais de cycle.
2. **Une responsabilité par module.** Pas de classe-dieu (le retargeter V1 fait ~7 métiers
   dans une classe — on explose).
3. **Contrats typés aux frontières** (dataclasses frozen dans `contracts.py`), pas de
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
Entrée = `SceneSpec` (data identity) + `Config` (knobs). Source de vérité des types :
`contracts.py` (+ `fields.py` pour les impls de champ).

```
holov2/
  contracts.py     TOUS les contrats (SceneSpec, RobotSpec, Config, assets, cibles, …) — ne dépend de rien
  fields.py        impls de Field : GridSDF (grille) + PlaneField (sol analytique)

  prepare/         ÉTAPE 1 — offline ; SEUL endroit qui instancie SMPL/meshes/robot     [PREPARE.md]
    load/            base + 1/dataset (-> RawMotion) · smpl (-> BodyModel) · mesh (-> ObjectMesh) · robot (-> RobotModel)
    calibration/     LIVRABLE : grounding scène (humain + objet)
    sdf/             LIVRABLE : GridSDF objets (sol = PlaneField, non caché)
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
contracts.py ◄── prepare/ , targets/ , viz/ , solve/
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
