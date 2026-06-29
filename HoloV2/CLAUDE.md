# HoloV2 — guide d'implémentation

Réécriture épurée du pipeline de retargeting (humain → robot G1). Objectif : **clair, simple,
bien structuré, sans spaghetti, adaptable aux cas complexes** (0/1/N objets, terrain, mains,
multi-robot, online). L'ancienne version est le dossier frère `../HoloNew/` — **réf. pour porter
la LOGIQUE algorithmique, jamais la structure** (voir la carte de portage plus bas).

## Règles d'or (NON négociables)

1. **Dépendances à sens unique, zéro cycle.** Pipeline LINÉAIRE `prepare → targets → solve` : l'aval
   importe la **sortie publique** de l'amont, jamais ses internes ⇒ acyclique par construction. Une
   responsabilité par module. Pas de classe-dieu.
2. **Chaque étage possède SES types ET SA config**, co-localisés — pas de noyau partagé central.
   Surface publique d'un étage = son `contracts.py` (types de DONNÉES, dataclasses `frozen`, numpy-only)
   + son `config.py` (knobs, dataclasses `frozen`) + son point d'entrée public (`prepare.runner.prepare`).
   L'aval importe la sortie publique de l'amont (`targets` fait
   `from ..prepare.contracts import GroundedScene, InteractionContext`), JAMAIS un sous-module interne
   (`prepare/load/*`, …). Un type/knob ne se duplique JAMAIS ailleurs (docs incluses : on y pointe).
   Config de `prepare` = `prepare/config.py` : `PrepareConfig()` = défaut, override inline
   `PrepareConfig(sdf=SdfConfig(spacing=0.005))` ; un CLI tyro s'y attachera avec le point d'entrée de run.
3. **Cœur pur, effets de bord aux extrémités.** Les ops de calcul ne font ni I/O, ni log, ni
   mutation de leurs inputs ; elles prennent des données et rendent un artefact `frozen`. Disque
   dans `prepare/load`, écran dans `viz/`.
4. **Data-oriented (vitesse).** Structure-of-Arrays numpy, layout **canal-first `(C,P)`**,
   vectorisé (zéro boucle Python par point), tout le lourd amorti offline.
5. **Observabilité aux seams.** `obs.Profile` (spans) **dans les orchestrateurs uniquement**,
   jamais dans les ops pures. No-op quand off.
6. **Visu = consommateur.** `viz/` lit `FrameTrace` ; zéro hook de visu dans le calcul.
7. **Critère de découpe = dépendance à `q`** (config robot optimisée par le solveur) :
   indépendant ⇒ `prepare` (offline) + `targets` (online) ; dépendant ⇒ `solve`.
8. **Homogénéité plutôt que cas particuliers.** Mêmes fonctions humain/objets/robot
   (`pose_cloud`, `eval_fields`) ; canaux uniformes (sol + N objets) ; `K=1` rigide vs `K=N` blend.
9. **Adaptabilité par abstraction — MAIS YAGNI.** Préférer un protocol à une cascade de `if`
   **quand un 2ᵉ cas existe déjà ou est imminent**. Sinon, fonction simple. Pas de couche « au cas où ».

## Architecture

Carte complète + flux : `docs/ARCHITECTURE.md` (source unique de la structure ; **ne pas la
redupliquer ici**). En bref : `SceneSpec`+`PrepareConfig` → **prepare/** (offline, build-once cachés) →
`{GroundedScene, InteractionContext, Calibration}` → **targets/** (online per-frame) →
`FrameTargets` → **solve/** (à venir). `viz/` et `obs.py` = consommateurs. Détails par étape :
`PREPARE/TARGETS/VIZ/CACHE/OBS.md`.

## Conventions de code

- **Poids des imports** : chaque `<étage>/contracts.py` + `<étage>/config.py` et `src/obs.py` =
  **numpy-only** (la config = stdlib-only, dataclasses) — tous légers, importables partout.
  torch/smplx/trimesh/coal/pinocchio **uniquement** dans `src/prepare/load/` et les builders
  `src/prepare/`. `targets`/`solve`/`viz` ne dépendent jamais de torch. **Exception assumée** : les
  viewers de DEBUG par étage (`viz/{scene,cloud,sdf,hoim3_multiperson}`) importent `trimesh` en LAZY
  (dans la fonction, pour le rendu mesh) et pilotent les internes de l'étage qu'ils visualisent — ils
  sont des consommateurs, pas le pipeline (cf. `docs/ARCHITECTURE.md`). Le viewer prod reste sur la seam.
- **Imports** : relatifs DANS `src/` ; chaque étage importe la **sortie publique** de l'amont
  (`from ..prepare.contracts import X`, `from ..prepare.config import PrepareConfig`), jamais un
  sous-module interne. Tests : absolus (`from src.… import …`).
- **Types partout** ; tableaux annotés `np.ndarray` + **forme en commentaire** (`# (T, J, 3)`).
- **dtype** : compute en `float64` ; arrays **stockés/cachés** en `float32` (grilles SDF, nuages).
- **Erreurs** : valider les invariants de contrat par un `raise ValueError` explicite à la
  construction/assemblage (cf. `MultiChannelField.__post_init__`) ; `assert` réservé aux
  invariants internes.
- **Quaternions wxyz** ; poses `(x,y,z,qw,qx,qy,qz)`.
- **`J_demo`** (joints dataset, `style`) ≠ **`J_bones`** (squelette SMPL, nuages) — jamais confondus.
- **Per-frame = unité canonique** ; séquence = `list[FrameTargets]`.
- Nuages = **skinning creux** `(parts, weights, offsets)`, posés mesh-free (`K=1` rigide / `K~4` humain).
- Sol = canal `ground` : **toujours un `SDF`** — plan exact par défaut (`build_plane_sdf`, non caché)
  OU **terrain** (escalier/pente/climbing). `Channel.sdf` jamais `None` ⇒ éval homogène (chemin unique).
- SMPL/meshes/robot instanciés **uniquement dans `prepare/`**.
- **Commentaires autonomes** (pas de réf. à une discussion ; acronymes locaux définis).

## Tests & vérification (obligatoire)

Tests dans `HoloV2/tests/` (pytest ; `pip install pytest` dans l'env si absent), lancés avec
le python de l'env `holonew` : `…/envs/holonew/bin/python -m pytest tests/ -q` (depuis `HoloV2/`).
- **Op pure** → un test unitaire (petit cas synthétique : formes + une valeur connue).
- **Builder** → test de **déterminisme** (build ×2 ⇒ identique) **+ round-trip cache**
  (`save`→`load` == `build`). Sinon le cache est faux.
- **Portage V1** → test de **parité** vs sortie V1 sur une séquence démo (tolérance documentée).
- **Perf** → pour un chemin chaud, vérifier le timing via `obs.Profile.render()`.
- Après tout changement d'un `contracts.py`/`config.py` d'étage : `python -m py_compile` **+** un import dans l'env.

## Workflow par module (« definition of done »)

1. Le type est dans le `contracts.py` de l'étage qui le produit (le knob dans son `config.py`).
2. Logique = fonction(s) pure(s), sans effet de bord, sans muter les inputs.
3. Câblée dans l'orchestrateur (`runner`/`pipeline`) avec un `prof.span(...)`.
4. Test unitaire (+ déterminisme/parité si pertinent).
5. Import OK dans l'env ; doc d'étape mise à jour si le contrat a bougé.

## Carte de portage V1 (`../HoloNew/`) → V2

| Module V2 | Source(s) V1 à porter |
|---|---|
| `prepare/load/` (datasets→RawMotion) | `src/data_loaders/*`, `src/utils.py` (load_intermimic…), `data_type.py` (mappings V1) |
| `prepare/load/smpl.py` (BodyModel) | `src/test_socp/correspondence/human_body.py`, `src/data_loaders/hodome.py` |
| `prepare/load/robot.py` (RobotModel) | `src/robot_fk.py`, `src/test_socp/pin_model.py`, correspondence `g1_surface.py` |
| `prepare/calibration/` | `src/holosoma/preprocess.py` (ground/scale), `contact/smplx_field.robust_floor_offset`, omomo betas-FK |
| `prepare/sdf/` (SDF objets/terrain) | `src/test_socp/contact/backends/sdf.py` + `backends/coal.py` |
| `prepare/point_cloud/human|objects` | `human_body.build_point_cloud_cache`, `utils.weighted_surface_sampling`, `movable.sample_object_surface` |
| `prepare/point_cloud/correspondence/` | `src/test_socp/correspondence/*` (build, ot_couple, transport, segments) |
| `targets/interaction/eval` | `contact/contact_field.py`, `contact/combined.py`, `backends/floor.py` |
| `targets/interaction/transport` | `correspondence/transport.py` |
| `targets/style/` | `src/gmr_socp/*`, mappings `data_type.py` (V1) |
| `viz/` | `src/viewer.py`, `viser_player.py`, `correspondence/viz.py`, `contact/viz.py` |
| sol plat → SDF de plan (`prepare/sdf.build_plane_sdf`) | `contact/backends/floor.py` (convention dist/dir) |

## Cache (`HoloV2/cache/`, hors package, gitignoré sauf `corr_neutral.npz`)

Caché **par dépendance**, pas par scène : `cache_key = hash(sous-config pertinente + inputs +
clés amont)`. Garde-fou asserté : `PointCloud.sampling_id == CorrespondenceTable.smpl_sampling_id`.
Détails : `docs/CACHE.md`.

## Environnement & données (machine locale)

- Python : `~/.holonew_deps/miniconda3/envs/holonew/bin/python` (numpy, torch, smplx, trimesh,
  viser…). Installé par `../scripts/setup_retargeting.sh`.
- Données/modèles : chemins dans `../HoloNew/path.yaml` (OMOMO, HODome, SFU ; SMPL-X sous
  `…/models/models_smplx_v1_1/models`). Démo : `../HoloNew/demo_data/`.

## Commits

- **NE JAMAIS tagger Claude** (aucun `Co-Authored-By`, aucune mention Claude/Anthropic).
- Auteur : `Guillaume-Bsst <guibesset@free.fr>`. Commits conventionnels.

## État & feuille de route

Contrats + squelette posés (`prepare/contracts.py`+`config.py`, `targets/contracts.py`, `obs.py` complets ; modules = stubs).
Ordre d'implémentation (dépendances) : **`prepare/load/base.py`** → un loader concret (OMOMO/SFU)
→ `calibration/` → `sdf/` + `point_cloud/` (+ `correspondence`) → `targets/interaction` →
`targets/style` → `viz/` → `solve/`. À chaque étape : tests + parité V1.
