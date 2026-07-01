# HoloV2 — Cheatsheet (commandes)

Aide-mémoire des commandes pour **lancer, visualiser, tester et débugger** le pipeline
`prepare → targets → solve`. Pour le « pourquoi », voir `CLAUDE.md` + `docs/*.md`.

> ⚠️ Toujours utiliser le **python de l'env `holonew`** (cvxpy/pinocchio/mujoco/smplx/viser)
> et lancer **depuis `HoloV2/`** (les chemins `models/...` et le package `src` sont relatifs).

---

## 0. Setup d'environnement (à coller dans le terminal)

```bash
# Racine du paquet + python de l'env
export HOLOV2=/home/gbesset/Documents/wbt_rl/modules/01_retargeting/HoloNew/HoloV2
export PY=~/.holonew_deps/miniconda3/envs/holonew/bin/python
cd "$HOLOV2"                      # IMPORTANT : tout se lance depuis ici

# Données + modèles (chemins réels, cf. ../HoloNew/path.yaml)
export DATA=/home/gbesset/Documents/wbt_rl/data/00_raw_datasets
export SMPLX=$DATA/models/models_smplx_v1_1/models/smplx   # contient SMPLX_NEUTRAL.npz
```

> 💡 Plutôt que `$DATA`/`$SMPLX` à la main, copie une fois `paths.example.toml` → `paths.toml`
> (gitignoré) et édite tes chemins. C'est le registre **central** de tous les fichiers externes
> machine-local : section `[models]` (`smplx` requis ; `smplh`, `smpl2smplx` deftrafo optionnels,
> HOI-M3) et une section `[datasets.<nom>]` par dataset avec `motion` (base d'un `--motion-path`
> relatif) et `meta` (release betas/scales/objets ; défaut = `motion`). Les viewers remplissent
> `--model-dir`/`--dataset-root` depuis là, et `--motion-path` peut être **relatif** à `motion`.

```bash
cp paths.example.toml paths.toml      # une fois, puis éditer les chemins de ta machine
```

Exemple `paths.toml` (extrait) :

```toml
[models]
smplx = "/abs/.../models_smplx_v1_1/models/smplx"   # requis (tous datasets)
smplh = "/abs/.../models/smplh"                      # HOI-M3 (sinon dérivé de smplx)
# smpl2smplx = "/abs/.../model_transfer/smpl2smplx_deftrafo_setup.pkl"  # HOI-M3 (.pkl deftrafo)
[datasets.hodome]
motion = "/abs/.../HODome"
[datasets.omomo]
motion = "/abs/.../OMOMO_new/OMOMO_new"   # les .pt
meta   = "/abs/.../OMOMO"                  # betas/scales + captured_objects/
[datasets.hoim3]
motion = "/abs/.../HOI-M3/mocap_ground"
```

Vérif rapide que l'env est bon :

```bash
$PY -c "import cvxpy, pinocchio, mujoco, smplx, viser, trimesh; print('env OK')"
$PY -c "import sys; print(sys.executable)"   # doit pointer vers .../envs/holonew/bin/python
```

---

## 1. Visualiseurs (viser, dans le navigateur → http://localhost:8080)

### 🎬 Démos prêtes à l'emploi (copier-coller)

> Prérequis : le **Setup §0** (`export PY=…`, `export HOLOV2=…`, `cd "$HOLOV2"`) + un `paths.toml`
> configuré (le tien l'est). Puis ouvre **http://localhost:8080**. Séquences ci-dessous vérifiées
> présentes sur cette machine.

```bash
# ── DÉMO A · humain + cibles (pré-solve, rapide) — HODome "baseball" (avec objet)
$PY -m src.viz.app --dataset hodome --motion-path smplx/subject01_baseball.npz --max-frames 30

# ── DÉMO B · ⭐ LE ROBOT G1 RÉSOLU + cost dashboard (un solve/frame → max-frames bas, ~1 min de bake)
$PY -m src.viz.app --dataset hodome --motion-path smplx/subject01_baseball.npz --max-frames 8 --solve

# ── DÉMO C · locomotion sans objet — SFU jogging (ajoute --solve pour le robot)
$PY -m src.viz.app --dataset sfu --motion-path 0005/0005_Jogging001_stageii.npz --max-frames 30

# ── DÉMO D · check INSTANTANÉ sans aucune donnée — SDF d'un sol plan (pure géométrie)
$PY -m src.viz.debug.sdf --plane 4.0 --spacing 0.02
```

Dans le navigateur : les dossiers de gauche = une checkbox par couche (actives **même en pause**).
Avec `--solve`, coche **robot G1 résolu** / **contact robot cible↔atteint** (+ **witness cible/atteint**) /
**contacts objets** / **cloud objet résolu** (vert vs source) / **lignes SMPL↔G1** ; le panel
**cost dashboard** trace le coût par terme (frames non-convergées en rouge). Port occupé → `fuser -k 8080/tcp`.

### Détail des flags

Le viewer prod (`src.viz.app`) + les debug `scene`/`cloud` partagent les mêmes flags de scène :
`--dataset`, `--motion-path`, `--model-dir`, `--dataset-root` (objets/betas), `--port 8080`,
`--frame-step N`, `--max-frames N`, `--person-id`, `--object-names a,b`. (`debug.sdf` et
`debug.hoim3` ont **leurs propres parsers** — voir plus bas.) **Garde `--max-frames` bas** (debug :
3–30 ; **3–10 avec `--solve`**) — un bake long est lent, un solve par frame encore plus.

> 💡 Libère le port D'ABORD si besoin : `fuser -k 8080/tcp` (ne jamais `pkill -f` le script viewer :
> il se tuerait lui-même).

### Viewer principal unifié — `src.viz.app` (humain/cibles **+ robot résolu**)

`BakeSource → Player → 13 couches composables`, chacune avec sa checkbox (actives **même en pause**) :
sol (SDF réel) · ghost SMPL · squelette · clouds objets (source + **résolu**) · champ humain
(witness/normales) · cibles style (points/frames/labels) · **robot G1 résolu** · contact robot
cible-vs-atteint (+ **witness cible/atteint**) · **contacts objets** (cible/atteint + witness, en ref objet) ·
lignes SMPL↔G1 · iso-surface SDF · champ géodésique. GUI : dossier **Playback** (slider/play/fps) + **Selectors**
(canal / mode couleur uniform-distance-active / taille des points).

```bash
# Pré-solve (humain + cibles ; le robot et les couches solve-gated restent masqués)
# Forme courte : lit [models].smplx + [datasets.<nom>] de paths.toml ; --motion-path relatif à `motion`
$PY -m src.viz.app --dataset hodome --motion-path smplx/subject01_baseball.npz --max-frames 30
$PY -m src.viz.app --dataset sfu    --motion-path 0005/0005_Jogging001_stageii.npz --max-frames 30
$PY -m src.viz.app --dataset omomo  --motion-path sub10_clothesstand_000.pt --max-frames 30
```

```bash
# --solve : BakeSource lance le solveur SQP par frame → rend le G1 RÉSOLU (ViserUrdf) superposé à
#           l'humain/aux cibles + active le panel "cost dashboard" (coût par terme empilé sur la
#           séquence, frames non-convergées marquées en rouge). Lent → garde --max-frames très bas.
$PY -m src.viz.app --dataset hodome --motion-path smplx/subject01_baseball.npz --max-frames 8 --solve
```

```bash
# HODome (objets) — exemple complet, chemins explicites (marche sans paths.toml)
$PY -m src.viz.app --dataset hodome \
    --motion-path $DATA/HODome/smplx/subject01_baseball.npz \
    --model-dir $SMPLX --dataset-root $DATA/HODome \
    --max-frames 30 --frame-step 2 --port 8080 [--solve]
```

### Étapes de debug incrémentales (viewers focalisés — `src/viz/debug/`)

```bash
# debug/scene.py : étape load/grounding (mesh SMPL posé, squelette, objets, sol)
$PY -m src.viz.debug.scene  --dataset hodome \
    --motion-path $DATA/HODome/smplx/subject01_baseball.npz \
    --model-dir $SMPLX --dataset-root $DATA/HODome --max-frames 30

# debug/cloud.py : bake du point_cloud (nuage humain posé + nuages objets) ; --corr = table OT
$PY -m src.viz.debug.cloud  --dataset hodome \
    --motion-path $DATA/HODome/smplx/subject01_baseball.npz \
    --model-dir $SMPLX --dataset-root $DATA/HODome \
    --corr cache/correspondence/corr_neutral.npz --max-frames 30

# debug/sdf.py : inspecter un SDF (objet/terrain) ou un sol plan
$PY -m src.viz.debug.sdf --mesh models/largebox/largebox.obj --spacing 0.02 --margin 0.05
$PY -m src.viz.debug.sdf --plane 4.0 --spacing 0.02         # sol plat 4×4 m

# debug/hoim3.py : scène multi-personnes (HOI-M3)
$PY -m src.viz.debug.hoim3 --motion-path <human.npz> --model-dir $SMPLX --max-frames 30
```

### Autres datasets (mêmes flags, change `--dataset` + chemins)

```bash
# OMOMO (InterMimic .pt ; dataset-root = release OMOMO pour betas/scale/meshes)
$PY -m src.viz.app --dataset omomo \
    --motion-path $DATA/OMOMO_new/OMOMO_new/sub10_clothesstand_000.pt \
    --model-dir $SMPLX --dataset-root $DATA/OMOMO --max-frames 30

# SFU (locomotion, sans objet ni dataset-root)
$PY -m src.viz.app --dataset sfu \
    --motion-path $DATA/SFU/SFU/0005/0005_Jogging001_stageii.npz \
    --model-dir $SMPLX --max-frames 30
```

Datasets enregistrés (`@register_loader`) : `hodome`, `omomo`, `sfu`, `hoim3`.
Sur machine distante / X manquant : redirige le port (`ssh -L 8080:localhost:8080 …`).

---

## 2. Builds offline (caches `prepare/`)

```bash
# Reconstruire la correspondance OT humain↔robot (table cachée)
$PY -m src.prepare.point_cloud.correspondence.build \
    --model-dir $SMPLX --urdf models/g1/g1_29dof.urdf \
    --robot-name g1 --out cache/correspondence/corr_neutral.npz
```

Les caches vivent dans `HoloV2/cache/` (gitignoré sauf `corr_neutral.npz`). Clé de cache =
hash(sous-config + inputs + clés amont) → **changer la config invalide le cache automatiquement**.
Pour forcer un rebuild : supprimer le `.npz`/`.npy` concerné dans `cache/`. Détails : `docs/CACHE.md`.

---

## 3. Tests (pytest)

```bash
cd "$HOLOV2"

# Toute la suite (⚠ peut être long : quelques tests font des solves complets)
$PY -m pytest tests/ -q

# Un fichier / un test ciblé (préféré pendant l'itération)
$PY -m pytest tests/test_style.py -q
$PY -m pytest tests/test_runner_prepare.py::test_xxx -q

# Filtrer par mot-clé, s'arrêter au 1er échec, traceback court
$PY -m pytest tests/ -q -k "corr or sdf" -x --tb=short

# Voir les prints / logs (pas de capture) + le plus verbeux
$PY -m pytest tests/test_sdf.py -s -vv
```

Convention de tests (cf. `CLAUDE.md` §Tests) :
- **op pure** → cas synthétique (formes + 1 valeur connue) ;
- **builder** → déterminisme (build ×2 identique) + round-trip cache (`save`→`load` == `build`) ;
- **portage V1** → test de parité vs `../HoloNew/` (tolérance documentée).

---

## 4. Debug & vérification rapide

```bash
# Compilation + import d'un module (check obligatoire après tout changement contracts.py/config.py)
$PY -m py_compile src/prepare/contracts.py src/prepare/config.py
$PY -c "from src.prepare.config import PrepareConfig; print(PrepareConfig())"

# Tester un loader/scène en REPL sans viewer
$PY -c "
from pathlib import Path
import os
from src.prepare.contracts import SceneSpec, RobotSpec
from src.prepare.config import PrepareConfig
from src.prepare.runner import prepare
robot = RobotSpec(name='g1', urdf_path=Path('models/g1/g1_29dof.urdf'), link_names=('pelvis',), dof=29, height=1.3)
spec = SceneSpec(dataset='sfu',
                 motion_path=Path(os.environ['DATA'])/'SFU/SFU/0005/0005_Jogging001_stageii.npz',
                 robot=robot, smpl_model_dir=Path(os.environ['SMPLX']))
out = prepare(spec, PrepareConfig())   # 2e arg requis ; override : PrepareConfig(sdf=SdfConfig(spacing=0.005))
print(type(out))                       # {GroundedScene, InteractionContext, Calibration}
"
```

### Profiling (observabilité aux seams — `src/obs.py`)

`obs.Profile` instrumente les orchestrateurs (`runner`/`pipeline`) via `prof.span(...)` ;
no-op quand désactivé. Pour voir les timings d'un chemin chaud :

```bash
$PY -c "
from src.obs import Profile
prof = Profile(enabled=True)
with prof.span('demo'):
    pass
print(prof.render())          # arbre des spans + durées
"
```

### Réflexes de debug
- **`ParseXML: Error opening file …`** → tu n'es pas dans `HoloV2/` (chemins `models/...` relatifs).
- **`ModuleNotFoundError: cvxpy/pinocchio/...`** → mauvais python ; utilise `$PY` (env `holonew`).
- **Bake/solve trop lent** → baisse `--max-frames` (3–10) et/ou monte `--frame-step`.
- **Cache douteux** → supprime le fichier dans `cache/` et relance ; garde-fou asserté
  `PointCloud.sampling_id == CorrespondenceTable.smpl_sampling_id`.
- **Port 8080 occupé** → `--port 8081` (ou `lsof -i :8080`).

---

## 5. Carte des docs (le « pourquoi »)

| Fichier | Contenu |
|---|---|
| `CLAUDE.md` | règles d'or, conventions, workflow par module, carte de portage V1→V2 |
| `docs/ARCHITECTURE.md` | structure complète + flux `prepare → targets → solve` |
| `docs/PREPARE.md` | étage offline (load, calibration, sdf, point_cloud, correspondence) |
| `docs/TARGETS.md` | étage online per-frame (interaction + style) |
| `docs/VIZ.md` | viewer unifié `app` (seam `Source→VizFrame→Layers`, robot résolu, cost dashboard) + viewers debug sous `debug/` |
| `docs/CACHE.md` | stratégie de cache par dépendance |
| `docs/OBS.md` | `obs.Profile` (spans, profiling) |
</content>
</invoke>
