# Spec — Résolveur de chemins V2 (`paths.toml`)

**Date** : 2026-06-29 · **Étage** : bord/CLI (hors pipeline pur) · **Statut** : implémenté

> **Révision (post-audit, même jour).** À la demande de l'utilisateur (« tous les fichiers
> externes qui impactent le pipe doivent être dans le TOML »), un audit exhaustif des lectures
> disque du pipeline a élargi le périmètre du schéma plat initial (`smplx` + `[roots]`) vers un
> schéma **`[models]` + `[datasets.<nom>]`** détaillé. Cette spec décrit l'état final. L'audit a
> classé chaque lecture en : machine-local (→ TOML), in-repo (URDF/meshes robot — repo-relatif,
> hors TOML, décision utilisateur), cache régénérable (`HoloV2/cache/`), ou per-run (CLI).

## Problème

Lancer les viewers/tests impose de retaper à la main les chemins absolus (`--model-dir`,
`--dataset-root`, `--motion-path` complet) à chaque commande. On veut une source de vérité
machine-locale pour ne plus les répéter — sans violer les règles d'or de `CLAUDE.md`.

## Objectifs

- Une **source de vérité unique, V2-local et gitignorée** pour les racines machine-spécifiques
  (dataset roots + dossier SMPL-X).
- Les `main()` CLI remplissent leurs défauts depuis cette source ; on passe juste
  `--dataset` + un `--motion-path` court (relatif à la racine du dataset).
- **Rétro-compatibilité** : les commandes avec chemins absolus continuent de marcher.

## Non-objectifs (YAGNI)

- Pas de « scènes nommées » / presets (`--scene baseball`).
- Pas de chemins dans `config.py`/`PrepareConfig` (ce ne sont pas des knobs algorithmiques).
- Pas de dépendance sur `../HoloNew/path.yaml` (V2 reste découplé du legacy).
- L'URDF robot + ses meshes (in-repo, `HoloV2/models/`) **ne** sont **pas** dans le TOML
  (versionnés, pas machine-local ; résolus repo-relatif par `_g1_robot()`).
- Le résolveur ne vérifie pas l'existence des fichiers (les loaders dégradent/raisent).

## Conformité aux règles d'or (`CLAUDE.md`)

- **#3 (cœur pur, effets aux extrémités)** : la résolution disque est un effet de **bord**.
  Seuls les points d'entrée (`main()` viz, futur `run`) appellent le résolveur ;
  `prepare`/`targets` reçoivent un `SceneSpec` déjà résolu et n'importent **jamais** `paths.py`.
- **#2 (pas de noyau partagé central / pas de duplication)** : les chemins ne sont pas des
  types ni des knobs d'étage → hors `config.py`. `paths.py` n'est importé que par les entrées,
  donc pas de hub, pas de cycle. La factorisation `_scene_args` supprime la duplication
  actuelle du bloc `RobotSpec`/`SceneSpec` entre les 4 `main()` viz.

## Format : TOML lu par `tomllib`

`tomllib` est en stdlib (Python 3.11 de l'env `holonew`) → **zéro dépendance**. `pyyaml` n'est
pas installé (V1 avait dû hand-roller un parseur YAML à plat). TOML supporte nesting + commentaires.

### `HoloV2/paths.toml` (gitignoré, machine-local — source de vérité)

Schéma `[models]` (assets modèles) + `[datasets.<nom>]` (`motion` = base d'un `--motion-path`
relatif ; `meta` = release betas/scales/objets, défaut = `motion`) :

```toml
[models]
smplx      = "/abs/.../models_smplx_v1_1/models/smplx"   # dir SMPLX_*.npz — REQUIS (tous datasets)
smplh      = "/abs/.../models/smplh"                      # dir <gender>/model.npz — HOI-M3, OPTIONNEL
smpl2smplx = "/abs/.../model_transfer/smpl2smplx_deftrafo_setup.pkl"  # .pkl deftrafo — HOI-M3, OPTIONNEL

[datasets.hodome]
motion = "/abs/.../HODome"                # smplx/<seq>.npz ; object/ + scaned_object/ frères (meta=motion)
[datasets.sfu]
motion = "/abs/.../SFU/SFU"               # <subject>/<seq>.npz ; pas d'objets
[datasets.omomo]
motion = "/abs/.../OMOMO_new/OMOMO_new"   # InterMimic .pt
meta   = "/abs/.../OMOMO"                  # pickles betas/scales + captured_objects/
[datasets.hoim3]
motion = "/abs/.../HOI-M3/mocap_ground"   # <seq>_human.npz ; _object.npz + scanned_object/ dérivés
```

`smplh`/`smpl2smplx` omis ⇒ **dérivés par convention** de `smplx` (`parents[2]/smplh`,
`parents[2]/model_transfer/...`). `meta` omis ⇒ défaut = `motion`.

### `HoloV2/paths.example.toml` (committé — template)

Copie de la structure ci-dessus avec des chemins placeholder/relatifs. Onboarding :
`cp paths.example.toml paths.toml` puis éditer.

### `.gitignore` (HoloV2)

Ajouter `paths.toml` (garder `paths.example.toml` versionné).

## Composants

### `src/paths.py` — résolveur pur (stdlib only)

Ancré sur `HOLOV2_ROOT = Path(__file__).resolve().parents[1]`. Aucun import de
`prepare`/`targets`/argparse → léger et testable isolément.

| Fonction | Rôle |
|---|---|
| `load_paths(path=None) -> dict` | lit `paths.toml` via `tomllib`; `FileNotFoundError` clair pointant vers `paths.example.toml` si absent |
| `smplx_dir(cfg=None) -> Path` | `[models].smplx`; **requis** → `ValueError` si absent |
| `smplh_dir(cfg=None) -> Path \| None` | `[models].smplh`; **optionnel** → `None` si absent |
| `smpl2smplx_pkl(cfg=None) -> Path \| None` | `[models].smpl2smplx`; **optionnel** → `None` si absent |
| `dataset_motion_root(name, cfg=None) -> Path` | `[datasets.name].motion`; **requis** (résolution relative) → `ValueError` si absent |
| `dataset_meta_root(name, cfg=None) -> Path \| None` | `meta` sinon `motion` sinon `None` (dataset absent) |
| `resolve_motion(name, motion, cfg=None) -> Path` | absolu → tel quel; relatif → `dataset_motion_root(name)/motion` |

`smplh`/`smpl2smplx` (optionnels, `None` par défaut) sont threadés au bord dans deux nouveaux
champs `SceneSpec.smplh_dir` / `SceneSpec.smpl2smplx_pkl` ; le loader HOI-M3 (`_resolve_assets`)
les utilise s'ils sont fournis, sinon **fallback convention** (`parents[2]`). Edge-only préservé :
le loader lit ces **champs de `SceneSpec`** (son contrat d'entrée), jamais `paths.toml`.

### `src/viz/_scene_args.py` — colle argparse → `SceneSpec` (partagée par les mains viz)

- `add_scene_args(ap)` : déclare `--dataset`, `--motion-path`, `--model-dir` (optionnel),
  `--dataset-root` (optionnel), `--person-id`, `--object-names`, `--port`, `--frame-step`,
  `--max-frames`.
- `scene_from_args(a) -> SceneSpec` :
  - chargement **tolérant** : `paths.toml` n'est requis que si un défaut en dépend
    (`--model-dir` absent **ou** motion relative) ; sinon `cfg = {}` (les invocations absolues
    marchent sans `paths.toml`).
  - `--model-dir` ← `a.model_dir` sinon `paths.smplx_dir(cfg)`.
  - `--dataset-root` ← `a.dataset_root` sinon `paths.dataset_meta_root(a.dataset, cfg)` (`None`
    si dataset absent).
  - `motion_path` ← `paths.resolve_motion(a.dataset, a.motion_path, cfg)`.
  - `smplh_dir`/`smpl2smplx_pkl` ← `paths.smplh_dir(cfg)`/`paths.smpl2smplx_pkl(cfg)` (`None` si absent).
  - `RobotSpec` ← helper `_g1_robot()` (vrai URDF repo-relatif `models/g1/g1_29dof.urdf`,
    single-sourced ; remplace le placeholder `Path("g1.urdf")`).
- Adoptée par `viz/viewer.py`, `viz/scene.py`, `viz/cloud.py` (supprime la duplication).
  `viz/hoim3_multiperson.py` : `main()` aligné (même chargement tolérant + `resolve_motion("hoim3", …)`
  + population `smplh`/`smpl2smplx`), garde sa forme CLI propre (pas de `--dataset`).

## Flux

```
paths.toml ──load_paths──> dict ──┐
CLI args ─────────────────────────┼──> scene_from_args ──> SceneSpec (résolu) ──> prepare(...)
models/g1/g1_29dof.urdf ──────────┘
```

## Gestion d'erreurs

- `paths.toml` absent → `FileNotFoundError` (« copier `paths.example.toml` ») **seulement** quand un
  défaut en dépend (chargement tolérant : `--model-dir` absent ou motion relative).
- `[models].smplx` absent quand on s'appuie sur le défaut → `ValueError` nommant la clé et le fichier.
- `[datasets.name].motion` absent : n'erre **que** si une `--motion-path` *relative* a besoin de la
  racine (motion absolue → OK). `dataset_meta_root` renvoie `None` si le dataset est absent
  (→ `SceneSpec.dataset_root = None`, ex. SFU sans objets).
- `smplh`/`smpl2smplx` absents → `None` (pas d'erreur ; le loader HOI-M3 retombe sur la convention,
  qui lève un `FileNotFoundError` clair seulement à l'exécution si le fichier manque réellement).
- Existence des chemins **non** vérifiée par le résolveur (responsabilité des loaders).

## Tests (sur un `paths.toml` temporaire en `tmp_path`)

- `tests/test_paths.py` : `smplx_dir` (requis/`ValueError`) ; `smplh_dir`/`smpl2smplx_pkl`
  (optionnels → `None`) ; `dataset_motion_root` (requis/`ValueError`) ; `dataset_meta_root`
  (`meta` sinon `motion` sinon `None`) ; `resolve_motion` (absolu/relatif) ; `load_paths`
  (`FileNotFoundError` + round-trip).
- `tests/test_scene_args.py` : défauts depuis `paths.toml` (smplx, motion via `[datasets].motion`,
  `dataset_root` via `meta`/défaut) ; `smplh`/`smpl2smplx` peuplés ; overrides explicites ;
  invocations absolues **sans** `paths.toml` (y compris `--dataset-root` omis) ; URDF réel.
- `tests/test_smpl2smplx_resolve.py` : `_resolve_assets` — convention (`parents[2]`) **vs** overrides.

## CLI résultant

```bash
cp paths.example.toml paths.toml          # une fois, éditer ses chemins
$PY -m src.viz.viewer --dataset hodome --motion-path smplx/subject01_baseball.npz
$PY -m src.viz.viewer --dataset omomo  --motion-path sub10_clothesstand_000.pt   # relatif (motion=OMOMO_new)
# (les formes absolues, avec --model-dir/--dataset-root explicites, restent valides)
```

`CHEATSHEET.md` mis à jour en conséquence (section setup + exemples raccourcis).

## Hors périmètre / suites possibles

- Scènes nommées (presets) si le besoin réapparaît.
- Réutilisation de `_scene_args`/`paths.py` par un futur point d'entrée `run` (solve).
```
</content>
