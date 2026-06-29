# Spec — Résolveur de chemins V2 (`paths.toml`)

**Date** : 2026-06-29 · **Étage** : bord/CLI (hors pipeline pur) · **Statut** : design approuvé

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

- Pas de « scènes nommées » / presets (`--scene baseball`). Racines seulement.
- Pas de chemins dans `config.py`/`PrepareConfig` (ce ne sont pas des knobs algorithmiques).
- Pas de dépendance sur `../HoloNew/path.yaml` (V2 reste découplé du legacy).
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

```toml
# Racines machine-spécifiques. Éditer pour sa propre machine.
smplx = "/home/.../models_smplx_v1_1/models/smplx"   # dossier contenant SMPLX_NEUTRAL.npz

[roots]
hodome = "/home/.../HODome"
omomo  = "/home/.../OMOMO"
sfu    = "/home/.../SFU/SFU"
```

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
| `load_paths(path: Path \| None = None) -> dict` | lit `paths.toml` via `tomllib`; `FileNotFoundError` clair pointant vers `paths.example.toml` si absent |
| `smplx_dir(cfg=None) -> Path` | racine SMPL-X; `ValueError` si clé `smplx` absente |
| `dataset_root(dataset: str, cfg=None) -> Path` | `roots[dataset]`; `ValueError` explicite si manquant |
| `resolve_motion(dataset: str, motion: str \| Path, cfg=None) -> Path` | absolu → tel quel; relatif → `dataset_root(dataset)/motion` |

### `src/viz/_scene_args.py` — colle argparse → `SceneSpec` (partagée par les mains viz)

- `add_scene_args(ap)` : déclare `--dataset`, `--motion-path`, `--model-dir` (optionnel),
  `--dataset-root` (optionnel), `--person-id`, `--object-names`, `--port`, `--frame-step`,
  `--max-frames`.
- `scene_from_args(a) -> SceneSpec` :
  - `--model-dir` ← `a.model_dir` sinon `paths.smplx_dir()`.
  - `--dataset-root` ← `a.dataset_root` sinon `paths.dataset_root(a.dataset)` (toléré absent
    pour les datasets sans objets type SFU).
  - `motion_path` ← `paths.resolve_motion(a.dataset, a.motion_path)`.
  - `RobotSpec` construit avec le **vrai** URDF repo-relatif `HOLOV2_ROOT/models/g1/g1_29dof.urdf`
    (remplace le placeholder `Path("g1.urdf")` des `main()` actuels — fix inclus dans ce lot).
- Adoptée par `viz/viewer.py`, `viz/scene.py`, `viz/cloud.py` (supprime la duplication).
  `viz/hoim3_multiperson.py` : alignement best-effort (signature différente, hors objets).

## Flux

```
paths.toml ──load_paths──> dict ──┐
CLI args ─────────────────────────┼──> scene_from_args ──> SceneSpec (résolu) ──> prepare(...)
models/g1/g1_29dof.urdf ──────────┘
```

## Gestion d'erreurs

- `paths.toml` absent → `FileNotFoundError` : « copier `paths.example.toml` vers `paths.toml` ».
- Clé `smplx` absente quand on s'appuie sur le défaut → `ValueError` nommant la clé et le fichier.
- Clé `roots[dataset]` absente : n'erre **que** si une `--motion-path` *relative* a besoin de la
  racine pour se résoudre (motion absolue → OK sans racine ; dataset sans objets → `SceneSpec.dataset_root` = None).
- Existence des chemins **non** vérifiée par le résolveur (responsabilité des loaders).

## Tests — `tests/test_paths.py`

Op pure ⇒ test unitaire (sur un `paths.toml` temporaire en `tmp_path`) :
- `resolve_motion` : chemin absolu rendu tel quel ; relatif joint à la racine du dataset.
- `dataset_root`/`smplx_dir` : valeur correcte ; `ValueError` si clé manquante.
- `load_paths` : `FileNotFoundError` si fichier absent ; round-trip d'un TOML écrit puis relu.

## CLI résultant

```bash
cp paths.example.toml paths.toml          # une fois, éditer ses chemins
$PY -m src.viz.viewer --dataset hodome --motion-path smplx/subject01_baseball.npz
# (les formes absolues actuelles restent valides)
```

`CHEATSHEET.md` mis à jour en conséquence (section setup + exemples raccourcis).

## Hors périmètre / suites possibles

- Scènes nommées (presets) si le besoin réapparaît.
- Réutilisation de `_scene_args`/`paths.py` par un futur point d'entrée `run` (solve).
```
</content>
