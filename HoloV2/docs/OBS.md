# HoloV2 — Observabilité (`holov2/obs.py`)

Timing + logs **à chaque étape**, sans spaghetti. Primitive : `Profile` (spans imbriqués +
events), **opt-in** et **no-op quand désactivé** (le hot path ne paie rien).

## Principe : instrumenter aux SEAMS, jamais dans les ops pures

Le calcul est des fonctions pures composées dans des orchestrateurs (`prepare/runner`,
`targets/pipeline`). On met les spans **dans les orchestrateurs**, autour des appels — jamais
dans `pose_cloud`/`eval_fields`/`transport`/les builders. C'est l'analogue timing de `viz`
qui lit `FrameTrace` : l'observabilité est un **wrapper**, pas un hook embarqué.

**Granularité = composition** : nos ops sont petites et explicites → un span par op vient
gratuitement. Besoin de plus fin ? on **scinde l'op** (pas de hook dedans).

## API
```python
class Profile:
    def span(self, name, **meta): ...   # context manager : time + imbrique
    def event(self, name, **meta): ...  # point de log (ex: cache hit/miss)
    def tree(self) -> Span              # arbre des spans + durées
    def render(self) -> str             # flame-tree indenté (ms + % + meta)
NULL = Profile(enabled=False)           # défaut partout -> overhead ~nul
```
- `enabled=False` ⇒ `span`/`event` retournent immédiatement (aucune mutation).
- `logger` optionnel ⇒ une ligne debug structurée par span en sortie.

## Câblage : `prof=NULL` optionnel sur les orchestrateurs
```python
# prepare/runner.py
def prepare(scene_spec, config, prof=NULL):
    with prof.span("prepare"):
        with prof.span("calibration"):    calib = calibration.build_or_load(...); prof.event("cache hit", item="calib")
        with prof.span("sdf", n=N):       ...
        with prof.span("correspondence"): ...

# targets/pipeline.py
def process_frame(grounded, ctx, config, f, prof=NULL):
    with prof.span("frame", f=f):
        with prof.span("pose"):                       ...
        with prof.span("interaction.eval", n_channels=C, n_points=P): ...
        with prof.span("interaction.transport"):      ...
```
(Stubs déjà en place avec ces signatures + le plan de spans.)

## Ce que ça donne
- **prepare** : durée + **cache hit/miss** par livrable (calibration/sdf/point_cloud/correspondance).
- **targets** : durée par op et par frame ; fps soutenu.
- **arbre imbriqué** (render) : où passe le temps, en % — flame-graph textuel.

## Exemple de sortie
```
frame      6.99ms {'f': 12}
  pose      2.23ms   32%
  eval      4.69ms   67% {'n_points': 3925, 'n_channels': 3}
  · sdf cache hit {'item': 'obj0'}
```

## Règle (cohérente avec viz / contrats)
- spans UNIQUEMENT dans les orchestrateurs ; `obs` ne dépend de rien ; le calcul ne dépend
  pas de `obs`.
- `Profile` est l'artefact d'observabilité, comme `FrameTrace` est l'artefact de visu.
- Pour un inner-loop ultra-chaud, garder `if prof.enabled` autour des micro-spans (sinon le
  coût d'un span ~µs reste négligeable face à des étapes en ms).
