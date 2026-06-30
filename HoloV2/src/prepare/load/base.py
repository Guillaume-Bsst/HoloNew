"""Chargeurs de données : le protocol ``MotionLoader`` + un registre name->loader.

Chaque ensemble de données a UN chargeur qui transforme un ``SceneSpec`` en contrat uniforme
``RawMotion`` ; tout en aval est agnostique du dataset. Ajouter un dataset = ajouter un module
avec une classe ``@register_loader("name")`` — rien d'autre ne change. Les chargeurs sont
sans état (tous les inputs proviennent de la spec).
"""
from __future__ import annotations

import importlib
from typing import Protocol, runtime_checkable

from ..contracts import RawMotion, SceneSpec


@runtime_checkable
class MotionLoader(Protocol):
    """Transforme un ``SceneSpec`` en ``RawMotion`` (joints, params SMPL optionnels, poses
    d'objets + chemins mesh). Le seul code spécifique à un dataset dans le pipeline."""

    def load(self, spec: SceneSpec) -> RawMotion: ...


_LOADERS: dict[str, type[MotionLoader]] = {}


def register_loader(name: str):
    """Décorateur de classe enregistrant un ``MotionLoader`` sous ``name`` (lève une exception en cas de doublon)."""

    def _register(cls: type[MotionLoader]) -> type[MotionLoader]:
        if name in _LOADERS:
            raise ValueError(f"duplicate loader for dataset {name!r}")
        _LOADERS[name] = cls
        return cls

    return _register


def get_loader(dataset: str) -> MotionLoader:
    """Instancier le chargeur enregistré pour ``dataset`` (lève ValueError si inconnu).

    Les chargeurs s'enregistrent à l'import. Par convention, le module pour ``dataset`` est
    ``src.prepare.load.datasets.<dataset>``, il est importé paresseusement au premier usage —
    gardant ce package léger (torch/smplx ne sont pas importés tant qu'un dataset paramétré
    n'est pas vraiment demandé)."""
    if dataset not in _LOADERS:
        try:
            importlib.import_module(f"{__package__}.datasets.{dataset}")
        except ModuleNotFoundError as e:
            if e.name != f"{__package__}.datasets.{dataset}":
                raise                  # une vraie dépendance manque dans le loader — la faire remonter
    try:
        cls = _LOADERS[dataset]
    except KeyError:
        known = ", ".join(sorted(_LOADERS)) or "(none registered)"
        raise ValueError(f"unknown dataset {dataset!r}; known: {known}") from None
    return cls()


def load(spec: SceneSpec) -> RawMotion:
    """Entrée de haut niveau : dispatche vers le chargeur enregistré pour ``spec.dataset``."""
    return get_loader(spec.dataset).load(spec)
