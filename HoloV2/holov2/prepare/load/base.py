"""Dataset loaders: the ``MotionLoader`` protocol + a name->loader registry.

Each dataset has ONE loader turning a ``SceneSpec`` into the uniform ``RawMotion`` contract;
everything downstream is dataset-agnostic. Adding a dataset = adding a module with a
``@register_loader("name")`` class — nothing else changes. Loaders are stateless (all inputs
come from the spec).
"""
from __future__ import annotations

from typing import Protocol, runtime_checkable

from ...contracts import RawMotion, SceneSpec


@runtime_checkable
class MotionLoader(Protocol):
    """Turns a ``SceneSpec`` into a ``RawMotion`` (joints, optional SMPL params, object poses +
    mesh paths). The only dataset-specific code in the pipeline."""

    def load(self, spec: SceneSpec) -> RawMotion: ...


_LOADERS: dict[str, type[MotionLoader]] = {}


def register_loader(name: str):
    """Class decorator registering a ``MotionLoader`` under ``name`` (raises on duplicate)."""

    def _register(cls: type[MotionLoader]) -> type[MotionLoader]:
        if name in _LOADERS:
            raise ValueError(f"duplicate loader for dataset {name!r}")
        _LOADERS[name] = cls
        return cls

    return _register


def get_loader(dataset: str) -> MotionLoader:
    """Instantiate the loader registered for ``dataset`` (raises ValueError if unknown)."""
    try:
        cls = _LOADERS[dataset]
    except KeyError:
        known = ", ".join(sorted(_LOADERS)) or "(none registered)"
        raise ValueError(f"unknown dataset {dataset!r}; known: {known}") from None
    return cls()


def load(spec: SceneSpec) -> RawMotion:
    """Top-level entry: dispatch to the loader registered for ``spec.dataset``."""
    return get_loader(spec.dataset).load(spec)
