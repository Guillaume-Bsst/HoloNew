"""Chargement de données : SceneSpec -> RawMotion. Importer un module loader concret pour l'enregistrer."""
from .base import MotionLoader, get_loader, load, register_loader

__all__ = ["MotionLoader", "register_loader", "get_loader", "load"]
