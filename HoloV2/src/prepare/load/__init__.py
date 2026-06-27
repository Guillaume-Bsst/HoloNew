"""Dataset loading: SceneSpec -> RawMotion. Import a concrete loader module to register it."""
from .base import MotionLoader, get_loader, load, register_loader

__all__ = ["MotionLoader", "register_loader", "get_loader", "load"]
