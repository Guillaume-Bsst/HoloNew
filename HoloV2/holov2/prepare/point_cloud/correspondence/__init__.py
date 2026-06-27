"""SMPL<->robot correspondence (built offline by per-segment OT; reused from the cache here).
Logic in ``load.py``."""
from .load import load_correspondence

__all__ = ["load_correspondence"]
