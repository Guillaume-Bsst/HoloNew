"""prepare/geodesic — table all-pairs de distances géodésiques par mesh (objets/terrain), build + cache.

Thin re-export de la surface publique ; logique de build dans ``build.py``, I/O .npz dans ``cache.py``."""
from .cache import load_geo, save_geo

__all__ = ["save_geo", "load_geo"]
