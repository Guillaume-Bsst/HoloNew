"""prepare/sdf — grilles distance-signée + witness pour objets/terrain/sol (build + cache).

Mince ré-export de la surface publique ; logique build dans ``build.py``, I/O .npz dans ``cache.py``."""
from .build import SdfBuilder, build_plane_sdf, build_sdf
from .cache import load_sdf, save_sdf

__all__ = ["SdfBuilder", "build_plane_sdf", "build_sdf", "save_sdf", "load_sdf"]
