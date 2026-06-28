"""prepare/sdf — signed-distance + witness grids for objects/terrain/ground (build + cache).

Thin re-export of the public surface; build logic in ``build.py``, .npz I/O in ``cache.py``."""
from .build import SdfBuilder, build_plane_sdf, build_sdf
from .cache import load_sdf, save_sdf

__all__ = ["SdfBuilder", "build_plane_sdf", "build_sdf", "save_sdf", "load_sdf"]
