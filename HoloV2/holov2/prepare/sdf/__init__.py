"""prepare/sdf — signed-distance + witness grids for objects/terrain/ground (build + cache).

Thin re-export of the public surface; logic lives in ``build.py``."""
from .build import SdfBuilder, build_plane_sdf, build_sdf

__all__ = ["SdfBuilder", "build_plane_sdf", "build_sdf"]
