"""Human<->robot surface correspondence, built offline by per-segment optimal transport (and read
back from the cache). Logic in the submodules; ``robot_surface``/``segments``/``ot_couple`` are the
generic build steps, ``build`` the orchestrator + ``AssetBuilder``, ``load`` the reader."""
from .build import CorrespondenceBuilder, build_correspondence, regenerate
from .load import load_correspondence

__all__ = ["load_correspondence", "build_correspondence", "CorrespondenceBuilder", "regenerate"]
