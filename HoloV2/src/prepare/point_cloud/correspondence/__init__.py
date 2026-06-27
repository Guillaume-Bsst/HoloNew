"""Human<->robot surface correspondence, built offline by per-segment optimal transport (and read
back from the cache). Logic in the submodules; ``robot_surface``/``segments``/``ot_couple`` are the
generic build steps, ``build`` the orchestrator + ``AssetBuilder``, ``cache`` the (save+load) format."""
from .build import CorrespondenceBuilder, build_correspondence, regenerate
from .cache import load_correspondence, save_correspondence

__all__ = ["load_correspondence", "save_correspondence", "build_correspondence",
           "CorrespondenceBuilder", "regenerate"]
