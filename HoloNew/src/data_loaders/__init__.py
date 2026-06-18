"""Dataset-specific motion loaders for the robot_retarget CLI façade."""
from HoloNew.src.data_loaders.base import (  # noqa: F401
    MotionLoader, DATASET_TO_FORMAT, DATASET_LOADERS, register_loader, resolve_loader,
)

# Import the concrete loader modules so their @register_loader side-effects run
# when the package is imported.
from HoloNew.src.data_loaders import omomo, hoim3, legacy  # noqa: E402,F401
