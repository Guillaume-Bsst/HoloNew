"""Put the HoloV2 root on ``sys.path`` so the ``src`` package is importable when running tests
from here (types/knobs live per stage, e.g. ``src.prepare.contracts`` / ``src.prepare.config``)."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
