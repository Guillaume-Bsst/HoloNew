"""Put the HoloV2 root on ``sys.path`` so the top-level packages (``src``, ``config_types``,
``config_values``) are importable when running tests from here."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
