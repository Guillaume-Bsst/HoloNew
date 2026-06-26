"""Make the ``holov2`` package importable when running tests from the HoloV2 root."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
