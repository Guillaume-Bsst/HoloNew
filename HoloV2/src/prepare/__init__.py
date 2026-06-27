"""``prepare`` stage — offline, q-independent: load + ground + build the geometry assets.

Public surface (what downstream stages import): ``prepare.contracts`` (the data types it produces),
``prepare.config`` (the knobs), and ``prepare.runner.prepare`` (the entry point). Downstream imports
those, never ``prepare``'s internal submodules (``load/`` ``calibration/`` ``sdf/`` ``point_cloud/``).
``prepare`` is the only place that instantiates SMPL / meshes / robot.
"""
