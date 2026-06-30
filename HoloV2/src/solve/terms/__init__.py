"""``solve/terms`` — residual builders (``style``/``contact``/``object``/``reg``/``constraints``) +
their complex residual-only ops (``_ops``). INTERNAL to ``solve`` (used by ``solve/assemble`` in
Plan C); deliberately NOT re-exported by ``solve/__init__`` so the public ``solve`` import stays
cvxpy/torch/pinocchio-free. Each builder folds the ``SolveConfig`` weights into the ``ResidualBlock``
``A``/``c`` (no separate weight) and returns Plan A contract objects."""
