"""``solve/terms`` — builders résiduels (``style``/``contact``/``object``/``reg``/``constraints``) +
leurs ops résiduels-only complexes (``_ops``). INTERNE à ``solve`` (utilisé par ``solve/assemble`` en
Plan C) ; délibérément NON ré-exporté par ``solve/__init__`` pour que l'import public ``solve`` reste
cvxpy/torch/pinocchio-free. Chaque builder replie les poids ``SolveConfig`` dans ``A``/``c`` du ``ResidualBlock``
(pas de poids séparé) et retourne des objets de contrats Plan A."""
