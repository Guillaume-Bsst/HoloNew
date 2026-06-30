"""Étape ``prepare`` — offline, q-indépendante : load + ancrage + construction des assets géométrie.

Surface publique (ce que les étapes aval importent) : ``prepare.contracts`` (les types données qu'elle
produit), ``prepare.config`` (les knobs), et ``prepare.runner.prepare`` (le point d'entrée). Les étapes
aval importent ceux-ci, jamais les sous-modules internes de ``prepare`` (``load/`` ``calibration/``
``sdf/`` ``point_cloud/``). ``prepare`` est l'unique endroit qui instancie SMPL / meshes / robot.
"""
