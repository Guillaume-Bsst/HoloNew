"""Chargeurs de motion par dataset (un module par dataset). Chacun s'enregistre via
``@register_loader("name")`` à l'import ; le registre dans ``..base`` les importe paresseusement par nom.
L'infrastructure partagée (BodyModel, transfert SMPL->SMPL-X, mesh d'objet) reste dans le parent ``load``.
"""
