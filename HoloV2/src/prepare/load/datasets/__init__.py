"""Per-dataset motion loaders (one module per dataset). Each registers itself via
``@register_loader("name")`` on import; the registry in ``..base`` imports them lazily by name.
Shared infrastructure (BodyModel, SMPL->SMPL-X transfer, object mesh) stays in the parent ``load``.
"""
