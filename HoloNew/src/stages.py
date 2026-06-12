"""Per-method registry for the annex stage viewer.

Each method declares its robot key and the ordered skeleton stages of its
pipeline; the implicit final "Robot" stage drives the solved robot mesh.
"""
from __future__ import annotations

from dataclasses import dataclass

ROBOT_STAGE = "Robot"


@dataclass(frozen=True)
class MethodSpec:
    label: str                     # dropdown label
    robot_key: str                 # robot instance key, /world/robot_<robot_key>
    skeleton_stages: tuple[str, ...]  # ordered preprocessing stages (skeletons)


METHODS: tuple[MethodSpec, ...] = (
    MethodSpec("holosoma", "holosoma", ("Original", "Grounded", "Scaled", "Mapped")),
    MethodSpec("GMR-SOCP v1", "gmr_socp_v1", ("Original", "Grounded", "Mapped", "Scaled", "Offset", "Floor")),
    MethodSpec("GMR-SOCP v2", "gmr_socp_v2", ("Original", "Grounded", "Mapped", "Scaled", "Offset", "Floor")),
)

_BY_LABEL = {m.label: m for m in METHODS}


def method_labels() -> list[str]:
    return [m.label for m in METHODS]


def method_for_label(label: str) -> MethodSpec:
    return _BY_LABEL[label]


def robot_key_for_method(label: str) -> str:
    return _BY_LABEL[label].robot_key


def stages_for_method(label: str) -> list[str]:
    """Ordered stage labels for a method: its skeleton stages + the Robot stage."""
    return list(_BY_LABEL[label].skeleton_stages) + [ROBOT_STAGE]
