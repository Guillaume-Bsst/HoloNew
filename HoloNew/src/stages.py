"""Single source of truth for the viewer's retargeting stages.

Adding a future stage (e.g. GMR-SOCP) is one StageSpec entry here plus a
producer that fills its data. Dropdown, ghost overlay and robot-mesh gating all
derive from this registry.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class StageSpec:
    label: str           # dropdown label
    key: str | None      # key into RetargetResult.stages; None = raw human skeleton
    produces_qpos: bool  # True => drives a robot mesh under /world/robot_<key>


STAGE_SPECS: tuple[StageSpec, ...] = (
    StageSpec("Original",    None,          False),
    StageSpec("Mapped",      "mapped",      False),
    StageSpec("InObject",    "in_object",   False),
    StageSpec("SOCP",        "socp",        True),
    StageSpec("GMR-SOCP v1", "gmr_socp_v1", True),
    StageSpec("GMR-SOCP v2", "gmr_socp_v2", True),
)

_BY_LABEL: dict[str, StageSpec] = {s.label: s for s in STAGE_SPECS}


def stage_labels() -> list[str]:
    return [s.label for s in STAGE_SPECS]


def spec_for_label(label: str) -> StageSpec:
    return _BY_LABEL[label]


def key_for_label(label: str) -> str | None:
    return _BY_LABEL[label].key


def produces_qpos(label: str) -> bool:
    return _BY_LABEL[label].produces_qpos
