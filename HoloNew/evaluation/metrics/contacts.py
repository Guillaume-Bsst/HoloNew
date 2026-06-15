"""Contact metrics: timing (F1) + placement + no-slip.

Pure aggregator over per-frame / per-contact-point arrays. Contact detection,
surface distance, and slip are extracted at the call site (it owns the geometry);
this just scores the arrays, so floor (analytic) and object (mesh) channels share
one definition.
"""
from __future__ import annotations

import numpy as np


def _ratio(num: float, den: float) -> float:
    return float(num / den) if den > 0 else 0.0


def compute_contacts(robot_contact: np.ndarray, ref_contact: np.ndarray,
                    placement_dist: np.ndarray, slip: np.ndarray) -> dict[str, float]:
    """Timing / placement / no-slip contact quality.

    All inputs are (T, C): ``robot_contact`` / ``ref_contact`` bool, ``placement_dist``
    (m, robot point to its target surface), ``slip`` (m, per-frame tangential drift).
    ``ref_contact`` is the ground truth (the source says contact here).
    """
    rb = np.asarray(robot_contact, dtype=bool)
    rf = np.asarray(ref_contact, dtype=bool)

    tp = int(np.sum(rb & rf))
    fp = int(np.sum(rb & ~rf))
    fn = int(np.sum(~rb & rf))
    precision = _ratio(tp, tp + fp)
    recall = _ratio(tp, tp + fn)
    f1 = _ratio(2 * precision * recall, precision + recall)

    both = rb & rf
    place_err = float(np.mean(placement_dist[both])) if np.any(both) else 0.0

    sustained = np.zeros_like(rb)
    sustained[1:] = rb[1:] & rb[:-1]
    slip_mean = float(np.mean(slip[sustained])) if np.any(sustained) else 0.0

    return {
        "contact_precision": precision,
        "contact_recall": recall,
        "contact_f1": f1,
        "contact_place_err": place_err,
        "contact_slip_mean": slip_mean,
    }
