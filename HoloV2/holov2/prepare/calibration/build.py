"""prepare/calibration — grounds the whole scene (human + objects) and characterises the subject.

Produces the per-(subject, take) ``Calibration`` (``contracts.Calibration``): three scene-level
parameters, computed offline once and cached. ROBOT-FREE by design — the human->robot scale is a
(human, robot) quantity, owned and applied by the correspondence + transport layer (the module
whose job IS the human<->robot pairing, where both surfaces meet); calibration only supplies the
robot-free subject ``human_stature`` it composes with the robot height.

  - ``human_stature`` = the subject's rest-pose stature (m), from betas-FK (vertical extent of the
                        rest mesh). The REAL subject's size — feeds ``scale = robot_height / stature``
                        downstream. Uniform across datasets: every parametric source has a SMPL-X
                        ``BodyModel``, so there is no per-dataset stature path.
  - ``human_offset``  = world z-shift resting the human SOLE on z=0 (median over sampled frames of
                        the lowest posed vertex, robust to crouch / airborne outliers).
  - ``object_offsets``= world z-shift PER object, grounding each one independently of the human
                        (single-human / multi-object): an object may already rest on the floor while
                        the human floats, so a shared scene shift would push it through the floor.
                        PROVISIONAL: 0 = keep the captured pose (correct where objects are already
                        grounded, e.g. OMOMO); the per-object computation is still being designed.
  - ``root_frame``    = a (4,4) framing of the root. Identity for now (provisional hook; the V1
                        kept XY raw). Generalise only when a non-trivial framing is actually needed.

Decisions (confirmed): the human->robot scale is NOT computed here (calibration stays robot-free,
cacheable per subject); the scene stays at HUMAN scale, where the interaction pipeline lives (human
cloud + object are size-consistent, and the SMPL->robot transport absorbs the size difference).
Grounding is ALWAYS recomputed (uniform): an already-grounded clip (SFU) simply yields
``floor_offset ~= 0``.

Dataset-agnostic by construction: every difference between datasets lives in the DATA the loader
already produced (``smpl_params`` present -> surface refinement, else toe-joint fallback;
``betas`` -> stature; ``joint_names`` -> toe indices; empty object list -> nothing to ground),
never in a branch here.

Ported from HoloNew: ``holosoma/preprocess.ground_to_floor``,
``contact/smplx_field.robust_floor_offset``, ``data_loaders/omomo.omomo_height_from_betas``.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

from ...contracts import BodyModel, Calibration, CalibrationConfig, RawMotion, SmplParams

# Fallback stature (m) for a non-parametric source (no betas to FK); matches the previous HoloNew
# default human height.
DEFAULT_HUMAN_HEIGHT = 1.78

# How many frames to sample for the robust floor offset. The lowest posed VERTEX per frame needs a
# full SMPL-X forward, so we subsample the clip; 40 is plenty for a robust median (V1 used the same).
_FLOOR_N_SAMPLES = 40


# =============================================================================
# Pure functions (no I/O, no mutation) — the calibration math
# =============================================================================
def human_stature(body: BodyModel, params: SmplParams) -> float:
    """Rest-pose stature (m) = vertical extent of the subject's rest mesh, from betas-FK.

    ``rest_vertices`` is in the model's NATIVE Y-up frame, so the stature is the Y-extent (the same
    quantity the previous HoloNew measured for both the SMPL-X and the OMOMO betas paths)."""
    rv = np.asarray(body.rest_vertices(params), np.float64)   # (V, 3) native Y-up
    return float(rv[:, 1].max() - rv[:, 1].min())


def toe_ground_offset(joint_pos: np.ndarray, toe_idx: list[int], mat_height: float) -> float:
    """Coarse z-drop from the demo TOE joints (the non-parametric fallback — no body mesh to pose).

    Lowest toe z over the whole clip rests on z=0; if it already sits a full ``mat_height`` above
    the floor the subject is standing on a mat, so we keep them on the mat top instead of shoving
    them ``mat_height`` into a phantom sub-mat floor. Returns the value to SUBTRACT from world z.
    Mirrors ``holosoma/preprocess.ground_to_floor``."""
    z_min = float(np.asarray(joint_pos)[:, toe_idx, 2].min())
    if z_min >= mat_height:
        z_min -= mat_height
    return z_min


def sole_floor_offset(body: BodyModel, params: SmplParams, mat_height: float,
                      contact_margin: float = 0.0, n_samples: int = _FLOOR_N_SAMPLES) -> float:
    """Robust z-drop resting the human SOLE on z=0, from the posed SMPL surface (deterministic FK).

    Each sampled frame contributes the lowest posed VERTEX z (the skin sole, a few cm below the toe
    JOINT); the median over frames centres the nearly-stationary planted feet while staying robust
    to a deep-crouch / airborne frame. ``contact_margin`` biases the result so the foot ends a hair
    BELOW z=0 (a small penetration reads as solid contact, preferable to floating). The ``mat_height``
    rule matches ``toe_ground_offset``. Returns the value to SUBTRACT from world z.
    Mirrors ``contact/smplx_field.robust_floor_offset`` (applied to the raw posed surface)."""
    T = params.n_frames
    idx = np.unique(np.linspace(0, T - 1, min(T, n_samples)).astype(int))
    mins = np.array([float(np.asarray(body.posed_vertices(params, t))[:, 2].min()) for t in idx])
    sole = float(np.median(mins))
    if sole >= mat_height:
        sole -= mat_height
    return sole + float(contact_margin)


def _toe_indices(joint_names: tuple[str, ...]) -> list[int]:
    """Indices of the foot joints (case-insensitive 'foot', else 'ankle'). Raises if neither."""
    feet = [i for i, n in enumerate(joint_names) if "foot" in n.lower()]
    if not feet:
        feet = [i for i, n in enumerate(joint_names) if "ankle" in n.lower()]
    if not feet:
        raise ValueError(f"no foot/ankle joint to ground on in {joint_names!r}")
    return feet


# =============================================================================
# CalibrationBuilder — the AssetBuilder for this deliverable (build / cache)
# =============================================================================
class CalibrationBuilder:
    """``AssetBuilder`` producing the ``Calibration`` for a (subject, take). Scoped to that pair
    (NOT a geometry cache): the floor offset depends on the whole motion, the stature on the
    subject. ROBOT-FREE (no robot input -> caches per subject). The runner wraps ``build``/``load``
    in a ``prof.span("calibration")``."""

    def cache_key(self, config: CalibrationConfig, raw: RawMotion) -> str:
        """Stable hash of everything the calibration depends on: the calibration config and the
        subject + motion (the params that drive the stature and the floor offset; the demo joints
        for the non-parametric fallback). No robot term — calibration is robot-free."""
        h = hashlib.sha1()
        h.update(f"{config.mat_height}".encode())
        p = raw.smpl_params
        if p is not None:
            h.update(str(p.gender).encode())
            for arr in (p.betas, p.global_orient, p.body_pose, p.left_hand_pose,
                        p.right_hand_pose, p.transl):
                h.update(np.ascontiguousarray(arr, np.float32).tobytes())
        else:
            h.update("|".join(raw.joint_names).encode())
            h.update(np.ascontiguousarray(raw.joint_pos, np.float32).tobytes())
        return h.hexdigest()

    def build(self, config: CalibrationConfig, raw: RawMotion,
              body: BodyModel | None = None, smpl_model_dir: Path | None = None) -> Calibration:
        """Compute the ``Calibration``. For a parametric source, reuse ``body`` if supplied, else
        build it from ``raw.smpl_params`` + ``smpl_model_dir`` (one of the two is required — the
        model dir lives on ``SceneSpec``, not ``RawMotion``). A positions-only clip needs neither:
        it falls back to toe-joint grounding and the default stature."""
        if raw.is_parametric and body is None:
            if smpl_model_dir is None:
                raise ValueError("a parametric scene needs a BodyModel or smpl_model_dir to "
                                 "compute the stature and the surface floor offset")
            from ..load.smpl import build_body_model
            body = build_body_model(raw.smpl_params, smpl_model_dir)

        # Stature: betas-FK from the real subject when parametric; the default otherwise.
        stature = human_stature(body, raw.smpl_params) if body is not None else DEFAULT_HUMAN_HEIGHT

        # Human floor offset: robust surface sole when parametric, else the coarse toe-joint drop.
        if body is not None:
            human = sole_floor_offset(body, raw.smpl_params, config.mat_height)
        else:
            human = toe_ground_offset(raw.joint_pos, _toe_indices(raw.joint_names), config.mat_height)

        # One floor offset PER object, grounded independently of the human. PROVISIONAL: 0 keeps the
        # object's captured pose (correct where it already rests on the floor, e.g. OMOMO); the real
        # per-object computation is still being designed (see Calibration docstring).
        object_offsets = (0.0,) * len(raw.object_poses_raw)

        return Calibration(human_stature=stature, human_offset=human,
                           object_offsets=object_offsets, root_frame=np.eye(4))

    def save(self, calib: Calibration, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(str(path), human_stature=np.float64(calib.human_stature),
                 human_offset=np.float64(calib.human_offset),
                 object_offsets=np.asarray(calib.object_offsets, np.float64),
                 root_frame=np.asarray(calib.root_frame, np.float64))

    def load(self, path: Path) -> Calibration:
        d = np.load(str(path))
        return Calibration(human_stature=float(d["human_stature"]), human_offset=float(d["human_offset"]),
                           object_offsets=tuple(np.asarray(d["object_offsets"], np.float64).tolist()),
                           root_frame=np.asarray(d["root_frame"], np.float64))


def build_calibration(raw: RawMotion, config: CalibrationConfig, body: BodyModel | None = None,
                      smpl_model_dir: Path | None = None) -> Calibration:
    """Convenience wrapper around ``CalibrationBuilder.build`` (the common call site)."""
    return CalibrationBuilder().build(config, raw, body=body, smpl_model_dir=smpl_model_dir)
