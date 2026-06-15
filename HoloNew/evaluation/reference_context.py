"""Reference context for style/contact/root metrics that need GMR-grounded targets.

The style metric scores a motion against per-robot-link reference orientations and
positions, which live in ``rt.gmr_ground`` (the source motion grounded onto the robot
skeleton by GMR) — not in the result npz. This wraps a retargeter ``rt`` and exposes
the reference arrays plus forward kinematics over an arbitrary qpos trajectory, so any
method's output (ours or an external one) can be scored against the same source.
"""
from __future__ import annotations

import numpy as np


class ReferenceContext:
    """Per-clip reference (GMR-grounded targets) + FK, backing the style metric."""

    def __init__(self, rt):
        from HoloNew.src.test_socp.tables import IK_MATCH_TABLE1, ROBOT_ROOT_NAME
        from HoloNew.src.test_socp.targets import ground_frame_targets

        self.rt = rt
        self._gft = ground_frame_targets
        self._table = IK_MATCH_TABLE1
        self._root_name = ROBOT_ROOT_NAME
        self._gpos = rt.gmr_ground["pos"]      # (Tf, B, 3)
        self._gquat = rt.gmr_ground["quat"]    # (Tf, B, 4) wxyz

        # Fixed body order + tracked mask + pelvis index, from a frame-0 target build.
        tg0 = ground_frame_targets(self._gpos[0], self._gquat[0], self._table)
        self.frames = [f for f in self._table if f in tg0]
        self.body_order = [rt.robot_link_names[f] for f in self.frames]
        self.pelvis_idx = next(
            i for i, f in enumerate(self.frames)
            if rt.robot_link_names[f] == self._root_name)
        # Track non-pelvis links that carry an orientation weight (rot_w > 0), matching
        # the existing pelvis-relative fidelity metric.
        self.tracked = np.array([
            (rt.robot_link_names[f] != self._root_name) and (tg0[f][3] > 0)
            for f in self.frames], dtype=bool)

    @classmethod
    def from_rt(cls, rt) -> "ReferenceContext":
        return cls(rt)

    @classmethod
    def from_config(cls, task_type: str, task_name: str,
                    data_format: str) -> "ReferenceContext":
        from HoloNew.examples.robot_retarget import RetargetingConfig
        from HoloNew.src.test_socp.test_socp import TestSocpRetargeter
        rt = TestSocpRetargeter.from_config(RetargetingConfig(
            task_type=task_type, task_name=task_name, data_format=data_format))
        return cls(rt)

    def reference_RP(self, T: int):
        """Reference per-link rotations (T, K, 3, 3) and positions (T, K, 3)."""
        K = len(self.frames)
        Rref = np.empty((T, K, 3, 3))
        pref = np.empty((T, K, 3))
        for t in range(T):
            tg = self._gft(self._gpos[t], self._gquat[t], self._table)
            for i, f in enumerate(self.frames):
                p_t, R_t, _, _ = tg[f]
                pref[t, i] = p_t
                Rref[t, i] = R_t
        return Rref, pref

    def fk_links(self, qpos: np.ndarray):
        """FK a qpos trajectory to per-link world rotations / positions in body order."""
        rt = self.rt
        T, K = qpos.shape[0], len(self.frames)
        rot = np.empty((T, K, 3, 3))
        pos = np.empty((T, K, 3))
        for t in range(T):
            q = rt.q_init_full.copy()
            q[:qpos.shape[1]] = qpos[t]
            for i, name in enumerate(self.body_order):
                rot[t, i] = rt.body_rotation(q, name)
                pos[t, i] = rt.body_position(q, name)
        return rot, pos

    def score_style(self, method_qpos: np.ndarray,
                    gmr_baseline_qpos: np.ndarray | None = None) -> dict[str, float]:
        """Style of ``method_qpos`` vs the source (gmr_ground), and vs a GMR baseline.

        Returns ``style_{orient,shape}_vs_smpl`` always, plus ``*_vs_gmr`` when a
        GMR-baseline trajectory is given.
        """
        from HoloNew.evaluation.metrics.style import compute_style

        T = min(method_qpos.shape[0], self._gpos.shape[0])
        rot_m, pos_m = self.fk_links(method_qpos[:T])
        rot_ref, pos_ref = self.reference_RP(T)
        s = compute_style(rot_m, pos_m, rot_ref, pos_ref, self.pelvis_idx, self.tracked)
        out = {
            "style_orient_vs_smpl": s["style_orient_err"],
            "style_shape_vs_smpl": s["style_shape_err"],
        }
        if gmr_baseline_qpos is not None:
            Tg = min(T, gmr_baseline_qpos.shape[0])
            rot_g, pos_g = self.fk_links(gmr_baseline_qpos[:Tg])
            sg = compute_style(rot_m[:Tg], pos_m[:Tg], rot_g, pos_g,
                              self.pelvis_idx, self.tracked)
            out["style_orient_vs_gmr"] = sg["style_orient_err"]
            out["style_shape_vs_gmr"] = sg["style_shape_err"]
        return out
