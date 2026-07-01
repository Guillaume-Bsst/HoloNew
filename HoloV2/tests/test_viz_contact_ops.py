"""Tests unitaires pour les helpers purs ``src/viz/layers/_contact_ops.py``.

Chaque test utilise des entrées à valeur connue dont le résultat est vérifiable
à la main, conformément à la règle « op pure → test unitaire (petit cas
synthétique + valeur connue) » du CLAUDE.md.
"""
from __future__ import annotations

import numpy as np
import pytest

from src.viz.layers._contact_ops import object_cloud_solved, witness_segments


# ---------------------------------------------------------------------------
# object_cloud_solved
# ---------------------------------------------------------------------------

class TestObjectCloudSolved:
    """Vérifie la composition T_résolu ∘ T_source⁻¹ appliquée à un nuage."""

    def test_source_identity_translation_only(self):
        """Pose source = identité, pose résolue = translation pure → cloud décalé."""
        cloud = np.array([[1.0, 0.0, 0.0],
                          [0.0, 1.0, 0.0],
                          [0.0, 0.0, 1.0]], np.float64)
        R_src = np.eye(3)
        t_src = np.zeros(3)
        R_sol = np.eye(3)
        t_sol = np.array([10.0, 0.0, 0.0])

        result = object_cloud_solved(cloud, R_src, t_src, R_sol, t_sol)

        expected = cloud + t_sol
        np.testing.assert_allclose(result, expected, atol=1e-12)

    def test_source_identity_rotation_only(self):
        """Pose source = identité, pose résolue = rotation de 90° autour de Z."""
        cloud = np.array([[1.0, 0.0, 0.0]], np.float64)
        R_src = np.eye(3)
        t_src = np.zeros(3)
        # Rotation 90° autour de Z : x→y, y→-x
        R_sol = np.array([[0., -1., 0.],
                          [1.,  0., 0.],
                          [0.,  0., 1.]], np.float64)
        t_sol = np.zeros(3)

        result = object_cloud_solved(cloud, R_src, t_src, R_sol, t_sol)

        # [1,0,0] @ R_sol.T = R_sol @ [1,0,0] = [0,1,0]
        expected = np.array([[0., 1., 0.]])
        np.testing.assert_allclose(result, expected, atol=1e-12)

    def test_non_identity_source_un_pose_then_re_pose(self):
        """Pose source non identité : vérifie dé-pose puis re-pose.

        Cloud source posé à (R_src, t_src) avec un seul point.
        On sait que le point LOCAL de l'objet est p_local = R_src.T @ (p - t_src).
        Re-posé en résolu : p_résolu = R_sol @ p_local + t_sol.
        """
        # Un seul point à vérifier à la main
        # p en monde source = [2, 1, 0], t_src = [1, 0, 0] → p - t_src = [1, 1, 0]
        # R_src = rotation 90° autour de Z
        R_src = np.array([[0., -1., 0.],
                          [1.,  0., 0.],
                          [0.,  0., 1.]], np.float64)
        t_src = np.array([1.0, 0.0, 0.0])
        cloud = np.array([[2.0, 1.0, 0.0]])   # dans le monde source

        # p_local = R_src.T @ (p - t_src) = R_src.T @ [1,1,0]
        # R_src.T = [[0,1,0],[-1,0,0],[0,0,1]]  → R_src.T @ [1,1,0] = [1,-1,0]
        p_local_expected = np.array([1., -1., 0.])

        # Pose résolue : rotation identité + translation [0, 5, 0]
        R_sol = np.eye(3)
        t_sol = np.array([0.0, 5.0, 0.0])

        # p_résolu = R_sol @ p_local + t_sol = p_local + [0,5,0] = [1, 4, 0]
        expected = p_local_expected + t_sol

        result = object_cloud_solved(cloud, R_src, t_src, R_sol, t_sol)
        np.testing.assert_allclose(result, expected[np.newaxis], atol=1e-12)

    def test_source_eq_solved_is_identity(self):
        """Si la pose résolue égale la pose source, le cloud doit rester inchangé."""
        cloud = np.array([[3.0, 1.0, -2.0],
                          [0.5, 0.0,  4.0]], np.float64)
        R = np.array([[0., 0., 1.],
                      [0., 1., 0.],
                      [-1., 0., 0.]], np.float64)
        t = np.array([1.0, -2.0, 3.0])

        result = object_cloud_solved(cloud, R, t, R, t)
        np.testing.assert_allclose(result, cloud, atol=1e-12)

    def test_output_shape_multiple_points(self):
        """La forme de sortie est (P, 3) quel que soit P."""
        P = 50
        cloud = np.random.default_rng(42).standard_normal((P, 3))
        R = np.eye(3)
        t = np.zeros(3)
        result = object_cloud_solved(cloud, R, t, R, t)
        assert result.shape == (P, 3)


# ---------------------------------------------------------------------------
# witness_segments
# ---------------------------------------------------------------------------

class TestWitnessSegments:
    """Vérifie la construction des segments sonde→witness."""

    def _rot_z(self, angle: float) -> np.ndarray:
        """Rotation autour de Z d'angle ``angle`` (radians)."""
        c, s = np.cos(angle), np.sin(angle)
        return np.array([[c, -s, 0.], [s, c, 0.], [0., 0., 1.]])

    def test_known_segments_with_rotation(self):
        """Canal objet avec R_obj/t_obj connus → endpoints vérifiables à la main."""
        # 2 sondes, toutes deux actives
        probe_pts = np.array([[0.0, 0.0, 0.0],
                              [1.0, 0.0, 0.0]], np.float64)
        # Witness local [1,0,0] et [0,1,0]
        witness_local = np.array([[1.0, 0.0, 0.0],
                                  [0.0, 1.0, 0.0]], np.float64)
        active = np.array([True, True])

        # Rotation 90° autour de Z, translation [0,0,2]
        R_obj = self._rot_z(np.pi / 2)   # x→y, y→-x
        t_obj = np.array([0.0, 0.0, 2.0])

        segs = witness_segments(probe_pts, witness_local, active, R_obj, t_obj)

        assert segs.shape == (2, 2, 3)
        assert segs.dtype == np.float32

        # Sonde 0 : probe = [0,0,0]
        np.testing.assert_allclose(segs[0, 0], [0., 0., 0.], atol=1e-6)
        # wit_local[0] = [1,0,0] → monde : [1,0,0] @ R_obj.T + t_obj
        # R_obj.T @ [1,0,0] = col0 of R_obj = [cos90, sin90, 0] = [0, 1, 0]
        # + t_obj = [0, 1, 2]
        np.testing.assert_allclose(segs[0, 1], [0., 1., 2.], atol=1e-6)

        # Sonde 1 : probe = [1,0,0]
        np.testing.assert_allclose(segs[1, 0], [1., 0., 0.], atol=1e-6)
        # wit_local[1] = [0,1,0] → monde : [0,1,0] @ R_obj.T + t_obj
        # R_obj.T @ [0,1,0] = col1 of R_obj = [-sin90, cos90, 0] = [-1, 0, 0]
        # + t_obj = [-1, 0, 2]
        np.testing.assert_allclose(segs[1, 1], [-1., 0., 2.], atol=1e-6)

    def test_ground_channel_identity(self):
        """Canal sol (R_obj=I, t_obj=0) → witness déjà monde, inchangé."""
        probe_pts = np.array([[5.0, 3.0, 1.0]], np.float64)
        witness_local = np.array([[7.0, 2.0, -1.0]], np.float64)
        active = np.array([True])

        segs = witness_segments(
            probe_pts, witness_local, active,
            R_obj=np.eye(3), t_obj=np.zeros(3)
        )

        assert segs.shape == (1, 2, 3)
        np.testing.assert_allclose(segs[0, 0], [5., 3., 1.], atol=1e-6)
        np.testing.assert_allclose(segs[0, 1], [7., 2., -1.], atol=1e-6)

    def test_only_active_probes_included(self):
        """Seules les sondes actives apparaissent dans les segments."""
        probe_pts = np.array([[0., 0., 0.],
                              [1., 0., 0.],
                              [2., 0., 0.]])
        witness_local = np.array([[0.1, 0., 0.],
                                  [1.1, 0., 0.],
                                  [2.1, 0., 0.]])
        # Seule la sonde 1 est active
        active = np.array([False, True, False])

        segs = witness_segments(
            probe_pts, witness_local, active,
            R_obj=np.eye(3), t_obj=np.zeros(3)
        )

        assert segs.shape == (1, 2, 3)
        np.testing.assert_allclose(segs[0, 0], [1., 0., 0.], atol=1e-6)
        np.testing.assert_allclose(segs[0, 1], [1.1, 0., 0.], atol=1e-6)

    def test_empty_active_returns_0_2_3(self):
        """Aucune sonde active → forme (0, 2, 3) float32."""
        P = 10
        probe_pts = np.zeros((P, 3))
        witness_local = np.zeros((P, 3))
        active = np.zeros(P, bool)

        segs = witness_segments(
            probe_pts, witness_local, active,
            R_obj=np.eye(3), t_obj=np.zeros(3)
        )

        assert segs.shape == (0, 2, 3)
        assert segs.dtype == np.float32

    def test_subsample_cap_exact_count(self):
        """Plus de cap sondes actives → exactement cap segments retournés."""
        cap = 10
        P = 50
        rng = np.random.default_rng(7)
        probe_pts = rng.standard_normal((P, 3))
        witness_local = rng.standard_normal((P, 3))
        active = np.ones(P, bool)

        segs = witness_segments(
            probe_pts, witness_local, active,
            R_obj=np.eye(3), t_obj=np.zeros(3),
            cap=cap,
        )

        assert segs.shape == (cap, 2, 3)

    def test_subsample_deterministic(self):
        """Le sous-échantillonnage avec graine 0 est reproductible à l'appel identique."""
        cap = 10
        P = 100
        rng = np.random.default_rng(99)
        probe_pts = rng.standard_normal((P, 3))
        witness_local = rng.standard_normal((P, 3))
        active = np.ones(P, bool)

        segs_a = witness_segments(
            probe_pts, witness_local, active,
            R_obj=np.eye(3), t_obj=np.zeros(3), cap=cap
        )
        segs_b = witness_segments(
            probe_pts, witness_local, active,
            R_obj=np.eye(3), t_obj=np.zeros(3), cap=cap
        )

        np.testing.assert_array_equal(segs_a, segs_b)

    def test_below_cap_no_subsample(self):
        """Nombre de sondes actives < cap → tous les segments sont retournés."""
        P = 5
        probe_pts = np.arange(P * 3, dtype=float).reshape(P, 3)
        witness_local = probe_pts + 0.5
        active = np.ones(P, bool)

        segs = witness_segments(
            probe_pts, witness_local, active,
            R_obj=np.eye(3), t_obj=np.zeros(3),
            cap=400,
        )

        assert segs.shape == (P, 2, 3)

    def test_output_dtype_float32(self):
        """Le tableau retourné est float32 (pour le rendu viser)."""
        probe_pts = np.ones((3, 3), np.float64)
        witness_local = np.zeros((3, 3), np.float64)
        active = np.ones(3, bool)

        segs = witness_segments(
            probe_pts, witness_local, active,
            R_obj=np.eye(3), t_obj=np.zeros(3)
        )

        assert segs.dtype == np.float32
