"""viser-confined helpers pour les layers / Player / app — conversion quaternion, le ``hide`` (remplace
le hack légal « triangle dégénéré à opacité 0 »), et fins wrappers add_*. Les helpers purs
(``quat_wxyz_to_R``, ``hide``) s'importent et se testent sans écran ; viser ne s'utilise que via
la scène d'un ``server`` passé par l'appelant (les wrappers n'importent jamais viser)."""
from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation as _Rot


def quat_wxyz_to_R(quat: np.ndarray) -> np.ndarray:
    """(L, 4) quaternions wxyz -> (L, 3, 3) matrices de rotation (scipy est xyzw). Le seul chemin
    quat->R (autrefois dupliqué ×3 à travers les viewers)."""
    q = np.asarray(quat, np.float64)
    return _Rot.from_quat(q[:, [1, 2, 3, 0]]).as_matrix()


def hide(handle) -> None:
    """Cache un handle de scène viser. Remplace le vieux trick « ajouter un triangle dégénéré
    à opacité 0 » : tout handle de scène viser moderne expose ``.visible``, donc cacher est
    une simple assignation."""
    handle.visible = False


def add_point_cloud(server, name: str, points, colors, *, point_size: float):
    """Fin wrapper sur ``server.scene.add_point_cloud`` (viser-confined). Retourne le handle."""
    return server.scene.add_point_cloud(name, np.asarray(points, np.float32),
                                        np.asarray(colors, np.uint8), point_size=float(point_size))


def add_line_segments(server, name: str, segments, colors, *, line_width: float):
    """Fin wrapper sur ``server.scene.add_line_segments``. ``segments`` (S, 2, 3), ``colors`` (S, 2, 3)."""
    return server.scene.add_line_segments(name, np.asarray(segments, np.float32),
                                          np.asarray(colors, np.uint8), line_width=float(line_width))


def add_label(server, name: str, text: str, position):
    """Fin wrapper sur ``server.scene.add_label`` (viser-confined). Retourne le handle."""
    return server.scene.add_label(name, text, position=tuple(float(v) for v in position))
