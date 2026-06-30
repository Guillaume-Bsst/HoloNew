"""(Dé)sérialisation de la correspondance humain<->robot ``.npz`` — sauvegarde ET chargement au
même endroit, de sorte que l'écrivain et le lecteur du format ne peuvent pas diverger. L'actif est
la paire ``(CorrespondenceTable, SurfaceSampling)`` : la table mappe chaque point de surface du
robot à son échantillon humain moteur, l'échantillonnage est le ``(tri_idx, bary)`` canonique que le
nuage humain doit réutiliser pour garder ``smpl_idx`` valide.

Cartographie des champs (contrat <-> .npz) : ``smpl_idx`` <-> ``human_idx`` (l'échantillon humain
moteur de chaque point du robot), ``link_idx`` / ``offset_local`` / ``link_names`` se reportent
directement, et ``tri_idx`` / ``bary`` sont l'échantillonnage canonique. ``smpl_sampling_id`` n'est
pas stocké : il est recalculé à partir de ``tri_idx`` / ``bary`` au chargement pour qu'il correspond
toujours à l'échantillonnage et l'assertion de liaison du runner (``cloud.sampling_id ==
table.smpl_sampling_id``) tient bon.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from ...contracts import CorrespondenceTable
from ..sampling import SurfaceSampling, sampling_id


def save_correspondence(asset: tuple[CorrespondenceTable, SurfaceSampling], path: Path) -> None:
    """Écrit la paire ``(CorrespondenceTable, SurfaceSampling)`` dans ``path`` comme un ``.npz``
    (crée les répertoires parents). Le schéma est celui que ``load_correspondence`` lit ;
    ``smpl_sampling_id`` est omis (redérivé à partir de ``tri_idx`` / ``bary`` au chargement)."""
    table, sampling = asset
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    np.savez(Path(path), human_idx=table.smpl_idx, link_idx=table.link_idx,
             offset_local=table.offset_local, link_names=np.array(table.link_names, dtype="<U64"),
             tri_idx=sampling.tri_idx, bary=sampling.bary)


def load_correspondence(path: Path) -> tuple[CorrespondenceTable, SurfaceSampling]:
    """Lit ``corr_neutral.npz`` → ``(CorrespondenceTable, SurfaceSampling)``. L'échantillonnage est
    l'identité canonique du nuage humain ; le nuage humain se crée contre lui et en hérite l'id."""
    d = np.load(Path(path), allow_pickle=False)
    sid = sampling_id(d["tri_idx"], d["bary"])
    sampling = SurfaceSampling(tri_idx=d["tri_idx"].astype(np.int64),
                               bary=d["bary"].astype(np.float32), sampling_id=sid)
    table = CorrespondenceTable(
        smpl_idx=d["human_idx"].astype(np.int64),
        link_idx=d["link_idx"].astype(np.int64),
        offset_local=d["offset_local"].astype(np.float32),
        link_names=tuple(str(x) for x in d["link_names"]),
        smpl_sampling_id=sid,
    )
    if int(table.smpl_idx.max()) >= sampling.n_points:
        raise ValueError(f"smpl_idx max {int(table.smpl_idx.max())} out of range for "
                         f"{sampling.n_points} human samples — cache is inconsistent")
    return table, sampling
