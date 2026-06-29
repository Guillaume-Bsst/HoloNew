"""prepare/geodesic — construit la table all-pairs de distances géodésiques d'un mesh rigide
(objet ou terrain) sur les points du ``object_cloud``.

Offline, scopé géométrie, caché une fois (``GeodesicBuilder``) : un asset subject-/robot-free partagé
par toute séquence touchant le même mesh. Pipeline : ré-échantillonner la surface À L'IDENTIQUE du
``object_cloud`` (mêmes densité/seed → points bit-identiques) en récupérant les normales, construire un
graphe k-NN de surface (poids euclidien) GATÉ par les normales (coupe les arêtes "cross-gap" des
plaques fines), puis Dijkstra all-pairs (scipy). La matrice est consommée comme des CHAMPS mono-source
(``geo[j]``) — voir ``GeodesicTable``. Sampling du champ à un ``witness(q)`` continu : pas ici (online,
``targets/interaction/geodesic.py``).

scipy/trimesh sont confinés à ce builder (il tourne une fois et est caché) ; le contrat et la query
restent numpy-only."""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

from ..contracts import GeodesicTable
from ..config import CloudConfig, GeodesicConfig
from .cache import load_geo, save_geo


def sample_surface_with_normals(vertices: np.ndarray, faces: np.ndarray, density: float,
                                seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Rejoue EXACTEMENT le sampling du ``object_cloud`` (``max(64, int(area*density))`` points,
    ``sample_surface_even`` déterministe en ``seed``) → points bit-identiques au cloud, PLUS la normale
    de surface par point (normale de la face échantillonnée). Pur : construit un trimesh interne, no I/O."""
    import trimesh
    mesh = trimesh.Trimesh(vertices=np.asarray(vertices, np.float64),
                           faces=np.asarray(faces), process=False)
    n = max(64, int(float(mesh.area) * density))
    pts, fid = trimesh.sample.sample_surface_even(mesh, n, seed=seed)   # fid = index de face par point
    normals = np.asarray(mesh.face_normals)[np.asarray(fid)]
    return np.asarray(pts, np.float64), np.asarray(normals, np.float64)


def build_knn_graph(points: np.ndarray, normals: np.ndarray, k: int, normal_gate: float):
    """Graphe k-NN de surface (poids = distance euclidienne), GATÉ par normales : arête i--j seulement
    si ``dot(n_i, n_j) > normal_gate``. Symétrisé (non orienté). Pur ; renvoie une ``csr_matrix`` (P,P)."""
    from scipy.spatial import cKDTree
    from scipy.sparse import csr_matrix
    pts = np.asarray(points, np.float64)
    nrm = np.asarray(normals, np.float64)
    p = pts.shape[0]
    kq = min(k + 1, p)                                          # +1 : le 1er voisin est soi-même
    dist, idx = cKDTree(pts).query(pts, k=kq)
    dist = np.atleast_2d(dist); idx = np.atleast_2d(idx)
    src = np.repeat(np.arange(p), kq - 1)
    dst = idx[:, 1:].reshape(-1)
    wts = dist[:, 1:].reshape(-1)
    keep = np.einsum("ij,ij->i", nrm[src], nrm[dst]) > normal_gate
    src, dst, wts = src[keep], dst[keep], wts[keep]
    a = csr_matrix((wts, (src, dst)), shape=(p, p))
    return a.maximum(a.T)                                       # non orienté : arête si l'un OU l'autre


def _bridge_disconnected(points: np.ndarray, graph):
    """Garantit un graphe de surface CONNEXE. Si le k-NN gaté se fragmente (mesh fin/non-manifold,
    échantillonnage clairsemé), relie chaque composante à l'ensemble déjà connecté par leur paire de
    points EUCLIDIENNE la plus proche (le pont ≈ géodésique à travers le trou d'échantillonnage),
    plutôt que de laisser ``all_pairs_geodesic`` lever et avorter tout ``prepare``. Pur, déterministe ;
    no-op si déjà connexe."""
    from scipy.sparse.csgraph import connected_components
    from scipy.sparse import csr_matrix
    from scipy.spatial import cKDTree
    n_comp, labels = connected_components(graph, directed=False)
    if n_comp <= 1:
        return graph
    pts = np.asarray(points, np.float64)
    order = np.argsort(labels, kind="stable")                 # indices groupés par composante
    comp_points = [order[labels[order] == c] for c in range(n_comp)]
    rows, cols, wts = [], [], []
    accumulated = comp_points[0]                              # ensemble connecté (croît)
    for c in range(1, n_comp):
        cur = comp_points[c]
        tree = cKDTree(pts[accumulated])
        dd, ii = tree.query(pts[cur])                         # plus proche point connecté par point de cur
        j = int(np.argmin(dd))
        a = int(cur[j]); bnode = int(accumulated[ii[j]]); d = float(dd[j])
        rows += [a, bnode]; cols += [bnode, a]; wts += [d, d]  # arête de pont (non orientée)
        accumulated = np.concatenate([accumulated, cur])
    bridges = csr_matrix((wts, (rows, cols)), shape=graph.shape)
    return (graph + bridges)                                  # union (croisements sans recouvrement)


def all_pairs_geodesic(graph) -> np.ndarray:
    """All-pairs plus court chemin (Dijkstra) sur le graphe de surface → (P,P) f32. Lève si le graphe
    est disconnecté (un ``inf`` = paire sans chemin) plutôt que de stocker des ``inf``. Dans le chemin
    normal, ``_bridge_disconnected`` est appelé en amont et rend ce cas inatteignable ; ce ValueError
    reste comme filet de sécurité si on passe un graphe brut directement."""
    from scipy.sparse.csgraph import shortest_path
    d = shortest_path(graph, method="D", directed=False)
    if not np.isfinite(d).all():
        raise ValueError("geodesic graph is disconnected (some pairs have no path) — "
                         "increase k_neighbors or lower normal_gate")
    d = 0.5 * (d + d.T)                                         # nettoie l'asymétrie FP
    return d.astype(np.float32)


def _sampling_id(cloud_cfg: CloudConfig, vertices: np.ndarray, faces: np.ndarray) -> str:
    """Hash stable du sampling (densité+seed) + géométrie — provenance/garde-fou (16 hex)."""
    h = hashlib.sha1()
    h.update(f"{cloud_cfg.object_density}|{cloud_cfg.seed}".encode())
    h.update(np.ascontiguousarray(vertices, np.float32).tobytes())
    h.update(np.ascontiguousarray(faces, np.int64).tobytes())
    return h.hexdigest()[:16]


def build_geodesic_table(vertices: np.ndarray, faces: np.ndarray, cloud_cfg: CloudConfig,
                         geo_cfg: GeodesicConfig, name: str = "") -> GeodesicTable:
    """Échantillonne (à l'identique du cloud) puis calcule la table géodésique. Garde-fou ``max_points``
    (stockage 4*P^2). Pur : no I/O, ne mute pas ses inputs."""
    pts, nrm = sample_surface_with_normals(vertices, faces, cloud_cfg.object_density, cloud_cfg.seed)
    p = pts.shape[0]
    if p > geo_cfg.max_points:
        raise ValueError(f"geodesic sampling has P={p} > max_points={geo_cfg.max_points} "
                         f"(storage is 4*P^2 bytes) — lower object_density or raise max_points")
    graph = build_knn_graph(pts, nrm, geo_cfg.k_neighbors, geo_cfg.normal_gate)
    geo = all_pairs_geodesic(_bridge_disconnected(pts, graph))
    return GeodesicTable(points=pts.astype(np.float32), normals=nrm.astype(np.float32), geo=geo,
                         name=name, sampling_id=_sampling_id(cloud_cfg, vertices, faces))


class GeodesicBuilder:
    """``AssetBuilder`` de la table géodésique d'un mesh (objet/terrain). Scopé GÉOMÉTRIE (+ le
    sampling ``CloudConfig`` qui fixe les points, + les knobs graphe ``GeodesicConfig``) : deux
    séquences partageant un mesh partagent la table cachée, indépendamment du subject/robot. Le runner
    enveloppe ``build``/``load`` dans un ``prof.span("sdf+geodesic")``. ``max_points`` est un garde-fou
    (ne change pas l'asset produit) ⇒ HORS clé."""

    def cache_key(self, cloud_cfg: CloudConfig, geo_cfg: GeodesicConfig, vertices: np.ndarray,
                  faces: np.ndarray) -> str:
        h = hashlib.sha1()
        h.update(f"{cloud_cfg.object_density}|{cloud_cfg.seed}|{geo_cfg.k_neighbors}|"
                 f"{geo_cfg.normal_gate}".encode())
        h.update(np.ascontiguousarray(vertices, np.float32).tobytes())
        h.update(np.ascontiguousarray(faces, np.int64).tobytes())
        return h.hexdigest()

    def build(self, cloud_cfg: CloudConfig, geo_cfg: GeodesicConfig, vertices: np.ndarray,
              faces: np.ndarray, name: str = "") -> GeodesicTable:
        return build_geodesic_table(vertices, faces, cloud_cfg, geo_cfg, name=name)

    def save(self, table: GeodesicTable, path: Path) -> None:
        return save_geo(table, path)

    def load(self, path: Path) -> GeodesicTable:
        return load_geo(path)
