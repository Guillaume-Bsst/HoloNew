"""Lecteur FBX binaire minimal — extrait la trajectoire 6-DoF d'un objet rigide d'un ``<seq>_o.fbx``.

Motivation : les FBX objets PA-HOI (export Noitom, binaire v7500) portent UN nœud ``Model`` (Mesh)
avec sa géométrie statique (repère local) + une piste d'animation ``Lcl Translation``/``Lcl Rotation``
(1 clé par frame). Aucune bibliothèque FBX n'est requise (ni dispo dans l'env) : on parse le format
node-tree documenté (records imbriqués, propriétés primitives/tableaux zlib) juste assez pour ces objets.

Convention de sortie : ``rot_native``/``transl_native`` restent dans le repère NATIF de la capture
(Y-up, mètres) — le loader applique le ``YUP_TO_ZUP`` partagé via ``frames.object_pose_zup`` (même
chemin que l'humain). Le mesh reste dans son repère LOCAL objet (mètres). Cible unique : les ``_o.fbx``
(petits, réguliers) ; ce n'est pas un parseur FBX généraliste.
"""
from __future__ import annotations

import re
import struct
import zlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation as R

# 1 KTime (unité de temps FBX) = 1/46186158000 s. Sert à retrouver le fps depuis les KeyTime.
_KTIME_PER_S = 46186158000
_AXIS = {"d|X": 0, "d|Y": 1, "d|Z": 2}


@dataclass(frozen=True)
class ObjectFbx:
    """Trajectoire + mesh d'un objet rigide extraits d'un ``_o.fbx`` (numpy-only, frozen)."""

    rot_native: np.ndarray        # (T, 3, 3)  rotation objet, repère natif Y-up
    transl_native: np.ndarray     # (T, 3)     translation objet, mètres, repère natif Y-up
    vertices: np.ndarray          # (V, 3)     mesh proxy, mètres, repère LOCAL objet
    faces: np.ndarray             # (F, 3) int triangles
    name: str                     # nom d'objet, préfixe d'index retiré ("01_milkbox" -> "milkbox")
    fps: float


# ---------------------------------------------------------------------------
# parsing bas niveau : node-tree binaire FBX
# ---------------------------------------------------------------------------
def _read_prop(b: bytes, o: int):
    """Lire une propriété (typée par 1 char) -> (valeur, offset). Gère primitives, tableaux (zlib), S/R."""
    t = chr(b[o]); o += 1
    if t == "Y": return struct.unpack_from("<h", b, o)[0], o + 2
    if t == "C": return bool(b[o]), o + 1
    if t == "I": return struct.unpack_from("<i", b, o)[0], o + 4
    if t == "F": return struct.unpack_from("<f", b, o)[0], o + 4
    if t == "D": return struct.unpack_from("<d", b, o)[0], o + 8
    if t == "L": return struct.unpack_from("<q", b, o)[0], o + 8
    if t in "fdlib":                                       # tableau
        n, enc, clen = struct.unpack_from("<III", b, o); o += 12
        raw = b[o:o + clen]; o += clen
        if enc == 1:
            raw = zlib.decompress(raw)
        fmt = {"f": "f", "d": "d", "l": "q", "i": "i", "b": "b"}[t]
        return list(struct.unpack(f"<{n}{fmt}", raw)), o
    if t in "SR":                                         # string / raw
        ln = struct.unpack_from("<I", b, o)[0]; o += 4
        data = b[o:o + ln]; o += ln
        return (data.decode("utf-8", "replace") if t == "S" else data), o
    raise ValueError(f"type de propriété FBX inconnu {t!r} à l'offset {o}")


def _read_node(b: bytes, o: int, u64: bool):
    """Lire un record de node -> (dict|None, offset). ``None`` = record null (sentinelle de fin de liste)."""
    if u64:
        end, nprops, _plen = struct.unpack_from("<QQQ", b, o); o += 24
    else:
        end, nprops, _plen = struct.unpack_from("<III", b, o); o += 12
    nlen = b[o]; o += 1
    if end == 0:                                          # record null -> fin de la liste imbriquée
        return None, o
    name = b[o:o + nlen].decode("utf-8", "replace"); o += nlen
    props = []
    for _ in range(nprops):
        v, o = _read_prop(b, o)
        props.append(v)
    children = []
    if o < end:                                           # une liste imbriquée suit (terminée par le null record)
        while True:
            child, o = _read_node(b, o, u64)
            if child is None:
                break
            children.append(child)
        o = end
    return {"name": name, "props": props, "children": children}, o


def _parse(path: Path):
    """Fichier FBX binaire -> (version, list[node racine])."""
    b = Path(path).read_bytes()
    if b[:20] != b"Kaydara FBX Binary  ":
        raise ValueError(f"pas un FBX binaire : {path}")
    version = struct.unpack_from("<I", b, 23)[0]
    u64 = version >= 7500
    o, roots = 27, []
    end_guard = len(b) - (25 if u64 else 13)
    while o < end_guard:
        node, o = _read_node(b, o, u64)
        if node is None:
            break
        roots.append(node)
    return version, roots


def _find(nodes, name):
    """Générateur récursif : tous les nodes nommés ``name`` dans l'arbre."""
    for n in nodes:
        if n["name"] == name:
            yield n
        yield from _find(n["children"], name)


def _child(node, name):
    """Premier enfant nommé ``name``, ou None."""
    return next((c for c in node["children"] if c["name"] == name), None)


# ---------------------------------------------------------------------------
# extraction objet
# ---------------------------------------------------------------------------
def _model_defaults(model) -> dict:
    """Constantes ``Lcl Translation``/``Lcl Rotation`` du Properties70 (fallback pour un axe non animé)."""
    out = {"Lcl Translation": [0.0, 0.0, 0.0], "Lcl Rotation": [0.0, 0.0, 0.0]}
    p70 = _child(model, "Properties70")
    if p70 is not None:
        for pr in p70["children"]:
            key = pr["props"][0] if pr["props"] else None
            if key in out and len(pr["props"]) >= 3:
                out[key] = [float(x) for x in pr["props"][-3:]]
    return out


def _triangulate(polygon_vertex_index) -> np.ndarray:
    """PolygonVertexIndex FBX -> triangles (fan). Le dernier index de chaque polygone est encodé négatif
    (``~idx``, complément à un) ; on le décode et on éclate chaque polygone en éventail."""
    faces, poly = [], []
    for raw in polygon_vertex_index:
        if raw < 0:
            poly.append(-raw - 1)                         # ~idx == -idx-1 : fin de polygone
            for k in range(1, len(poly) - 1):
                faces.append((poly[0], poly[k], poly[k + 1]))
            poly = []
        else:
            poly.append(raw)
    return np.asarray(faces, np.int64).reshape(-1, 3)


def _resample(times: np.ndarray, kt: np.ndarray, kv: np.ndarray) -> np.ndarray:
    """Valeurs d'une courbe sur la timeline ``times`` : lecture directe si les KeyTime coïncident,
    sinon interpolation linéaire (fallback défensif — les exports Noitom cuisent 1 clé/frame)."""
    if kt.shape == times.shape and np.array_equal(kt, times):
        return kv
    return np.interp(times.astype(np.float64), kt.astype(np.float64), kv.astype(np.float64))


def read_object_fbx(path: Path) -> ObjectFbx:
    """Lire un ``<seq>_o.fbx`` PA-HOI -> ``ObjectFbx`` (trajectoire native Y-up en mètres + mesh local)."""
    _version, roots = _parse(Path(path))

    model = None
    geom = None
    curves: dict[int, tuple[np.ndarray, np.ndarray]] = {}   # id -> (KeyTime, KeyValueFloat)
    curvenodes: set[int] = set()
    for objects in _find(roots, "Objects"):
        for c in objects["children"]:
            nm = c["name"]
            pid = c["props"][0] if c["props"] else None
            if nm == "Model" and (len(c["props"]) < 3 or c["props"][2] == "Mesh") and model is None:
                model = c
            elif nm == "Geometry" and geom is None:
                geom = c
            elif nm == "AnimationCurveNode":
                curvenodes.add(pid)
            elif nm == "AnimationCurve":
                kt = _child(c, "KeyTime"); kv = _child(c, "KeyValueFloat")
                curves[pid] = (np.asarray(kt["props"][0], np.int64) if kt else np.zeros(0, np.int64),
                               np.asarray(kv["props"][0], np.float64) if kv else np.zeros(0))
    if model is None or geom is None:
        raise ValueError(f"{path}: aucun Model(Mesh)/Geometry trouvé")

    # Connections : courbe --(d|X/Y/Z)--> curvenode --(Lcl Translation/Rotation)--> model.
    curve_to_node: dict[int, tuple[int, str]] = {}
    node_to_prop: dict[int, tuple[int, str]] = {}
    conns = list(_find(roots, "Connections"))
    for conn in (conns[0]["children"] if conns else []):
        p = conn["props"]
        if p and p[0] == "OP":
            src, dst, tag = p[1], p[2], p[3]
            if src in curves:
                curve_to_node[src] = (dst, tag)
            elif src in curvenodes:
                node_to_prop[src] = (dst, tag)

    # Assembler les 3+3 canaux par axe (T=Lcl Translation, R=Lcl Rotation).
    chan: dict[tuple[str, int], tuple[np.ndarray, np.ndarray]] = {}
    times = None
    for cid, (kt, kv) in curves.items():
        if cid not in curve_to_node:
            continue
        node, tag = curve_to_node[cid]
        prop = node_to_prop.get(node, (None, None))[1]
        ax = _AXIS.get(tag)
        if prop in ("Lcl Translation", "Lcl Rotation") and ax is not None:
            chan[(prop, ax)] = (kt, kv)
            if times is None and prop == "Lcl Translation":
                times = kt
    if times is None:                                     # aucune translation animée -> retomber sur une rotation
        times = next(iter(chan.values()))[0] if chan else np.zeros(1, np.int64)
    T = len(times)

    defaults = _model_defaults(model)
    trans = np.empty((T, 3), np.float64)
    rot_deg = np.empty((T, 3), np.float64)
    for ax in range(3):
        kt_kv = chan.get(("Lcl Translation", ax))
        trans[:, ax] = _resample(times, *kt_kv) if kt_kv else defaults["Lcl Translation"][ax]
        kt_kv = chan.get(("Lcl Rotation", ax))
        rot_deg[:, ax] = _resample(times, *kt_kv) if kt_kv else defaults["Lcl Rotation"][ax]

    transl_native = (trans * 0.01).astype(np.float64)     # cm -> m
    rot_native = R.from_euler("XYZ", rot_deg, degrees=True).as_matrix()   # ordre FBX par défaut (eEulerXYZ)

    verts = np.asarray(_child(geom, "Vertices")["props"][0], np.float64).reshape(-1, 3) * 0.01
    faces = _triangulate(_child(geom, "PolygonVertexIndex")["props"][0])

    raw_name = model["props"][1].split("\x00")[0] if len(model["props"]) > 1 else "object"
    name = re.sub(r"^\d+_", "", raw_name)                 # "01_milkbox" -> "milkbox"

    if T > 1:
        dt = float(np.median(np.diff(times.astype(np.float64))))
        fps = float(round(_KTIME_PER_S / dt)) if dt > 0 else 30.0
    else:
        fps = 30.0

    return ObjectFbx(rot_native=rot_native, transl_native=transl_native,
                     vertices=verts.astype(np.float64), faces=faces, name=name, fps=fps)
