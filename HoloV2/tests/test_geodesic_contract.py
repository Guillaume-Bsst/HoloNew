"""GeodesicTable: invariants de forme (geo carrée (P,P), normals alignées aux points) ; et Channel
porte un geodesic optionnel (None = sol plan, défaut), sans casser la signature SDF existante."""
import numpy as np
import pytest

from src.prepare.contracts import GeodesicTable, Channel, SDF


def _sdf():
    return SDF(grid=np.zeros((2, 2, 2), np.float32), witness=np.zeros((2, 2, 2, 3), np.float32),
               origin=np.zeros(3), spacing=0.1, name="g")


def _table(P=4):
    return GeodesicTable(points=np.zeros((P, 3), np.float32), normals=np.zeros((P, 3), np.float32),
                         geo=np.zeros((P, P), np.float32), name="obj0", sampling_id="abc")


def test_table_n_points_and_ok():
    t = _table(5)
    assert t.n_points == 5


def test_table_rejects_non_square_geo():
    with pytest.raises(ValueError):
        GeodesicTable(points=np.zeros((4, 3), np.float32), normals=np.zeros((4, 3), np.float32),
                      geo=np.zeros((4, 5), np.float32), name="x")


def test_table_rejects_normals_shape_mismatch():
    with pytest.raises(ValueError):
        GeodesicTable(points=np.zeros((4, 3), np.float32), normals=np.zeros((3, 3), np.float32),
                      geo=np.zeros((4, 4), np.float32), name="x")


def test_channel_geodesic_defaults_none():
    ch = Channel("ground", None, _sdf())            # sol plan : pas de table
    assert ch.geodesic is None


def test_channel_carries_geodesic():
    ch = Channel("obj0", 0, _sdf(), geodesic=_table())
    assert isinstance(ch.geodesic, GeodesicTable)
