"""GeodesicConfig: defaults sensés, validation des plages, et présence dans PrepareConfig (un seul
objet de knobs ; override inline). Le sampling (densité/seed) n'est PAS ici — il vient de CloudConfig."""
import pytest

from src.prepare.config import GeodesicConfig, PrepareConfig


def test_defaults():
    c = GeodesicConfig()
    assert c.k_neighbors == 8 and c.normal_gate == -0.5 and c.max_points == 6000


def test_prepareconfig_includes_geodesic_default():
    assert isinstance(PrepareConfig().geodesic, GeodesicConfig)
    # override inline, sans toucher aux autres sous-configs
    p = PrepareConfig(geodesic=GeodesicConfig(k_neighbors=12))
    assert p.geodesic.k_neighbors == 12
    assert p.sdf.spacing == 0.01            # défaut SdfConfig intact


@pytest.mark.parametrize("kwargs", [
    {"k_neighbors": 0}, {"normal_gate": 1.5}, {"normal_gate": -2.0}, {"max_points": 0},
])
def test_rejects_out_of_range(kwargs):
    with pytest.raises(ValueError):
        GeodesicConfig(**kwargs)
