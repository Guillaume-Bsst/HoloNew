"""Les 4 couches d'interaction roadmap sont importables, instanciables, conformes au protocole Layer
(attribut ``folder: str`` + ``setup``/``update`` appelables) et enregistrees par ``app.py``. Screen-free
(aucun viser construit) : on n'instancie que les classes et on inspecte ``app.py`` pour la presence des
couches dans son assemblage."""
import inspect

from src.viz.layers.contacts import ContactsLayer
from src.viz.layers.correspondence import CorrespondenceLayer
from src.viz.layers.geodesic import GeodesicLayer
from src.viz.layers.sdf_iso import SdfIsoLayer


def test_layers_conform_to_protocol():
    for cls in (ContactsLayer, CorrespondenceLayer, SdfIsoLayer, GeodesicLayer):
        layer = cls()
        assert isinstance(layer.folder, str) and layer.folder
        assert callable(layer.setup) and callable(layer.update)
        # signatures attendues : setup(server, gui, ctx) / update(frame, ui)
        assert list(inspect.signature(layer.setup).parameters)[:3] == ["server", "gui", "ctx"]
        assert list(inspect.signature(layer.update).parameters)[:2] == ["frame", "ui"]


def test_app_registers_the_four_layers():
    import src.viz.app as app
    src = inspect.getsource(app)
    for name in ("ContactsLayer", "CorrespondenceLayer", "SdfIsoLayer", "GeodesicLayer"):
        assert name in src, f"{name} not wired in app.py"
