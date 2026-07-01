"""app — la surface d'entrée prod unifiée. Importe comme consommateur pur et expose run_app/main ;
les 7 couches sont câblées (une liste) ; AUCUN serveur viser n'est démarré (c'est le chemin
``__main__`` en prod)."""
import src.viz as viz
from src.viz import app
from src.viz.core.layer import Layer


def test_app_exposes_entry():
    assert callable(viz.run_app) and callable(viz.main)
    assert viz.run_app is app.run_app and viz.main is app.main


def test_app_layer_set_is_the_seven_ported_layers():
    # Instancie la même liste de couches que app.run_app construit, et vérifie que ce sont bien
    # les 7 couches portées (toutes des Layer).
    from src.viz.layers.fields import FieldsLayer
    from src.viz.layers.ghost import GhostLayer
    from src.viz.layers.ground import GroundLayer
    from src.viz.layers.human_cloud import HumanCloudLayer
    from src.viz.layers.objects import ObjectsLayer
    from src.viz.layers.skeleton import SkeletonLayer
    from src.viz.layers.style import StyleLayer

    layers = [GroundLayer(), GhostLayer(), SkeletonLayer(), HumanCloudLayer(),
              ObjectsLayer(), FieldsLayer(), StyleLayer()]
    assert len(layers) == 7
    assert all(isinstance(layer, Layer) for layer in layers)
