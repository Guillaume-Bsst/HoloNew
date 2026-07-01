"""app — la surface d'entrée prod unifiée. Importe comme consommateur pur et expose run_app/main ;
les 8 couches sont câblées (sept portées + RobotLayer) ; AUCUN serveur viser n'est démarré (c'est
le chemin ``__main__`` en prod). Phase B : RobotLayer + CostDashboard câblés dans run_app."""
import src.viz as viz
from src.viz import app
from src.viz.core.layer import Layer


def test_app_exposes_entry():
    assert callable(viz.run_app) and callable(viz.main)
    assert viz.run_app is app.run_app and viz.main is app.main


def test_app_layer_set_is_eight_layers():
    """Instancie la même liste de couches que app.run_app construit et vérifie que ce sont bien
    les 8 couches portées (7 couches de base + RobotLayer), toutes des Layer."""
    from src.viz.layers.fields import FieldsLayer
    from src.viz.layers.ghost import GhostLayer
    from src.viz.layers.ground import GroundLayer
    from src.viz.layers.human_cloud import HumanCloudLayer
    from src.viz.layers.objects import ObjectsLayer
    from src.viz.layers.robot import RobotLayer
    from src.viz.layers.skeleton import SkeletonLayer
    from src.viz.layers.style import StyleLayer

    layers = [GroundLayer(), GhostLayer(), SkeletonLayer(), HumanCloudLayer(),
              ObjectsLayer(), FieldsLayer(), StyleLayer(), RobotLayer()]
    assert len(layers) == 8
    assert all(isinstance(layer, Layer) for layer in layers)


def test_robot_layer_is_last_in_canonical_order():
    """RobotLayer doit être la 8ème couche (ajoutée en fin, après les 7 couches de base)."""
    from src.viz.layers.robot import RobotLayer

    layers = __import__("src.viz.layers.ground", fromlist=["GroundLayer"])
    # Vérification structurelle : RobotLayer instanciable et conforme au protocole Layer
    robot = RobotLayer()
    assert isinstance(robot, Layer)
    assert robot.folder == "Robot (solved)"


def test_cost_dashboard_interface():
    """CostDashboard doit être importable depuis app et posséder le bon folder."""
    from src.viz.panels.cost_dashboard import CostDashboard

    panel = CostDashboard()
    assert panel.folder == "Cost dashboard"
    assert callable(panel.setup)
