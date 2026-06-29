"""Le protocol RobotModel expose la surface cinématique étendue (free-flyer + jacobiennes)."""
import numpy as np

from src.prepare.contracts import RobotModel


class _Dummy:
    link_names = ("a",)
    dof = 1
    nq = 8
    nv = 7

    def link_transforms(self, q): return np.zeros((1, 3, 3)), np.zeros((1, 3))
    def rest_transforms(self): return np.zeros((1, 3, 3)), np.zeros((1, 3))
    def neutral(self): return np.zeros(self.nq)
    def integrate(self, q, v): return np.zeros(self.nq)
    def link_jacobians(self, q):
        return (np.zeros((1, 3, 3)), np.zeros((1, 3)),
                np.zeros((1, 3, self.nv)), np.zeros((1, 3, self.nv)))


def test_dummy_satisfies_protocol():
    assert isinstance(_Dummy(), RobotModel)   # runtime_checkable structural check
