"""Garde-fou : import src.solve reste cvxpy/torch/pinocchio-free ; terms est interne (chargé à la
demande par assemble en Plan C), pas un attribut du package solve. Les builders restent importables
explicitement et n'amènent pas torch (geo_value_grad est numpy-only)."""
import sys


def test_solve_import_is_light():
    for m in ("cvxpy", "torch", "pinocchio", "src.solve", "src.solve.terms"):
        sys.modules.pop(m, None)
    # Also clear all src.solve.terms.* submodules to ensure clean slate
    to_pop = [k for k in sys.modules if k.startswith("src.solve.terms")]
    for m in to_pop:
        sys.modules.pop(m, None)

    import src.solve  # noqa: F401
    assert "cvxpy" not in sys.modules, "cvxpy leaked at import src.solve"
    assert "torch" not in sys.modules, "torch leaked at import src.solve"
    assert "pinocchio" not in sys.modules, "pinocchio leaked at import src.solve"
    assert not hasattr(src.solve, "terms"), "terms must stay internal (not exported by solve/__init__)"


def test_terms_importable_and_torch_free():
    import src.solve.terms.style       # noqa: F401
    import src.solve.terms.contact     # noqa: F401  (pulls geo_value_grad — numpy-only)
    import src.solve.terms.object      # noqa: F401
    import src.solve.terms.reg         # noqa: F401
    import src.solve.terms.constraints  # noqa: F401
    assert "torch" not in sys.modules and "cvxpy" not in sys.modules
