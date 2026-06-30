"""Config de l'étage ``solve`` — les KNOBS QP (une seule classe gelée, stdlib-only), co-localisée avec
l'étage (règle #2). ``SolveConfig()`` EST le défaut ; override en ligne
(``SolveConfig(w_cd=5.0, tr_joints=0.2)``), exactement comme ``targets.config.TargetsConfig``.

Trois familles de knobs :
  * POIDS par-terme (``w_pos`` … ``w_reg``) — les gains de coût repliés dans ``A``/``c`` de chaque
    ``ResidualBlock`` par les builders de ``solve/terms`` (le levier de tuning #1 ; voir ``FrameInfo.cost_by_term``);
  * ACTIVATION du contact — ``contact_gate`` (lignes seulement pour les paires actives démontrées) +
    un affaiblissement doux ``contact_d_ref_scale`` qui réduit les contacts démontrés éloignés (le ``alpha`` V1);
  * région de confiance + boucle — rayons de boîte par-DOF (unités hétérogènes : base m / base rad / joints rad /
    objet m+rad) et budget d'itérations SQP / tol de convergence / nom du backend.

Les VECTEURS de poids par-lien / par-canal sont un raffinement futur ; les poids v1 sont scalaires."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SolveConfig:
    """Tous les knobs de la boucle QP ``solve``. Gelés, stdlib-only, importables partout."""

    # --- poids par-terme (repliés dans ResidualBlock A et c) ----------------------------------
    w_pos: float = 1.0     # S-pos : suivi de position de lien de style
    w_rot: float = 0.5     # S-rot : suivi d'orientation de lien de style
    w_cd: float = 2.0      # C-D   : distance de contact robot (vs canal)
    w_cx: float = 1.0      # C-X   : géodésique de contact robot (witness sur la surface)
    w_cod: float = 1.0     # CO-D  : distance d'auto-contact d'objet (objet vs sol/autres objets)
    w_cox: float = 0.5     # CO-X  : géodésique d'auto-contact d'objet (DÉFÉRÉ en v1, voir plan)
    w_obj: float = 1.0     # O     : ancrage de pose d'objet à sa pose observée
    w_reg: float = 1e-2    # reg   : amortissement de pas (QP bien-conditionnée)

    # --- activation du contact ---------------------------------------------------------------
    contact_gate: bool = True        # lignes seulement pour les paires actives dans le champ démo (référence)
    contact_d_ref_scale: float = 0.05  # affaiblissement doux : poids *= exp(-(max(d_ref,0)/scale)^2);
                                       # <= 0 désactive l'affaiblissement (poids 1 pour paires actives)

    # --- rayons de boîte par-DOF de région de confiance (TrustRegion.radius, par-DOF, norm=-1) --
    tr_base_pos: float = 0.05   # pas de translation du free-flyer (m)   -> v[0:3]
    tr_base_rot: float = 0.10   # pas de rotation du free-flyer (rad)    -> v[3:6]
    tr_joints: float = 0.10     # pas d'articulation actionnée (rad)     -> v[6:6+dof]
    tr_object_pos: float = 0.05  # pas de translation d'objet (m)        -> δξ[0:3] par objet
    tr_object_rot: float = 0.10  # pas de rotation d'objet (rad)         -> δξ[3:6] par objet

    # --- boucle SQP --------------------------------------------------------------------------
    n_iter_first: int = 10       # itérations pour la trame de démarrage à froid (absorbe le raffinement articulaire)
    n_iter_per_frame: int = 4    # itérations pour les trames avec warm-start
    step_tol: float = 1e-4       # convergence : ‖dv‖ < step_tol
    backend: str = "cvxpy"       # backend de résolution (clé usine du Plan A)
    robot_name: str | None = None  # label optionnel transmis par runner.solve (pas de validation)

    def __post_init__(self) -> None:
        for name in ("w_pos", "w_rot", "w_cd", "w_cx", "w_cod", "w_cox", "w_obj", "w_reg"):
            if getattr(self, name) < 0.0:
                raise ValueError(f"SolveConfig.{name} must be >= 0, got {getattr(self, name)}")
        for name in ("tr_base_pos", "tr_base_rot", "tr_joints", "tr_object_pos", "tr_object_rot"):
            if getattr(self, name) <= 0.0:
                raise ValueError(f"SolveConfig.{name} must be > 0, got {getattr(self, name)}")
        if self.step_tol <= 0.0:
            raise ValueError(f"SolveConfig.step_tol must be > 0, got {self.step_tol}")
        if self.n_iter_first < 1 or self.n_iter_per_frame < 1:
            raise ValueError("SolveConfig.n_iter_* must be >= 1")
        if self.backend not in ("cvxpy",):
            raise ValueError(f"SolveConfig.backend must be 'cvxpy', got {self.backend!r}")
