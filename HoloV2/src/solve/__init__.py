"""``solve`` stage — online, q-DEPENDENT: turns the per-frame ``targets`` outputs (Evaluator + Refs)
into the retargeted ``qpos`` trajectory by a linearised QP (SQP/trust-region) loop. Public surface:
``solve.contracts`` (the data types), ``solve.config`` (knobs), and ``solve.runner.solve`` (entry).
Imports the upstream ``targets`` public surface; never a ``targets`` internal. cvxpy is confined to
``solve/backend/cvxpy.py`` — ``solve`` stays pinocchio/torch-free."""
