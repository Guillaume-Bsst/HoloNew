import numpy as np

from HoloNew.evaluation.metrics.contacts import compute_contacts


def test_perfect_agreement():
    rc = np.array([[True, False], [True, True], [True, False]])
    m = compute_contacts(rc, rc.copy(), np.zeros((3, 2)), np.zeros((3, 2)))
    assert m["contact_precision"] == 1.0
    assert m["contact_recall"] == 1.0
    assert m["contact_f1"] == 1.0
    assert m["contact_place_err"] == 0.0
    assert m["contact_slip_mean"] == 0.0


def test_one_fp_one_fn():
    # ref positives: (0,0),(1,0); robot positives: (0,0),(2,0)
    ref = np.array([[True], [True], [False]])
    rob = np.array([[True], [False], [True]])
    m = compute_contacts(rob, ref, np.zeros((3, 1)), np.zeros((3, 1)))
    # TP=1 (0,0); FP=1 (2,0); FN=1 (1,0)
    assert abs(m["contact_precision"] - 0.5) < 1e-12
    assert abs(m["contact_recall"] - 0.5) < 1e-12
    assert abs(m["contact_f1"] - 0.5) < 1e-12


def test_placement_only_on_agreed_contacts():
    ref = np.array([[True], [True]])
    rob = np.array([[True], [False]])           # only frame 0 agrees
    place = np.array([[0.03], [0.99]])          # frame 1 should be ignored
    m = compute_contacts(rob, ref, place, np.zeros((2, 1)))
    assert abs(m["contact_place_err"] - 0.03) < 1e-12


def test_slip_only_on_sustained_contact():
    # contact frames 1,2,3 sustained; frame 0 isolated. slip values per frame.
    rob = np.array([[True], [True], [True], [True]])
    ref = rob.copy()
    slip = np.array([[5.0], [0.1], [0.2], [0.3]])  # frame 0 not sustained -> excluded
    m = compute_contacts(rob, ref, np.zeros((4, 1)), slip)
    # sustained frames = 1,2,3 -> mean(0.1,0.2,0.3) = 0.2
    assert abs(m["contact_slip_mean"] - 0.2) < 1e-12
