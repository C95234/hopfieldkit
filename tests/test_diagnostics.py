import numpy as np

from hopfieldkit.diagnostics import classify_recall_outcome, detect_order_dependence
from hopfieldkit.hebbian import HopfieldNetwork


def test_classify_recall_outcome_exact_match():
    stored = [np.array([1.0, 1.0, -1.0]), np.array([-1.0, 1.0, 1.0])]
    result = classify_recall_outcome(stored[0], stored)
    assert result == {"outcome": "exact_match", "pattern_index": 0}


def test_classify_recall_outcome_negative_of_pattern():
    stored = [np.array([1.0, 1.0, -1.0]), np.array([-1.0, 1.0, 1.0])]
    result = classify_recall_outcome(-stored[0], stored)
    assert result == {"outcome": "negative_of_pattern", "pattern_index": 0}


def test_classify_recall_outcome_mixture_state():
    stored = [
        np.array([1.0, 1.0, 1.0, 1.0, 1.0]),
        np.array([1.0, -1.0, 1.0, -1.0, 1.0]),
        np.array([1.0, 1.0, -1.0, -1.0, -1.0]),
    ]
    mixture = np.sign(sum(stored))
    mixture[mixture == 0] = 1.0
    result = classify_recall_outcome(mixture, stored)
    assert result["outcome"] == "mixture_state"
    assert result["pattern_indices"] == [0, 1, 2]


def test_classify_recall_outcome_unknown_state():
    stored = [np.array([1.0, 1.0, -1.0]), np.array([-1.0, 1.0, 1.0])]
    unrelated = np.array([-1.0, -1.0, -1.0])
    result = classify_recall_outcome(unrelated, stored)
    assert result["outcome"] == "unknown"


def test_detect_order_dependence_finds_a_known_parasitic_case():
    # Same 4-territory example as backend/scripts/hopfield_social.py, where
    # a randomly-ordered recall is already known to sometimes converge to
    # the negative of a stored pattern instead of the sequential outcome.
    societe_apaisee = np.array([1.0, 1.0, 1.0, 1.0])
    fracture_nordsud = np.array([1.0, 1.0, -1.0, -1.0])
    net = HopfieldNetwork(4).fit([societe_apaisee, fracture_nordsud])
    corrupted = np.array([-1.0, 1.0, -1.0, -1.0])

    result = detect_order_dependence(net, corrupted, n_random_trials=30, seed=42)
    assert result["order_dependent"] is True
    assert result["n_disagreements"] >= 1
