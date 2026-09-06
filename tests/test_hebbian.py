import numpy as np
import pytest

from hopfieldkit.hebbian import HopfieldNetwork


def test_fit_requires_correct_pattern_shape():
    net = HopfieldNetwork(4)
    with pytest.raises(ValueError):
        net.fit(np.array([[1.0, 1.0, 1.0]]))  # wrong width


def test_weights_are_symmetric_and_zero_diagonal_after_fit():
    rng = np.random.default_rng(0)
    patterns = rng.choice([-1.0, 1.0], size=(5, 10))
    net = HopfieldNetwork(10).fit(patterns)
    assert np.allclose(net.weights, net.weights.T)
    assert np.allclose(np.diag(net.weights), 0.0)


def test_energy_is_non_increasing_along_recall_trajectory():
    rng = np.random.default_rng(1)
    n = 20
    patterns = rng.choice([-1.0, 1.0], size=(3, n))
    net = HopfieldNetwork(n).fit(patterns)
    corrupted = patterns[0].copy()
    corrupted[:4] *= -1

    _, energies = net.recall(corrupted, order="sequential", record_energy=True)
    assert all(energies[i] >= energies[i + 1] - 1e-9 for i in range(len(energies) - 1))


def test_recall_reaches_a_fixed_point():
    rng = np.random.default_rng(2)
    n = 20
    patterns = rng.choice([-1.0, 1.0], size=(3, n))
    net = HopfieldNetwork(n).fit(patterns)
    corrupted = patterns[0].copy()
    corrupted[:4] *= -1

    recalled = net.recall(corrupted, order="sequential")
    recalled_again = net.recall(recalled, order="sequential")
    assert np.array_equal(recalled, recalled_again)


def test_small_example_recovers_stored_pattern():
    # Exact example from backend/scripts/hopfield_social.py (4 territories).
    societe_apaisee = np.array([1.0, 1.0, 1.0, 1.0])
    fracture_nordsud = np.array([1.0, 1.0, -1.0, -1.0])
    net = HopfieldNetwork(4).fit([societe_apaisee, fracture_nordsud])

    corrupted = np.array([-1.0, 1.0, -1.0, -1.0])
    recovered = net.recall(corrupted, order="sequential")
    assert np.array_equal(recovered, fracture_nordsud)


def test_recall_rejects_invalid_order_argument():
    net = HopfieldNetwork(4).fit([[1.0, 1.0, -1.0, -1.0]])
    with pytest.raises(ValueError):
        net.recall(np.array([1.0, 1.0, 1.0, 1.0]), order="bogus")
