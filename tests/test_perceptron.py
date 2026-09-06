import numpy as np
import pytest

from hopfieldkit.hebbian import HopfieldNetwork
from hopfieldkit.perceptron import PerceptronHopfieldNetwork


def test_fit_requires_correct_pattern_shape():
    net = PerceptronHopfieldNetwork(4)
    with pytest.raises(ValueError):
        net.fit(np.array([[1.0, 1.0, 1.0]]))


def test_weights_are_symmetric_and_zero_diagonal_after_fit():
    rng = np.random.default_rng(0)
    patterns = rng.choice([-1.0, 1.0], size=(5, 15))
    net = PerceptronHopfieldNetwork(15).fit(patterns, seed=0)
    assert np.allclose(net.weights, net.weights.T)
    assert np.allclose(np.diag(net.weights), 0.0)


def test_converges_and_stabilises_all_patterns_within_easy_capacity():
    # Well within any plausible capacity bound (p << N) -- must converge and
    # leave every pattern a genuine stable point of its own dynamics.
    rng = np.random.default_rng(1)
    n, p = 40, 4
    patterns = rng.choice([-1.0, 1.0], size=(p, n))
    net = PerceptronHopfieldNetwork(n, kappa=0.1).fit(patterns, max_epochs=200, seed=1)

    assert net.converged
    for pattern in patterns:
        stabilities = pattern * (net.weights @ pattern)
        assert (stabilities > 0).all()


def test_recall_recovers_stored_pattern_from_corruption():
    rng = np.random.default_rng(2)
    n, p = 40, 4
    patterns = rng.choice([-1.0, 1.0], size=(p, n))
    net = PerceptronHopfieldNetwork(n).fit(patterns, seed=2)

    target = patterns[0].copy()
    corrupted = target.copy()
    idx = rng.choice(n, size=4, replace=False)
    corrupted[idx] *= -1

    recalled = net.recall(corrupted, order="sequential")
    assert np.array_equal(recalled, target)


def test_perceptron_reaches_higher_capacity_than_hebbian_on_same_task():
    # The central empirical claim: at a pattern count comfortably beyond the
    # Hebbian bound (~0.138*N) but still low relative to N, the perceptron
    # rule should recover stored patterns far more often than Hebb.
    n = 50
    p = 15  # 0.138*50 ~= 6.9 -- well beyond Hebbian's expected reliable range
    n_trials = 20
    flip_fraction = 0.1
    rng = np.random.default_rng(3)

    def success_rate(fit_network):
        successes = 0
        for trial in range(n_trials):
            patterns = rng.choice([-1.0, 1.0], size=(p, n))
            net = fit_network(patterns, trial)
            target = patterns[0].copy()
            corrupted = target.copy()
            n_flip = max(1, int(flip_fraction * n))
            idx = rng.choice(n, size=n_flip, replace=False)
            corrupted[idx] *= -1
            recalled = net.recall(corrupted, order="sequential")
            if np.array_equal(recalled, target):
                successes += 1
        return successes / n_trials

    hebb_rate = success_rate(lambda patterns, trial: HopfieldNetwork(n).fit(patterns))
    perceptron_rate = success_rate(lambda patterns, trial: PerceptronHopfieldNetwork(n).fit(patterns, seed=trial))

    assert perceptron_rate > hebb_rate


def test_kappa_defaults_to_zero():
    net = PerceptronHopfieldNetwork(4)
    assert net.kappa == 0.0
