"""Diagnostics for known limitations of Hopfield associative memory.

Two well-documented failure modes are checked here (not just described):

1. **Spurious ("parasitic") attractors** -- states the network converges
   to that were never stored. The most classic case is the exact negative
   of a stored pattern, always a stable state too, because the energy
   function is symmetric under global sign flip (E(-s) = E(s)). Mixtures
   of an odd number of stored patterns are also known to be stable for
   small numbers of patterns.
2. **Update-order dependence** -- for the SAME starting state,
   asynchronous recall can converge to different fixed points depending
   on the order units are updated in. This is a real, documented property
   of Hopfield dynamics, not an implementation bug.
"""
from __future__ import annotations

from itertools import combinations

import numpy as np


def classify_recall_outcome(recalled, stored_patterns) -> dict:
    """Classify a recalled state against the list of patterns that were
    actually stored.

    Returns
    -------
    dict with key "outcome" in {"exact_match", "negative_of_pattern",
    "mixture_state", "unknown"}, plus "pattern_index" (or
    "pattern_indices" for mixtures) identifying which stored pattern(s)
    are involved.
    """
    recalled = np.asarray(recalled, dtype=float)
    stored = np.asarray(stored_patterns, dtype=float)

    for idx, pattern in enumerate(stored):
        if np.array_equal(recalled, pattern):
            return {"outcome": "exact_match", "pattern_index": idx}

    for idx, pattern in enumerate(stored):
        if np.array_equal(recalled, -pattern):
            return {"outcome": "negative_of_pattern", "pattern_index": idx}

    n_patterns = len(stored)
    if n_patterns >= 3:
        for combo in combinations(range(n_patterns), 3):
            mixture = np.sign(stored[list(combo)].sum(axis=0))
            mixture[mixture == 0] = 1.0
            if np.array_equal(recalled, mixture):
                return {"outcome": "mixture_state", "pattern_indices": list(combo)}

    return {"outcome": "unknown", "pattern_index": None}


def detect_order_dependence(network, state, max_steps: int = 200, n_random_trials: int = 10, seed=None) -> dict:
    """Recall from the SAME starting state with sequential order, then with
    several independently-seeded random orders, and check for disagreement.

    Parameters
    ----------
    network : an object with a `.recall(state, order=..., max_steps=..., seed=...)`
        method, e.g. `hopfieldkit.hebbian.HopfieldNetwork` or
        `hopfieldkit.perceptron.PerceptronHopfieldNetwork`.
    """
    rng = np.random.default_rng(seed)
    sequential_outcome = network.recall(state, order="sequential", max_steps=max_steps)

    random_outcomes = []
    for _ in range(n_random_trials):
        trial_seed = int(rng.integers(0, 2**31 - 1))
        random_outcomes.append(network.recall(state, order="random", max_steps=max_steps, seed=trial_seed))

    disagreements = [not np.array_equal(sequential_outcome, outcome) for outcome in random_outcomes]

    return {
        "order_dependent": any(disagreements),
        "sequential_outcome": sequential_outcome,
        "random_outcomes": random_outcomes,
        "n_disagreements": int(sum(disagreements)),
    }
