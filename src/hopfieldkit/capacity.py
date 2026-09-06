"""Theoretical and empirical memory-capacity bounds for Hebbian Hopfield networks.

Two classical bounds are documented here as functions, not magic numbers:

- Hopfield's (1982) original estimate: p_max ~= N / (4 ln N) -- a rough
  bound from the original paper, based on requiring the number of
  bit-errors in a stored pattern to stay small in expectation.
- Amit, Gutfreund & Sompolinsky's (1985) replica-theory result:
  p_max ~= 0.138 * N -- the sharper, now-standard capacity bound for
  Hebbian storage with a small but nonzero fraction of recall errors
  tolerated, derived from statistical mechanics (spin-glass theory) rather
  than a simple bit-error argument.

`empirical_capacity_scan` runs the actual experiment (train, corrupt,
recall, check exact recovery) across a range of pattern counts, so the
theoretical bounds above can be compared to real behaviour rather than
just quoted.
"""
from __future__ import annotations

import numpy as np

from .hebbian import HopfieldNetwork


def hopfield_1982_estimate(n_units: int) -> float:
    """p_max ~= N / (4 ln N) (Hopfield, 1982)."""
    if n_units <= 1:
        raise ValueError("n_units must be > 1")
    return n_units / (4 * np.log(n_units))


def ags_1985_bound(n_units: int) -> float:
    """p_max ~= 0.138 * N (Amit, Gutfreund & Sompolinsky, 1985)."""
    return 0.138 * n_units


def empirical_capacity_scan(
    n_units: int,
    pattern_counts,
    n_trials: int = 30,
    flip_fraction: float = 0.15,
    seed: int | None = None,
) -> dict:
    """For each value of p in `pattern_counts`, train a fresh network on p
    random patterns, corrupt the first pattern (`flip_fraction` of bits
    flipped), recall it, and record the fraction of trials (out of
    `n_trials`) that recover it exactly.

    Returns
    -------
    dict with keys: pattern_counts (list[int]), success_rates (list[float]).
    """
    rng = np.random.default_rng(seed)
    n_flip = max(1, int(flip_fraction * n_units))
    success_rates = []

    for p in pattern_counts:
        successes = 0
        for _ in range(n_trials):
            patterns = rng.choice([-1.0, 1.0], size=(p, n_units))
            net = HopfieldNetwork(n_units).fit(patterns)
            target = patterns[0]
            corrupted = target.copy()
            idx = rng.choice(n_units, size=n_flip, replace=False)
            corrupted[idx] *= -1
            recalled = net.recall(corrupted, order="random")
            if np.array_equal(recalled, target):
                successes += 1
        success_rates.append(successes / n_trials)

    return {"pattern_counts": list(pattern_counts), "success_rates": success_rates}
