"""Hopfield network with Hebbian learning (Hopfield, 1982).

See `hopfieldkit.base` for the convergence argument shared by every
learning rule in this package (recall dynamics are the same regardless of
how the weight matrix was learned).
"""
from __future__ import annotations

import numpy as np

from .base import BaseHopfieldNetwork


class HopfieldNetwork(BaseHopfieldNetwork):
    """A Hopfield associative-memory network trained by Hebbian learning.

    Parameters
    ----------
    n_units : int
        Number of binary units (+1 / -1).
    """

    def fit(self, patterns) -> "HopfieldNetwork":
        """Store a list of +/-1 patterns via the Hebb rule.

        W = (1/n) * sum_mu (x^mu outer x^mu), diagonal zeroed.
        """
        patterns = np.asarray(patterns, dtype=float)
        if patterns.ndim != 2 or patterns.shape[1] != self.n_units:
            raise ValueError(f"patterns must have shape (n_patterns, {self.n_units})")
        w = np.zeros((self.n_units, self.n_units))
        for pattern in patterns:
            w += np.outer(pattern, pattern)
        w /= self.n_units
        np.fill_diagonal(w, 0.0)
        self.weights = w
        return self
