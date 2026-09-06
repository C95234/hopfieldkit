"""A second learning mode: perceptron-style (Gardner) learning.

Hebbian learning (`hopfieldkit.hebbian`) stores patterns in one shot, but
its capacity is limited (Amit-Gutfreund-Sompolinsky, 1985: ~0.138 N).
An alternative, studied by Gardner (1988, "The space of interactions in
neural network models") and Diederich & Opper (1987, "Learning of
correlated patterns in spin-glass networks by local learning rules"), is
to train each unit's incoming weights iteratively with a perceptron-style
rule: for every stored pattern, require that unit i's local field agrees
with the pattern in sign by at least a margin kappa (`x_i * h_i > kappa`);
whenever that fails, nudge the weights towards satisfying it. Gardner's
statistical-mechanics analysis showed this class of rule can, in
principle, reach a storage capacity of up to ~2N patterns (kappa -> 0,
N -> infinity) -- far above the Hebbian bound.

Adaptation (documented, not hidden): the textbook Gardner/Diederich-Opper
capacity bound is derived for the UNCONSTRAINED per-neuron problem, which
trains each row of W independently and generally produces an ASYMMETRIC
matrix. That is fine for a pure feedforward stability criterion, but it
breaks the energy-function argument this package relies on for recall
(`hopfieldkit.base`: convergence needs a symmetric W with a zero
diagonal). `PerceptronHopfieldNetwork` therefore enforces symmetry
DURING training: whenever pattern mu violates the margin for neuron i or
neuron j (or both), the SAME update is applied to both w_ij and w_ji in
that step, so the matrix is symmetric by construction throughout, not
just symmetrized after the fact. This keeps recall inside the
well-understood Hopfield/Lyapunov framework, at a cost: it is not a
literal reproduction of the unconstrained per-neuron algorithm, so the
empirical capacity this class reaches (see
`hopfieldkit.capacity.empirical_capacity_scan` applied to it) should be
read as "a Gardner-style margin rule constrained to stay energy-based",
not assumed to hit the textbook 2N bound exactly -- `tests/` measure the
real empirical gap over Hebb rather than asserting a specific number.
"""
from __future__ import annotations

import numpy as np

from .base import BaseHopfieldNetwork


class PerceptronHopfieldNetwork(BaseHopfieldNetwork):
    """Hopfield-type network trained by a symmetric perceptron-style rule
    (Gardner, 1988; Diederich & Opper, 1987 -- see module docstring for the
    symmetry adaptation).

    Parameters
    ----------
    n_units : int
        Number of binary units (+1 / -1).
    kappa : float, optional
        Stability margin required for every pattern/unit pair
        (`x_i * h_i > kappa`). Default 0.0 (a bare correct-sign
        requirement, no margin).
    learning_rate : float, optional
        Step size for each perceptron update. Default 1.0.
    """

    def __init__(self, n_units: int, kappa: float = 0.0, learning_rate: float = 1.0):
        super().__init__(n_units)
        self.kappa = kappa
        self.learning_rate = learning_rate
        self.converged = False
        self.n_epochs_run = 0

    def fit(self, patterns, max_epochs: int = 500, seed: int | None = None) -> "PerceptronHopfieldNetwork":
        patterns = np.asarray(patterns, dtype=float)
        if patterns.ndim != 2 or patterns.shape[1] != self.n_units:
            raise ValueError(f"patterns must have shape (n_patterns, {self.n_units})")
        n_patterns, n = patterns.shape
        w = np.zeros((n, n))
        rng = np.random.default_rng(seed)
        self.converged = False
        epoch = 0

        for epoch in range(max_epochs):
            order = rng.permutation(n_patterns)
            any_update = False
            for mu in order:
                x = patterns[mu]
                local_fields = w @ x
                stabilities = x * local_fields
                violated = stabilities <= self.kappa
                if not violated.any():
                    continue
                any_update = True
                delta = (self.learning_rate / n) * np.outer(x, x)
                np.fill_diagonal(delta, 0.0)
                update_mask = violated[:, None] | violated[None, :]
                w += delta * update_mask
            if not any_update:
                self.converged = True
                break

        self.n_epochs_run = epoch + 1
        self.weights = w
        return self
