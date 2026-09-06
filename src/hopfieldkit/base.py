"""Shared recall dynamics for any Hopfield-type network, independent of how
the weight matrix was learned (Hebbian rule or perceptron rule).

Convergence sketch: under asynchronous update, the energy
E(s) = -1/2 * s^T W s is non-increasing at every single-unit flip, because
a unit only flips when doing so strictly lowers its own contribution to E
(the update rule ``s_i <- sign(W_i . s)`` is exactly the greedy minimizer
of E with respect to s_i alone). Since E is bounded below on the finite
state space {-1, +1}^n and W is symmetric with a zero diagonal, repeated
asynchronous updates must reach a fixed point (Hopfield, 1982). This holds
regardless of how W was learned, as long as W stays symmetric with a zero
diagonal -- true for the Hebb rule by construction, and enforced explicitly
for the perceptron rule in `perceptron.py`.
"""
from __future__ import annotations

import numpy as np


class BaseHopfieldNetwork:
    def __init__(self, n_units: int):
        self.n_units = n_units
        self.weights = np.zeros((n_units, n_units))

    def energy(self, state) -> float:
        state = np.asarray(state, dtype=float)
        return float(-0.5 * state @ self.weights @ state)

    def recall(
        self,
        state,
        order: str = "random",
        max_steps: int = 200,
        record_energy: bool = False,
        seed: int | None = None,
    ):
        """Asynchronous update until convergence (no unit flips in a full pass).

        order: "random" reshuffles the update order every pass (classic
        Hopfield dynamics); "sequential" always updates 0, 1, ..., n-1.
        """
        if order not in ("random", "sequential"):
            raise ValueError('order must be "random" or "sequential"')
        rng = np.random.default_rng(seed)
        s = np.asarray(state, dtype=float).copy()
        energies = [self.energy(s)] if record_energy else None
        update_order = np.arange(self.n_units)

        for _ in range(max_steps):
            if order == "random":
                rng.shuffle(update_order)
            changed = False
            for i in update_order:
                local_field = self.weights[i] @ s
                new_si = 1.0 if local_field >= 0 else -1.0
                if new_si != s[i]:
                    s[i] = new_si
                    changed = True
                if record_energy:
                    energies.append(self.energy(s))
            if not changed:
                break

        return (s, energies) if record_energy else s
