# hopfieldkit

Hopfield associative-memory networks in Python: Hebbian learning, a
second learning mode by perceptron-style (Gardner) descent, capacity
theory, and diagnostics for the network's well-known limitations
(spurious attractors, update-order dependence).

This package exists because the Python ecosystem for Hopfield networks
is fragmented — a handful of small, isolated scripts and gists, none
with tests, documentation, or a second learning mode to compare against.
It grew out of a pedagogical demonstration built for
[Hélios](https://github.com/C95234/Helios), a research/outreach project
that uses a Hopfield network to model collective memory in social
groups — generalised here into a standalone, tested package, published
as a separate gesture to the research community rather than a
dependency of that project (Hélios keeps its own implementation and
never depends on this package).

## Install

```bash
pip install hopfieldkit          # once published
pip install -e .                 # from a local checkout, for now
```

## Two learning modes

**Hebbian** (Hopfield, 1982) — one-shot, biologically motivated, but
capacity-limited (~0.138·N patterns, Amit-Gutfreund-Sompolinsky, 1985):

```python
import numpy as np
from hopfieldkit import HopfieldNetwork

societe_apaisee  = np.array([1, 1, 1, 1])
fracture_nordsud = np.array([1, 1, -1, -1])

net = HopfieldNetwork(n_units=4).fit([societe_apaisee, fracture_nordsud])

corrupted = np.array([-1, 1, -1, -1])
recovered = net.recall(corrupted, order="sequential")
print(recovered)  # -> [1, 1, -1, -1], fracture_nordsud recovered
```

**Perceptron-style** (Gardner, 1988; Diederich & Opper, 1987) — an
iterative margin rule that reaches a much higher capacity than Hebb, at
the cost of needing multiple passes over the data instead of a single
Hebbian sum:

```python
from hopfieldkit import PerceptronHopfieldNetwork

net = PerceptronHopfieldNetwork(n_units=100, kappa=0.0)
net.fit(patterns, max_epochs=500, seed=0)
```

### The capacity gap, measured

`hopfieldkit.capacity.empirical_capacity_scan` runs the actual
experiment — train, corrupt 15% of the bits, recall, check exact
recovery — across a range of pattern counts. At N=100 (20 trials per
point, 15% corruption):

| patterns stored | Hebb: exact recovery rate | Perceptron: exact recovery rate |
|---:|---:|---:|
| 7  | 0.95 | 1.00 |
| 13 | 0.65 | 0.70 |
| 19 | 0.30 | 0.75 |
| 25 | 0.00 | 0.50 |
| 29 | 0.00 | 0.50 |

Hebb's collapse tracks the classical bounds reasonably well
(`ags_1985_bound(100) = 13.8`, `hopfield_1982_estimate(100) ≈ 5.4`); the
perceptron rule keeps recovering about half of corrupted patterns even
well beyond that range. It does **not** reach the textbook Gardner bound
of ~2N here — see "A documented trade-off" below for why, and treat
these numbers as this package's own measurement, not a re-derivation of
the published theoretical bound.

## Diagnostics for known limitations

```python
from hopfieldkit.diagnostics import classify_recall_outcome, detect_order_dependence

# Is a recalled state one of the patterns actually stored, or a
# spurious attractor (negative of a pattern, or an odd mixture)?
classify_recall_outcome(recalled_state, stored_patterns)

# Does the SAME starting state converge differently under sequential vs
# random update order? (A real, documented property of asynchronous
# Hopfield dynamics -- not a bug.)
detect_order_dependence(net, starting_state, n_random_trials=30)
```

## A documented trade-off

The textbook Gardner (1988) / Diederich & Opper (1987) capacity bound
(~2N) is derived for training each unit's incoming weights
*independently*, which generally produces an **asymmetric** weight
matrix. `PerceptronHopfieldNetwork` instead enforces a symmetric matrix
throughout training, so that recall stays inside the same
energy-function argument (`hopfieldkit.base`) used to guarantee
convergence for the Hebbian network. This is a deliberate,
documented choice — not a literal reproduction of the unconstrained
per-neuron algorithm — and it costs some capacity relative to the
textbook bound, as the measurements above show.

## What this is not

A pedagogical and research-utility package, not a competitor to modern
deep-learning libraries: no GPU support, no continuous-state variants,
no attention-based "modern Hopfield networks." It does one thing —
classical, binary-state Hopfield associative memory, two ways to train
it, and honest diagnostics of where it breaks.

## References

- Hopfield, J. J. (1982). "Neural networks and physical systems with
  emergent collective computational abilities." *PNAS*, 79(8), 2554-2558.
- Amit, D. J., Gutfreund, H., & Sompolinsky, H. (1985). "Storing infinite
  numbers of patterns in a spin-glass model of neural networks."
  *Physical Review Letters*, 55(14), 1530-1533.
- Gardner, E. (1988). "The space of interactions in neural network
  models." *Journal of Physics A*, 21(1), 257-270.
- Diederich, S., & Opper, M. (1987). "Learning of correlated patterns in
  spin-glass networks by local learning rules." *Physical Review
  Letters*, 58(9), 949-952.

## License

MIT.
