import numpy as np
import pytest

from hopfieldkit.capacity import ags_1985_bound, empirical_capacity_scan, hopfield_1982_estimate


def test_hopfield_1982_estimate_matches_formula():
    n = 100
    assert hopfield_1982_estimate(n) == pytest.approx(n / (4 * np.log(n)))


def test_hopfield_1982_estimate_raises_for_n_le_1():
    with pytest.raises(ValueError):
        hopfield_1982_estimate(1)


def test_ags_1985_bound_matches_formula():
    n = 100
    assert ags_1985_bound(n) == pytest.approx(0.138 * n)


def test_empirical_capacity_scan_output_shape_and_bounds():
    result = empirical_capacity_scan(n_units=30, pattern_counts=[1, 5, 10], n_trials=5, seed=0)
    assert result["pattern_counts"] == [1, 5, 10]
    assert len(result["success_rates"]) == 3
    assert all(0.0 <= rate <= 1.0 for rate in result["success_rates"])


def test_empirical_capacity_scan_declines_as_patterns_increase():
    result = empirical_capacity_scan(
        n_units=50, pattern_counts=[1, 25], n_trials=20, flip_fraction=0.15, seed=1
    )
    low_p_rate, high_p_rate = result["success_rates"]
    assert low_p_rate > high_p_rate
