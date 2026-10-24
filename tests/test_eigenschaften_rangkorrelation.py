"""Eigenschaftstests (metamorphe Relationen, ORACLE-PLAYBOOK §5): was bei Umformungen der Daten gleich bleiben oder sich vorhersagbar ändern muss."""

import numpy as np
import pytest

import prs_algorithm as PA
import rkr_algorithm as R
from prs_scenario import VRng, draw_pairs


def _pair(seed, n=40, family="t5"):
    x, y = draw_pairs(family, 1, n, 0.4, VRng(seed))
    return x[0], y[0]


@pytest.mark.parametrize("seed", range(12))
def test_rank_measures_are_invariant_under_monotone_maps_of_each_variable(seed):
    x, y = _pair(seed, family="normal")
    rs, tau = R.spearman_r(x, y), R.kendall_tau(x, y)
    assert R.spearman_r(np.exp(x), y**3) == pytest.approx(rs, abs=1e-12) and R.kendall_tau(np.exp(x), y**3) == pytest.approx(tau, abs=1e-12)
    assert R.spearman_r(3 * x + 7, 0.2 * y - 4) == pytest.approx(rs, abs=1e-12)
    assert PA.pearson_r(np.exp(2 * x), y) != pytest.approx(PA.pearson_r(x, y), abs=1e-3)


@pytest.mark.parametrize("seed", range(12))
def test_negating_one_variable_flips_the_sign_and_swapping_keeps_the_value(seed):
    x, y = _pair(seed)
    for f in (R.spearman_r, R.kendall_tau):
        assert f(-x, y) == pytest.approx(-f(x, y), abs=1e-12) and f(y, x) == pytest.approx(f(x, y), abs=1e-12)
    assert abs(R.spearman_r(x, y)) <= 1 and abs(R.kendall_tau(x, y)) <= 1


@pytest.mark.parametrize("seed", range(8))
def test_kendall_and_spearman_have_the_same_sign_and_the_known_order_for_normal_data(seed):
    x, y = _pair(seed, n=200, family="normal")
    rs, tau = R.spearman_r(x, y), R.kendall_tau(x, y)
    assert np.sign(rs) == np.sign(tau) and abs(tau) < abs(rs)                                         # bei Normalverteilung ist |τ| deutlich kleiner als |ρ_S|


@pytest.mark.parametrize("seed", range(8))
def test_a_far_lever_point_moves_the_ranks_at_most_by_a_bounded_amount(seed):
    x, y = _pair(seed, n=30, family="normal")
    base = R.spearman_r(x, y)
    x2, y2 = np.append(x, 1e6), np.append(y, 1e6)
    x3, y3 = np.append(x, 1e9), np.append(y, 1e9)
    assert R.spearman_r(x2, y2) == pytest.approx(R.spearman_r(x3, y3), abs=1e-12)          # wie weit der Punkt liegt, spielt für Ränge keine Rolle
    assert PA.pearson_r(x3, y3) > 0.99                                                               # Pearson folgt dem Punkt
    assert abs(R.spearman_r(x2, y2) - base) < 0.2


@pytest.mark.parametrize("seed", range(6))
def test_ties_make_tau_a_shrink_and_gamma_grow_in_magnitude(seed):
    x, y = _pair(seed, n=300, family="normal")
    xc, yc = np.round(x), np.round(y)                       # grobe Skala mit vielen Gleichständen
    assert abs(R.kendall_tau(xc, yc, "a")) < abs(R.kendall_tau(xc, yc, "b")) < abs(R.kendall_tau(xc, yc, "gamma"))


def test_conversions_are_monotone_and_odd():
    taus = np.linspace(-0.95, 0.95, 39)
    vals = [R.rho_from_tau(t) for t in taus]
    assert vals == sorted(vals) and R.rho_from_tau(0.0) == 0.0 and R.rho_from_tau(-0.3) == pytest.approx(-R.rho_from_tau(0.3))
    assert R.rho_from_spearman(-0.4) == pytest.approx(-R.rho_from_spearman(0.4))


def test_bonett_wright_interval_mirrors_under_sign_flip_and_shrinks_with_n():
    lo, hi = R.bonett_wright_interval(0.4, 50)
    lo2, hi2 = R.bonett_wright_interval(-0.4, 50)
    assert (lo2, hi2) == pytest.approx((-hi, -lo), abs=1e-12)
    widths = [R.bonett_wright_interval(0.4, n)[1] - R.bonett_wright_interval(0.4, n)[0] for n in (10, 30, 100, 1000)]
    assert widths == sorted(widths, reverse=True)
