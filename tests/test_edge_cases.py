"""Randfälle: kleinstes und größtes n, Hebelpunkte am Rand, extreme ρ, jede Familie, entartete Eingaben (DEMO-PLAYBOOK §1d)."""

import numpy as np
import pytest

import rkr_algorithm as R
import rkr_constants as C
import rkr_evaluation as ev


@pytest.mark.parametrize("family", list(C.FAMILIES))
@pytest.mark.parametrize("n", [C.N_MIN, 6, C.N_MAX])
def test_analyse_works_for_every_family_at_the_smallest_and_largest_n(family, n):
    a = ev.analyse(ev.Settings(family=family, n=n, boot_b=C.B_MIN))
    assert -1.0 <= a.r <= 1.0 and -1.0 <= a.rho_s <= 1.0 and -1.0 <= a.tau <= 1.0
    assert all(lo <= hi for lo, hi in a.intervals.values()) and len(a.infl_r) == len(a.infl_s) == n


@pytest.mark.parametrize("rho", [C.RHO_MIN, -0.5, 0.0, C.RHO_MAX])
def test_extreme_rho(rho):
    a = ev.analyse(ev.Settings(rho=rho, n=60))
    assert a.targets["ρ aus Kendall (Bootstrap)"] == rho
    if abs(rho) >= 0.5:
        assert np.sign(a.rho_s) == np.sign(rho) and np.sign(a.tau) == np.sign(rho)


@pytest.mark.parametrize("count,k", [(0, 0.0), (1, 0.0), (2, C.LEV_K_MAX), (1, C.LEV_K_MAX)])
@pytest.mark.parametrize("n", [C.N_MIN, 30])
def test_leverage_at_the_edges(count, k, n):
    a = ev.analyse(ev.Settings(n=n, lev_count=count, lev_k=k, lev_dir="gegenläufig"))
    assert -1.0 <= a.rho_s <= 1.0 and 0.0 <= a.p_spearman <= 1.0 and 0.0 <= a.p_kendall <= 1.0


def test_all_sliders_at_their_maxima_and_minima_together():
    low = ev.Settings(family="t3", rho=C.RHO_MIN, n=C.N_MIN, lev_count=C.LEV_COUNT_MAX, lev_k=C.LEV_K_MAX, boot_b=C.B_MIN, seed=0)
    high = ev.Settings(family="lognormal1", rho=C.RHO_MAX, n=C.N_MAX, lev_count=C.LEV_COUNT_MAX, lev_k=C.LEV_K_MAX, boot_b=C.B_MAX, seed=C.SEED_MAX)
    for s in (low, high):
        a = ev.analyse(s)
        assert -1.0 <= a.rho_s <= 1.0


def test_worst_case_at_the_slider_maximum_finishes():
    """n = 500, B = 1 000: 2,5·10^8 Vorzeichenprodukte; ein Datensatz rechnet durch (Timeout statt Zeit-Assertion, DEMO-PLAYBOOK §1c)."""
    a = ev.analyse(ev.Settings(n=C.N_MAX, boot_b=C.B_MAX, family="t3"))
    assert a.intervals["Kendall (Bootstrap)"][0] < a.intervals["Kendall (Bootstrap)"][1]


def test_perfect_monotone_data():
    x = np.arange(10.0)
    assert R.spearman_r(x, x**3) == pytest.approx(1.0) and R.kendall_tau(x, x**3) == pytest.approx(1.0)
    lo, hi = R.bonett_wright_interval(1.0, 10)
    assert lo <= hi <= 1.0 and R.spearman_t_p(1.0, 10) == 0.0


def test_four_points_is_the_smallest_interval_case(four):
    lo, hi = R.bonett_wright_interval(R.spearman_r(*four), 4)
    assert lo < 0.8 < hi
    with pytest.raises(ValueError):
        R.bonett_wright_interval(0.5, 3)


def test_two_points_for_the_rank_functions():
    assert R.spearman_r([0.0, 1.0], [0.0, 5.0]) == pytest.approx(1.0) and R.kendall_tau([0.0, 1.0], [5.0, 0.0], "a") == -1.0


def test_every_family_has_a_label_and_a_tau():
    assert set(C.FAMILIES) <= set(C.ALL_FAMILIES)
    for fam in C.ALL_FAMILIES:
        assert -1.0 <= R.tau_true(fam, 0.3) <= 1.0
