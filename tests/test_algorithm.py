"""Kern gegen Handrechnung (DEMO-PLAYBOOK §1b): jede Einheit an der Mini-Instanz, dazu Randfälle und Zweige."""

import math

import numpy as np
import pytest

import prs_algorithm as PA
import rkr_algorithm as R
from prs_scenario import VRng


def test_rank_rows_by_hand():
    assert list(R.rank_rows([10.0, 20.0, 20.0, 5.0])[0]) == [2.0, 3.5, 3.5, 1.0]
    assert list(R.rank_rows([3.0, 3.0, 3.0])[0]) == [2.0, 2.0, 2.0]
    assert list(R.rank_rows([[3.0, 1.0, 2.0], [1.0, 1.0, 2.0]]).ravel()) == [3.0, 1.0, 2.0, 1.5, 1.5, 3.0]
    assert list(R.rank_rows([7.0])[0]) == [1.0]


def test_ranks_sum_to_n_times_n_plus_one_over_two_even_with_ties():
    x = np.array([[5.0, 5.0, 1.0, 9.0, 9.0, 9.0, 2.0]])
    assert R.rank_rows(x).sum() == 7 * 8 / 2


def test_spearman_by_hand(four, tied):
    assert R.spearman_r(*four) == pytest.approx(0.8, abs=1e-15)
    assert R.spearman_r(*tied) == pytest.approx(5 / 6, abs=1e-15)


def test_spearman_is_one_for_any_increasing_map():
    x = np.linspace(-2, 2, 21)
    assert R.spearman_r(x, np.exp(3 * x)) == pytest.approx(1.0, abs=1e-15)
    assert R.spearman_r(x, -x**3) == pytest.approx(-1.0, abs=1e-15)


def test_spearman_rejects_constant_input():
    with pytest.raises(ValueError, match="konstant"):
        R.spearman_r([1.0, 1.0, 1.0], [1.0, 2.0, 3.0])
    with pytest.raises(ValueError, match="gleich lang"):
        R.spearman_r([1.0, 2.0], [1.0, 2.0, 3.0])


def test_kendall_counts_by_hand(four, tied):
    assert R.kendall_counts(*four) == {"C": 5, "D": 1, "Tx": 0, "Ty": 0, "n0": 6, "n": 4, "mx": 4, "my": 4}
    k = R.kendall_counts(*tied)
    assert (k["C"], k["D"], k["Tx"], k["Ty"], k["n0"], k["mx"], k["my"]) == (4, 0, 1, 1, 6, 3, 3)


def test_kendall_variants_by_hand(four, tied):
    assert R.kendall_tau(*four, "b") == pytest.approx(2 / 3) and R.kendall_tau(*four, "a") == pytest.approx(2 / 3)
    assert R.kendall_tau(*four, "gamma") == pytest.approx(4 / 6)
    assert R.kendall_tau(*tied, "a") == pytest.approx(4 / 6)
    assert R.kendall_tau(*tied, "b") == pytest.approx(0.8)
    assert R.kendall_tau(*tied, "c") == pytest.approx(0.75)
    assert R.kendall_tau(*tied, "gamma") == pytest.approx(1.0)


def test_kendall_errors():
    with pytest.raises(ValueError, match="konstant"):
        R.kendall_tau([1.0, 1.0, 1.0], [1.0, 2.0, 3.0], "b")
    with pytest.raises(ValueError, match="konstant"):
        R.kendall_tau([1.0, 1.0, 1.0], [1.0, 2.0, 3.0], "c")
    with pytest.raises(ValueError, match="kein Paar"):
        R.kendall_tau([1.0, 1.0], [1.0, 1.0], "gamma")
    with pytest.raises(ValueError, match="unbekannte Variante"):
        R.kendall_tau([1.0, 2.0], [1.0, 2.0], "d")
    with pytest.raises(ValueError, match="mindestens zwei"):
        R.kendall_counts([1.0], [1.0])


def test_kendall_two_points_is_plus_or_minus_one():
    assert R.kendall_tau([0.0, 1.0], [0.0, 5.0]) == 1.0 and R.kendall_tau([0.0, 1.0], [5.0, 0.0]) == -1.0


def test_kendall_rows_match_the_single_version(four, tied):
    X = np.array([four[0], tied[0], four[0][::-1]])
    Y = np.array([four[1], tied[1], four[1]])
    for variant in ("a", "b", "gamma"):
        rows = R.kendall_rows(X, Y, variant)
        assert list(rows) == pytest.approx([R.kendall_tau(X[i], Y[i], variant) for i in range(3)])


def test_kendall_rows_give_nan_for_constant_rows_and_reject_unknown_variants():
    assert np.isnan(R.kendall_rows(np.ones((1, 4)), np.arange(4.0).reshape(1, 4), "b")[0])
    with pytest.raises(ValueError):
        R.kendall_rows(np.ones((1, 4)), np.ones((1, 4)), "d")


def test_spearman_rows_match_the_single_version(four, tied):
    X = np.array([four[0], tied[0]])
    Y = np.array([four[1], tied[1]])
    assert list(R.spearman_rows(X, Y)) == pytest.approx([0.8, 5 / 6])
    assert np.isnan(R.spearman_rows(np.ones((1, 4)), np.arange(4.0).reshape(1, 4))[0])


def test_tests_by_hand():
    # n = 3, ρ_S = 0,5: t = 0,5·√(1/0,75), df = 1 (Cauchy): p = 1 − (2/π)·atan(t) = 2/3
    assert R.spearman_t_p(0.5, 3) == pytest.approx(2 / 3, abs=1e-12)
    assert R.spearman_t_p(1.0, 10) == 0.0 and R.spearman_t_p(-1.0, 10) == 0.0
    with pytest.raises(ValueError, match="mindestens drei"):
        R.spearman_t_p(0.5, 2)
    # τ = 1/3, n = 10: z = (1/3)·√(9·10·9/(2·25)) = (1/3)·√16,2 = 1,3416; p = 2(1 − Φ(1,3416)) = 0,1797
    assert R.kendall_z(1 / 3, 10) == pytest.approx(math.sqrt(16.2) / 3)
    assert R.kendall_z_p(1 / 3, 10) == pytest.approx(0.1797, abs=2e-4)
    assert R.kendall_z_p(-1 / 3, 10) == R.kendall_z_p(1 / 3, 10) and R.kendall_z_p(0.0, 10) == 1.0
    with pytest.raises(ValueError):
        R.kendall_z(0.1, 1)


def test_rejects_rows_agree_with_the_p_values():
    for n in (10, 30, 100):
        rs = np.array([0.0, 0.2, 0.4, -0.5, 0.9])
        assert list(R.rejects_spearman_rows(rs, n)) == [R.spearman_t_p(float(v), n) < 0.05 for v in rs]
        assert list(R.rejects_kendall_rows(rs, n)) == [R.kendall_z_p(float(v), n) < 0.05 for v in rs]


def test_permutation_p_extremes():
    x = np.arange(12.0)
    for measure in ("spearman", "kendall"):
        assert R.permutation_p_rank(x, 3 * x + 1, 199, VRng(3), measure) == 1 / 200
        assert R.permutation_p_rank(x, -x, 199, VRng(3), measure) == 1 / 200
    with pytest.raises(ValueError, match="unbekanntes Maß"):
        R.permutation_p_rank(x, x, 10, VRng(1), "pearson")


def test_permutation_p_is_one_when_the_statistic_is_zero():
    """x = (−1, 0, 1), y = (1, −2, 1): ρ_S ist 0 (Ränge (1, 2, 3) und (1,5; 3; 1,5): Σdxdy = 0), jede Permutation erreicht |T*| ≥ 0."""
    x, y = np.array([-1.0, 0.0, 1.0]), np.array([1.0, -2.0, 1.0])
    assert R.spearman_r(x, y) == pytest.approx(0.0, abs=1e-15)
    assert R.permutation_p_rank(x, y, 49, VRng(1), "spearman") == 1.0


def test_bonett_wright_by_hand():
    lo, hi = R.bonett_wright_interval(0.5, 103)       # se = √((1 + 0,125)/100) = 0,106066; atanh 0,5 = 0,549306
    z = 1.959963984540054 * math.sqrt(1.125 / 100)
    assert (lo, hi) == pytest.approx((math.tanh(0.5493061443340549 - z), math.tanh(0.5493061443340549 + z)), abs=1e-12)
    naive = R.naive_spearman_interval(0.5, 103)
    assert naive[0] > lo and naive[1] < hi                                     # der Pearson-Standardfehler ist kleiner: engeres Intervall
    with pytest.raises(ValueError):
        R.bonett_wright_interval(0.5, 3)


def test_bonett_wright_survives_one():
    lo, hi = R.bonett_wright_interval(1.0, 20)
    assert hi <= 1.0 and lo > 0.9


def test_rank_bootstrap_shapes_and_range():
    x, y = np.arange(12.0), np.array([3.0, 1, 2, 5, 4, 8, 7, 6, 10, 9, 12, 11])
    rs, tau = R.rank_bootstrap(x, y, 100, VRng(1))
    assert rs.shape == tau.shape == (100,)
    assert np.nanmax(np.abs(rs)) <= 1.0 and np.nanmax(np.abs(tau)) <= 1.0 and np.nanmedian(rs) > 0.7


def test_formulas_and_conversions_by_hand():
    assert R.tau_formula(0.0) == 0.0 and R.tau_formula(1.0) == pytest.approx(1.0) and R.tau_formula(0.5) == pytest.approx(1 / 3)
    assert R.spearman_formula(1.0) == pytest.approx(1.0) and R.spearman_formula(0.0) == 0.0
    assert R.spearman_formula(0.5) == pytest.approx(6 / math.pi * math.asin(0.25))
    assert R.rho_from_tau(1 / 3) == pytest.approx(0.5) and R.rho_from_tau(2 / 3) == pytest.approx(math.sqrt(3) / 2)
    assert R.rho_from_spearman(R.spearman_formula(0.7)) == pytest.approx(0.7)
    assert R.rho_from_tau(R.tau_formula(-0.4)) == pytest.approx(-0.4)


def test_true_values_by_family():
    assert R.tau_true("normal", 0.5) == pytest.approx(1 / 3) and R.tau_true("t3", 0.5) == pytest.approx(1 / 3) and R.tau_true("tindep3", 0.9) == 0.0
    assert R.spearman_true("lognormal1", 0.5) == pytest.approx(R.spearman_formula(0.5)) and R.spearman_true("normal", 0.5) == R.spearman_formula(0.5)
    assert R.spearman_true("t3", 0.5) is None and R.spearman_true("kontaminiert", 0.5) is None and R.spearman_true("tindep3", 0.5) == 0.0


def test_influence_rank_by_hand(four):
    # ohne Punkt 1: (2,3), (3,2), (4,4) → Ränge (1,2,3) und (2,1,3): ρ_S = 1 − 6·2/(3·8) = 0,5; ohne Punkt 2: (1,1), (3,2), (4,4) → ρ_S = 1
    infl = R.influence_rank(*four)
    assert infl == pytest.approx([0.8 - 0.5, 0.8 - 1.0, 0.8 - 1.0, 0.8 - 0.5])


def test_pearson_helpers_come_from_the_copy():
    assert PA.pearson_r([1, 2, 3], [1, 2, 3]) == pytest.approx(1.0)


def test_kendall_rows_variant_c_matches_the_single_version(tied):
    X = np.array([tied[0], [1.0, 2.0, 3.0, 4.0, 5.0, 6.0][:4]])
    Y = np.array([tied[1], [2.0, 1.0, 4.0, 3.0]])
    assert list(R.kendall_rows(X, Y, "c")) == pytest.approx([R.kendall_tau(X[0], Y[0], "c"), R.kendall_tau(X[1], Y[1], "c")])
    assert np.isnan(R.kendall_rows(np.ones((1, 4)), np.arange(4.0).reshape(1, 4), "c")[0])


def test_tau_c_works_for_binary_scales():
    """Zwei Stufen je Kennzahl (m = 2): τ-c ist definiert, bei 2 × 2 Tabellen mit a, b, c, d: 2(ad − bc)/(n²/2) = 4(ad − bc)/n²."""
    x = np.array([0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0])
    y = np.array([0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0])       # Tabelle: 00: 2, 01: 1, 10: 1, 11: 4 → ad − bc = 8 − 1 = 7
    assert R.kendall_tau(x, y, "c") == pytest.approx(4 * 7 / 64)
    assert R.kendall_rows(x[None], y[None], "c")[0] == pytest.approx(4 * 7 / 64)
    with pytest.raises(ValueError, match="konstant"):
        R.kendall_tau(np.ones(5), np.arange(5.0), "c")


@pytest.mark.parametrize("n", [10, 30, 100])
def test_kendall_rejection_boundary(n):
    crit = PA.norm_ppf(0.975) / math.sqrt(9.0 * n * (n - 1) / (2.0 * (2.0 * n + 5.0)))
    assert list(R.rejects_kendall_rows([crit - 1e-6, crit + 1e-6, -(crit + 1e-6), -(crit - 1e-6)], n)) == [False, True, True, False]
    assert list(R.rejects_kendall_rows([crit - 1e-6, crit + 1e-6], n, alpha=0.2)) == [True, True]        # strengeres Niveau 0,2: die Schwelle liegt tiefer
    assert R.kendall_z_p(crit, n) == pytest.approx(0.05, abs=1e-9)


@pytest.mark.parametrize("n", [5, 10, 30, 100])
def test_spearman_rejection_boundary(n):
    t = PA.t_critical(n - 2)
    crit = t / math.sqrt(t * t + n - 2)
    assert list(R.rejects_spearman_rows([crit - 1e-6, crit + 1e-6, -(crit + 1e-6), -(crit - 1e-6)], n)) == [False, True, True, False]
    assert R.spearman_t_p(crit, n) == pytest.approx(0.05, abs=1e-8)
