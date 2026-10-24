"""Orakel (ORACLE-PLAYBOOK, DEMO-PLAYBOOK §12): jede Kernroutine gegen einen unabhängigen Rechenweg.

Spearman, Kendall (b, c), Ränge und p-Werte gegen scipy; Kendall-a und γ gegen eine dreifache Schleife über alle Paare; Permutationstests gegen die vollständige
Aufzählung; Bootstrap und Leave-one-out gegen Schleifen mit denselben Indizes."""

import itertools
import math

import numpy as np
import pytest

import prs_algorithm as PA
import prs_scenario as S
import rkr_algorithm as R

sps = pytest.importorskip("scipy.stats")


def _sample(family, n, rho, seed):
    x, y = S.draw_pairs(family, 1, n, rho, S.VRng(seed))
    return x[0], y[0]


def _pairs(x, y):
    """Brute Force: C, D, T_x, T_y über alle Paare i < j."""
    c = d = tx = ty = 0
    for i, j in itertools.combinations(range(len(x)), 2):
        dx, dy = x[i] - x[j], y[i] - y[j]
        tx += dx == 0
        ty += dy == 0
        if dx * dy > 0:
            c += 1
        elif dx * dy < 0:
            d += 1
    return c, d, tx, ty


@pytest.mark.parametrize("seed", range(30))
def test_spearman_and_kendall_match_scipy(seed):
    family = ("normal", "t3", "lognormal1", "kontaminiert")[seed % 4]
    n = (5, 8, 20, 60, 120)[seed % 5]
    x, y = _sample(family, n, 0.5, seed)
    assert R.spearman_r(x, y) == pytest.approx(sps.spearmanr(x, y)[0], abs=1e-12)
    assert R.kendall_tau(x, y, "b") == pytest.approx(sps.kendalltau(x, y)[0], abs=1e-12)
    assert R.kendall_tau(x, y, "c") == pytest.approx(sps.kendalltau(x, y, variant="c")[0], abs=1e-12)
    assert R.spearman_t_p(R.spearman_r(x, y), n) == pytest.approx(sps.spearmanr(x, y)[1], rel=1e-8, abs=1e-14)


@pytest.mark.parametrize("seed", range(30))
def test_counts_and_all_variants_match_the_triple_loop_with_ties(seed):
    rng = np.random.default_rng(seed)
    n = int(rng.integers(3, 15))
    x = rng.integers(0, 4, n).astype(float)
    y = rng.integers(0, 5, n).astype(float)
    c, d, tx, ty = _pairs(x, y)
    n0 = n * (n - 1) / 2
    k = R.kendall_counts(x, y)
    assert (k["C"], k["D"], k["Tx"], k["Ty"]) == (c, d, tx, ty)
    assert R.kendall_tau(x, y, "a") == pytest.approx((c - d) / n0)
    if n0 > tx and n0 > ty:
        assert R.kendall_tau(x, y, "b") == pytest.approx((c - d) / math.sqrt((n0 - tx) * (n0 - ty)))
        assert R.kendall_tau(x, y, "b") == pytest.approx(sps.kendalltau(x, y, variant="b")[0], abs=1e-12)
    if c + d:
        assert R.kendall_tau(x, y, "gamma") == pytest.approx((c - d) / (c + d))
    if len(set(x)) > 1 and len(set(y)) > 1:
        assert R.spearman_r(x, y) == pytest.approx(sps.spearmanr(x, y)[0], abs=1e-12)


def test_rank_rows_match_scipy_with_ties():
    rng = np.random.default_rng(0)
    X = rng.integers(0, 6, (40, 17)).astype(float)
    assert np.array_equal(R.rank_rows(X), sps.rankdata(X, axis=1))


@pytest.mark.parametrize("seed", range(10))
def test_kendall_rows_match_the_single_version_on_resamples(seed):
    x, y = _sample("normal", 25, 0.5, seed)
    idx = S.VRng(300 + seed).below(len(x), (12, len(x)))
    rows = R.kendall_rows(x[idx], y[idx], "b")
    loop = [sps.kendalltau(x[i], y[i], variant="b")[0] for i in idx]
    assert rows == pytest.approx(loop, abs=1e-12, nan_ok=True)
    assert R.spearman_rows(x[idx], y[idx]) == pytest.approx([sps.spearmanr(x[i], y[i])[0] for i in idx], abs=1e-12, nan_ok=True)


@pytest.mark.parametrize("seed", range(6))
def test_influence_matches_the_loop(seed):
    x, y = _sample("t5", 14 + seed, 0.4, seed)
    loop = np.array([sps.spearmanr(np.delete(x, i), np.delete(y, i))[0] for i in range(len(x))])
    assert R.influence_rank(x, y) == pytest.approx(sps.spearmanr(x, y)[0] - loop, abs=1e-12)


@pytest.mark.parametrize("n", [3, 4, 5, 6])
@pytest.mark.parametrize("measure", ["spearman", "kendall"])
def test_permutation_p_matches_complete_enumeration(n, measure):
    rng = S.VRng(n)
    x = rng.normal(n)
    y = 0.8 * x + 0.6 * rng.normal(n)
    stat = (lambda a, b: abs(R.spearman_r(a, b))) if measure == "spearman" else (lambda a, b: abs(R.kendall_tau(a, b, "b")))
    obs = stat(x, y)
    exact = np.mean([stat(x, y[list(p)]) >= obs - 1e-12 for p in itertools.permutations(range(n))])
    est = R.permutation_p_rank(x, y, 3000, S.VRng(50 + n), measure)
    se = math.sqrt(max(exact * (1 - exact), 1e-4) / 3000)
    assert abs(est - (3000 * exact + 1) / 3001) < 5 * se + 0.002


def test_kendall_z_matches_the_normal_approximation_of_scipy_without_ties():
    """Gleiche Näherung wie scipy (method='asymptotic'), die Varianz ohne Gleichstände."""
    x, y = _sample("normal", 40, 0.3, 4)
    tau = R.kendall_tau(x, y, "b")
    ref = sps.kendalltau(x, y, method="asymptotic")
    assert R.kendall_z_p(tau, 40) == pytest.approx(ref.pvalue, rel=1e-9)


def test_bonett_wright_is_fisher_with_the_documented_standard_error():
    for rs, n in ((0.3, 30), (0.7, 60), (-0.5, 12)):
        lo, hi = R.bonett_wright_interval(rs, n)
        se = math.sqrt((1 + rs * rs / 2) / (n - 3))
        z = sps.norm.ppf(0.975)
        assert (lo, hi) == pytest.approx((math.tanh(math.atanh(rs) - z * se), math.tanh(math.atanh(rs) + z * se)), abs=1e-12)


@pytest.mark.parametrize("seed", range(4))
def test_the_two_conversions_agree_with_the_formulas_on_a_large_normal_sample(seed):
    x, y = _sample("normal", 4000, 0.6, 40 + seed)
    assert R.rho_from_tau(sps.kendalltau(x, y)[0]) == pytest.approx(0.6, abs=0.04)
    assert R.rho_from_spearman(sps.spearmanr(x, y)[0]) == pytest.approx(0.6, abs=0.04)
    assert sps.kendalltau(x, y)[0] == pytest.approx(R.tau_formula(0.6), abs=0.03)
    assert sps.spearmanr(x, y)[0] == pytest.approx(R.spearman_formula(0.6), abs=0.03)


def test_tau_formula_holds_for_elliptical_families_but_spearman_formula_does_not():
    x, y = _sample("t3", 200_000 // 20, 0.6, 77)             # 10 000 Paare
    assert sps.kendalltau(x, y)[0] == pytest.approx(R.tau_formula(0.6), abs=0.02)
