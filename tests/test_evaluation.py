"""Auswertung: Einzeldatensatz, Messreihen, Zellen des Rasters. Toleranzen aus dem Monte-Carlo-Fehler, keine exakten Zufallswerte (DEMO-PLAYBOOK §4)."""

import math

import numpy as np
import pytest

import prs_algorithm as PA
import rkr_algorithm as R
import rkr_constants as C
import rkr_evaluation as ev


def test_analyse_is_consistent():
    a = ev.analyse(ev.Settings())
    x, y = np.array(a.x), np.array(a.y)
    assert a.r == pytest.approx(PA.pearson_r(x, y)) and a.rho_s == pytest.approx(R.spearman_r(x, y)) and a.tau == pytest.approx(R.kendall_tau(x, y, "b"))
    assert a.infl_r == pytest.approx(tuple(PA.influence(x, y))) and a.infl_s == pytest.approx(tuple(R.influence_rank(x, y)))
    assert set(a.intervals) == set(ev.INTERVALS) == set(a.covers) == set(a.targets)
    for name, (lo, hi) in a.intervals.items():
        assert lo < hi
        assert a.covers[name] == (None if a.targets[name] is None else lo <= a.targets[name] <= hi)
    assert a.targets["Kendall (Bootstrap)"] == pytest.approx(1 / 3) and a.tau_true == pytest.approx(1 / 3)
    assert a.p_pearson == pytest.approx(PA.t_test_p(a.r, 30)) and a.p_spearman == pytest.approx(R.spearman_t_p(a.rho_s, 30)) and a.p_kendall == pytest.approx(R.kendall_z_p(a.tau, 30))


def test_analyse_targets_by_family():
    normal = ev.analyse(ev.Settings(family="normal", rho=0.5))
    assert normal.rs_true == pytest.approx(R.spearman_formula(0.5)) and normal.true_r == 0.5
    t3 = ev.analyse(ev.Settings(family="t3", rho=0.5))
    assert t3.rs_true is None and t3.covers["Spearman (Bonett–Wright)"] is None and t3.covers["Spearman (Bootstrap)"] is None
    assert t3.covers["Kendall (Bootstrap)"] is not None and t3.targets["ρ aus Kendall (Bootstrap)"] == 0.5
    logn = ev.analyse(ev.Settings(family="lognormal1", rho=0.5))
    assert logn.true_r == pytest.approx(0.3775, abs=1e-3) and logn.rs_true == pytest.approx(R.spearman_formula(0.5)) and logn.targets["ρ aus Kendall (Bootstrap)"] == 0.5


def test_analyse_is_deterministic_and_seed_dependent():
    s = ev.Settings(n=40, seed=3)
    assert ev.analyse(s) == ev.analyse(s) and ev.analyse(s) != ev.analyse(ev.Settings(n=40, seed=4))


def test_analyse_with_leverage_point_changes_pearson_much_more_than_the_ranks():
    a0 = ev.analyse(ev.Settings(rho=0.0, n=30))
    a1 = ev.analyse(ev.Settings(rho=0.0, n=30, lev_count=1, lev_k=8.0))
    assert a1.lever == (0,) and a1.r - a0.r > 0.4 and abs(a1.rho_s - a0.rho_s) < 0.25 and a1.infl_r[0] > 0.4 > abs(a1.infl_s[0])


def test_affordable_boot_respects_the_budget():
    assert ev.affordable_boot(1000, 30) == 1000 and ev.affordable_boot(1000, 500) == 1000
    assert ev.affordable_boot(1000, 1000) == 300 and ev.affordable_boot(1000, 10_000) == 50           # mindestens 50
    assert ev.affordable_boot(100, 30) == 100


def test_curve_experiment_matches_the_documented_population_value_and_ranks():
    res = ev.curve_experiment(2, 0.0, 100, 100, 1)
    assert res["rho_s"] == pytest.approx(1.0, abs=1e-12) and res["tau"] == pytest.approx(1.0, abs=1e-12) and res["population_r"] == pytest.approx(0.2732, abs=1e-4)
    assert 0.3 < res["r"] < 0.8 and res["runs"] == 100
    assert ev.curve_experiment(2, 0.5, 100, 20, 1)["population_r"] is None


def test_curve_r_falls_with_n():
    small, large = ev.curve_experiment(3, 0.0, 20, 100, 2), ev.curve_experiment(3, 0.0, 400, 50, 2)
    assert small["r"] > large["r"] + 0.15 and large["rho_s"] == pytest.approx(1.0)


def test_affordable_curve_runs_and_population_formula():
    assert ev.affordable_curve_runs(100, 200) == 200 and ev.affordable_curve_runs(2000, 200) == 50 and ev.affordable_curve_runs(100_000, 200) == 5
    assert ev.curve_population_r(1) == pytest.approx(1 / math.sqrt(math.e - 1)) and ev.curve_population_r(3) == pytest.approx(3 / math.sqrt(math.exp(9) - 1))


def test_relations_cell_matches_the_formulas_small():
    row = ev.relations_cell("normal", 0.6, runs=6, n=600)
    assert row["tau"] == pytest.approx(R.tau_formula(0.6), abs=0.03) and row["rho_s"] == pytest.approx(R.spearman_formula(0.6), abs=0.03)
    assert row["rho_from_tau"] == pytest.approx(0.6, abs=0.04) and row["true_r"] == 0.6
    assert ev.relations_cell("lognormal1", 0.6, runs=4, n=400)["true_r"] == pytest.approx(0.4785, abs=1e-3)


def test_lever_cell_ranks_barely_move_and_pearson_does():
    cell = ev.lever_cell(30, 8, seeds=(101,), runs_per_seed=400)
    assert cell["share_r"] > 0.95 and cell["share_s"] < 0.12 and cell["share_k"] < 0.12 and cell["runs"] == 400 and cell["mean_r"] > cell["mean_s"] + 0.3
    base = ev.lever_cell(30, 0, seeds=(101,), runs_per_seed=800)
    assert abs(base["share_r"] - 0.05) < 0.03 and abs(base["share_s"] - 0.05) < 0.03


def test_size_cell_common_scale_hits_pearson_but_hardly_the_rank_tests():
    cell = ev.size_cell("t3", 100, seeds=(101, 102), runs_per_seed=600)
    assert cell["pearson"] > 0.2 and cell["spearman"] < 0.12 and cell["kendall"] < 0.12 and cell["runs"] == 1200
    indep = ev.size_cell("tindep3", 100, seeds=(101, 102), runs_per_seed=600)
    assert abs(indep["pearson"] - 0.05) < 0.03


def test_bin_rows_makes_equal_sized_steps():
    X = np.arange(60.0).reshape(1, 60)
    assert list(np.bincount(ev.bin_rows(X, 3)[0].astype(int))) == [20, 20, 20]
    assert len(np.unique(ev.bin_rows(X, 100))) == 60                                 # feine Skala: jeder Wert behält seine eigene Stufe


def test_ties_cell_variants_differ_on_a_coarse_scale():
    row = ev.ties_cell(3, seeds=(101,), runs_per_seed=30, n=300)
    assert row["tau_a"] < row["tau_b"] < row["gamma"] and row["tau_b"] == pytest.approx(row["tau_c"], abs=1e-9)
    fine = ev.ties_cell(100, seeds=(101,), runs_per_seed=30, n=300)
    assert abs(fine["tau_a"] - fine["tau_b"]) < 0.02 and fine["rho_s"] == pytest.approx(fine["pearson_codes"], abs=0.01)


def test_cover_runs_are_close_to_95_percent_on_normal_data():
    res = ev.cover_runs("normal", 50, 0.5, 120, 100, 3)
    se = math.sqrt(0.95 * 0.05 / 120)
    assert abs(res["s_bw"] - 0.95) < 4 * se and abs(res["k_boot"] - 0.95) < 5 * se and res["runs"] == 120
    assert res["w_s_bw"] > 0 and res["w_k_boot"] > 0 and res["w_rho_k"] > 0


def test_cover_runs_on_heavy_tails_separate_kendall_from_pearson():
    res = ev.cover_runs("t3", 200, 0.5, 100, 100, 4)
    assert res["pearson"] < 0.8 and res["k_boot"] > res["pearson"] + 0.12
    assert res["s_bw"] == 0.0 and res["s_boot"] == 0.0                              # kein bekanntes Spearman-Ziel: nie als Treffer gezählt


def test_cover_cell_small():
    row = ev.cover_cell("normal", 30, 0.5, seeds=(101, 102), runs_per_seed=30, boot_b=60)
    assert row["runs"] == 60 and row["has_spearman_target"] is True and ev.cover_cell("t3", 30, 0.5, seeds=(101,), runs_per_seed=5, boot_b=50)["has_spearman_target"] is False
    for key in ("s_naive", "s_bw", "s_boot", "k_boot", "rho_k", "pearson"):
        assert 0.0 <= row[key] <= 1.0 and row[key + "_sd"] >= 0


def test_eff_cell_kendall_conversion_beats_pearson_on_heavy_tails():
    heavy = ev.eff_cell("t3", 0.6, seeds=(101,), runs_per_seed=400)
    assert heavy["k_sd"] < 0.7 * heavy["r_sd"] and heavy["k_mean"] == pytest.approx(0.6, abs=0.03) and heavy["s_mean"] < heavy["k_mean"]
    normal = ev.eff_cell("normal", 0.6, seeds=(101,), runs_per_seed=400)
    assert normal["k_sd"] > normal["r_sd"] and normal["k_mean"] == pytest.approx(0.6, abs=0.03)


def test_cell_seeds_are_distinct_across_cells():
    seeds = {ev.cell_seed(kind, f, n, rho, s) for kind in ("rel", "lever", "size", "ties", "cover", "eff") for f in ("normal", "t3") for n in (10, 100) for rho in (0.0, 0.5) for s in C.GRID_SEEDS}
    assert len(seeds) == 6 * 2 * 2 * 2 * 4


def test_grid_probe_is_a_small_deterministic_cell():
    assert ev.grid_probe() == ev.grid_probe() and ev.grid_probe()["runs"] == 8


def test_load_grid_handles_a_missing_or_broken_file(tmp_path):
    assert ev.load_grid(tmp_path / "gibt-es-nicht.json") is None
    bad = tmp_path / "kaputt.json"
    bad.write_text("{", encoding="utf-8")
    assert ev.load_grid(bad) is None


def test_find_and_cover_series_on_a_hand_built_grid():
    rows = [{"family": "t3", "n": 30, "rho": 0.5, "k_boot": 0.94}, {"family": "t3", "n": 10, "rho": 0.5, "k_boot": 0.93}, {"family": "t3", "n": 10, "rho": 0.0, "k_boot": 0.9}]
    assert ev.find(rows, family="t3", n=30, rho=0.5)["k_boot"] == 0.94 and ev.find(rows, family="t3", n=30, rho=0.9) is None
    assert [r["n"] for r in ev.cover_series({"cover": rows}, "t3", 0.5)] == [10, 30]
