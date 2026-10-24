"""Verdikte an von Hand gesetzten Werten (DEMO-PLAYBOOK §12): jede Zahl im Text und jede Schwelle der Art steht hier."""

import math

import pytest

import rkr_evaluation as ev

NAMES = ev.INTERVALS


def _analysis(covers, lever=(), n=30, family="normal", rho=0.5):
    return ev.Analysis(ev.Settings(family=family, rho=rho, n=n, lev_k=5.0), (0.0,), (0.0,), lever, 0.5, 0.33, 0.48, 0.62, 0.58, 0.41, 0.01, 0.02, 0.03,
                       {k: (0.0, 1.0) for k in NAMES}, {k: 0.5 for k in NAMES}, dict(zip(NAMES, covers)), (0.0,), (0.0,))


def test_main_all_known_targets_hit():
    kind, text = ev.verdict_main(_analysis((True, True, True, True, True)))
    assert kind == "success" and "r = **0.62**, ρ_S = **0.58**, τ = **0.41** (n = 30)" in text and "ρ = **0.50**, τ = **0.33** (Normal)" in text and "Alle 5 Intervalle" in text


def test_main_unknown_targets_are_not_counted():
    kind, text = ev.verdict_main(_analysis((True, None, None, True, True)))
    assert kind == "success" and "Alle 3 Intervalle" in text
    kind, _ = ev.verdict_main(_analysis((False, None, None, False, False)))
    assert kind == "warning"
    kind, text = ev.verdict_main(_analysis((None, None, None, None, None)))
    assert kind == "info" and "Für kein Intervall ist ein Zielwert bekannt" in text


def test_main_partial_names_hitters_and_missers():
    kind, text = ev.verdict_main(_analysis((False, True, True, True, True)))
    assert kind == "info" and "Es enthalten ihr Ziel: Spearman (Bonett–Wright), Spearman (Bootstrap), Kendall (Bootstrap), ρ aus Kendall (Bootstrap);" in text
    assert "es verfehlen es: Pearson (Fisher-z)" in text


def test_main_mentions_leverage_points_and_family():
    _, text = ev.verdict_main(_analysis((True,) * 5, lever=(0, 1), family="t3"))
    assert "2 Hebelpunkt(e) bei k = 5" in text and "t(3), gemeinsame Skala" in text
    assert "Hebelpunkt" not in ev.verdict_main(_analysis((True,) * 5))[1]


def _rel(family, rho, from_tau, from_s):
    return {"family": family, "rho": rho, "rho_from_tau": from_tau, "rho_from_spearman": from_s}


def test_relations_verdict_names_the_worst_conversion_and_its_threshold():
    rows = [_rel("normal", 0.6, 0.604, 0.602), _rel("t3", 0.6, 0.601, 0.579), _rel("t3", 0.9, 0.900, 0.887)]
    kind, text = ev.verdict_relations(rows)
    assert kind == "success" and "in allen 3 Zellen höchstens um **0.004**" in text and "bis zu **0.021**" in text and "t(3), gemeinsame Skala, ρ = 0.6: 0.579" in text
    kind, _ = ev.verdict_relations([_rel("normal", 0.6, 0.62, 0.6)])
    assert kind == "info"                                               # Kendall-Fehler 0,02 ≥ 0,01
    assert ev.verdict_relations([_rel("normal", 0.6, 0.6099, 0.6)])[0] == "success"


def _curve(r, rho_s, population=None):
    return {"c": 2, "noise": 0.5, "n": 100, "runs": 200, "r": r, "r_sd": 0.08, "rho_s": rho_s, "tau": 0.78, "population_r": population}


def test_curve_verdict():
    kind, text = ev.verdict_curve(_curve(0.58, 0.91))
    assert kind == "info" and "c = 2, Rauschen 0.5, n = 100 (200 Läufe)" in text and "r im Mittel **0.58** (Streuung 0.08), ρ_S **0.91**, τ **0.78**" in text and "Populationskorrelation" not in text
    assert ev.verdict_curve(_curve(0.73, 0.69))[0] == "warning"          # Pearson liegt über dem Rangmaß (c = 0,5 mit Rauschen)
    assert ev.verdict_curve(_curve(0.80, 0.80))[0] == "info"             # gleich: kein Warnfall
    assert "Populationskorrelation ohne Rauschen wäre 0.27" in ev.verdict_curve(_curve(0.4, 1.0, 0.2732))[1]


def _lever(n, k, r, s, kd):
    return {"n": n, "k": k, "runs": 4000, "share_r": r, "share_s": s, "share_k": kd, "mean_r": 0.47, "mean_s": 0.10, "mean_tau": 0.07}


def test_lever_verdict_branches():
    rows = [_lever(30, 0, 0.051, 0.054, 0.057), _lever(30, 3, 0.220, 0.063, 0.071), _lever(30, 5, 0.833, 0.064, 0.071)]
    kind, text = ev.verdict_lever(rows, 30, 0)
    assert kind == "info" and "5.1 / 5.4 / 5.7 %" in text and "erwartet 5 %" in text and "4000 Läufe" in text
    kind, text = ev.verdict_lever(rows, 30, 3)
    assert kind == "info" and "22.0 %" in text
    kind, text = ev.verdict_lever(rows, 30, 5)
    assert kind == "warning" and "**83.3 %**" in text and "**6.4 %**" in text and "**7.1 %**" in text and "5 Standardabweichungen" in text and "r 0.47, ρ_S 0.10, τ 0.07" in text
    assert ev.verdict_lever([_lever(30, 5, 0.5, 0.06, 0.07)], 30, 5)[0] == "warning"
    assert ev.verdict_lever([_lever(30, 5, 0.4999, 0.06, 0.07)], 30, 5)[0] == "info"


def _size(p, s, k, runs=8000, family="t3", n=100):
    return {"family": family, "n": n, "runs": runs, "pearson": p, "spearman": s, "kendall": k}


def test_size_verdict_threshold_is_three_standard_errors_for_pearson():
    se = math.sqrt(0.05 * 0.95 / 8000)
    kind, text = ev.verdict_size(_size(0.322, 0.070, 0.082))
    assert kind == "warning" and "t(3), gemeinsame Skala, n = 100" in text and "**32.2 / 7.0 / 8.2 %**" in text and "von 8000 Läufen" in text
    assert ev.verdict_size(_size(0.05 + 3.1 * se, 0.05, 0.05))[0] == "warning"
    kind, text = ev.verdict_size(_size(0.05 + 2.9 * se, 0.05, 0.05))
    assert kind == "success" and "nahe 5 %" in text
    assert ev.verdict_size(_size(0.01, 0.06, 0.06))[0] == "success"              # zu wenige Alarme sind kein Warnfall


def _tie(levels, a, b, g):
    return {"levels": levels, "tau_a": a, "tau_b": b, "gamma": g}


def test_ties_verdict_names_the_spread_at_the_coarsest_scale():
    kind, text = ev.verdict_ties([_tie(3, 0.30, 0.45, 0.62), _tie(100, 0.41, 0.41, 0.42)])
    assert kind == "info" and "Bei 3 Stufen" in text and "**0.30 (τ-a), 0.45 (τ-b) und 0.62 (γ)**" in text and "Spanne von 0.32" in text
    assert "bei 100 Stufen liegen sie zwischen 0.41 und 0.42" in text


def _cov(n, k, p, rho=0.5):
    return {"n": n, "rho": rho, "k_boot": k, "pearson": p}


def test_cover_verdict_threshold_is_90_percent_at_the_largest_n():
    kind, text = ev.verdict_cover([_cov(10, 0.936, 0.847), _cov(300, 0.945, 0.573)], "t(3)")
    assert kind == "success" and "bei n = 10 zu **93.6 %** und bei n = 300 zu **94.5 %**" in text and "84.7 % und 57.3 %" in text and "t(3), ρ = 0.5" in text
    assert ev.verdict_cover([_cov(10, 0.95, 0.9), _cov(300, 0.90, 0.9)], "x")[0] == "success"
    kind, text = ev.verdict_cover([_cov(10, 0.95, 0.9), _cov(300, 0.8999, 0.9)], "x")
    assert kind == "warning" and "Auch das Kendall-Intervall" in text


def _eff(family, rho, rsd, ksd):
    return {"family": family, "rho": rho, "n": 100, "r_sd": rsd, "k_sd": ksd}


def test_eff_verdict_picks_the_smallest_and_largest_ratio_and_skips_lognormal():
    rows = [_eff("normal", 0.6, 0.0665, 0.0717), _eff("t3", 0.6, 0.1507, 0.0797), _eff("lognormal1", 0.6, 0.0001, 0.5)]
    kind, text = ev.verdict_eff(rows)
    assert kind == "info" and "**0.53-mal** (bei t(3), gemeinsame Skala, ρ = 0.6)" in text and "**1.08-mal** (bei Normal, ρ = 0.6)" in text and "(n = 100)" in text
