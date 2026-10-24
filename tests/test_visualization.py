"""Abbildungen und PDF: jede Figur lässt sich bauen, alle Achsen sind gesperrt, die Inhalte stimmen mit den Eingaben überein."""

import numpy as np
import pytest

import rkr_constants as C
import rkr_evaluation as ev
import rkr_pdf_export as PDF
import rkr_visualization as V


def _locked(fig):
    axes = [v for k, v in fig.to_dict()["layout"].items() if k.startswith(("xaxis", "yaxis"))]
    assert axes and all(ax.get("fixedrange") for ax in axes)


def test_scatter_marks_the_leverage_points_and_the_line():
    fig = V.build_scatter([0, 1, 2, 3], [0, 1, 2, 9], lever=(3,), slope=1.0, intercept=0.5)
    assert [t.name for t in fig.data] == ["Beobachtungen", "Hebelpunkt", "Regressionsgerade"] and list(fig.data[1].x) == [3] and list(fig.data[0].x) == [0, 1, 2]
    _locked(fig)
    assert [t.name for t in V.build_scatter([0, 1], [0, 1]).data] == ["Beobachtungen"]


def test_rank_scatter_shows_ranks_not_values():
    fig = V.build_rank_scatter([10.0, 30.0, 20.0, 20.0], [1.0, 2.0, 3.0, 4.0], lever=(0,))
    assert list(fig.data[1].x) == [1.0] and list(fig.data[0].x) == [4.0, 2.5, 2.5]          # Durchschnittsränge bei Gleichstand
    _locked(fig)


def test_intervals_figure_has_one_bar_per_method_and_marks_known_targets_only():
    a = ev.analyse(ev.Settings(family="t3"))
    est = {n: 0.5 for n in ev.INTERVALS}
    fig = V.build_intervals(a.intervals, est, a.targets)
    known = [n for n in ev.INTERVALS if a.targets[n] is not None]
    assert len(fig.data) == 7 and list(fig.data[-1].y) == known and len(known) == 3
    for trace, (lo, hi) in zip(fig.data[:5], a.intervals.values()):
        assert trace.x == (lo, hi)
    _locked(fig)


def test_influence_has_two_series_per_observation():
    fig = V.build_influence([0.01, -0.2, 0.03], [0.0, 0.1, -0.1])
    assert [t.name for t in fig.data] == ["Pearson r", "Spearman ρ_S"] and list(fig.data[1].y) == [0.0, 0.1, -0.1]
    _locked(fig)


def test_curve_figure_has_three_measures():
    res = [ev.curve_experiment(c, 0.5, 40, 10, 1) for c in (1, 2)]
    fig = V.build_curve(res)
    assert [t.name for t in fig.data] == ["Pearson r", "Spearman ρ_S", "Kendall τ"] and list(fig.data[0].x) == ["c = 1", "c = 2"]
    _locked(fig)


def test_lever_size_ties_and_eff_figures_build_from_cells():
    lever = [ev.lever_cell(10, k, seeds=(101,), runs_per_seed=50) for k in (0, 5)]
    fig = V.build_lever(lever, 10)
    assert [t.name for t in fig.data] == ["Pearson-t", "Spearman-t", "Kendall-z"] and list(fig.data[0].x) == [0, 5]
    _locked(fig)
    size = [ev.size_cell(f, 30, seeds=(101,), runs_per_seed=50) for f in ("normal", "tindep3", "t3")]
    fig = V.build_size(size)
    assert len(fig.data) == 3 and len(fig.data[0].x) == 3
    _locked(fig)
    ties = [ev.ties_cell(lv, seeds=(101,), runs_per_seed=5, n=60) for lv in (3, 11)]
    fig = V.build_ties(ties)
    assert [t.name for t in fig.data][:3] == ["τ-a", "τ-b", "γ"]
    _locked(fig)
    eff = [ev.eff_cell(f, 0.6, seeds=(101,), runs_per_seed=30) for f in ("normal", "t3")]
    fig = V.build_eff(eff)
    assert len(fig.data) == 3
    _locked(fig)


def test_cover_figure_shows_spearman_only_with_a_known_target():
    base = {"n": 10, "rho": 0.5, "s_bw": 0.9, "s_boot": 0.9, "k_boot": 0.9, "pearson": 0.8, "s_bw_sd": 0.01, "s_boot_sd": 0.01, "k_boot_sd": 0.01, "pearson_sd": 0.01}
    with_s = V.build_cover([{**base, "has_spearman_target": True}, {**base, "n": 100, "has_spearman_target": True}], "Normal")
    assert [t.name for t in with_s.data] == ["Spearman (Bonett–Wright)", "Spearman (Bootstrap)", "Kendall (Bootstrap)", "Pearson (Fisher-z) gegen ρ"]
    without = V.build_cover([{**base, "has_spearman_target": False}, {**base, "n": 100, "has_spearman_target": False}], "t(3)")
    assert [t.name for t in without.data] == ["Kendall (Bootstrap)", "Pearson (Fisher-z) gegen ρ"]
    assert without.layout.shapes[0].y0 == 95
    _locked(with_s)


def test_pdf_is_a_valid_pdf_and_text_is_core_font_safe():
    a = ev.analyse(ev.Settings(family="kontaminiert", rho=0.3, n=40, lev_count=1, lev_k=6.0))
    data = PDF.build_pdf(a)
    assert data[:5] == b"%PDF-" and len(data) > 1500
    assert PDF._text("ρ = 0,5 – √n · σ τ") == "rho = 0,5 - sqrt n * sigma tau" and PDF._text("**fett** „zitat“") == 'fett "zitat"' and PDF._text("✓") == "?"
