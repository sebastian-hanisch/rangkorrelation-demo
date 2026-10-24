"""Jede Zahl der README steht hier und wird aus dem vorgerechneten Raster (data/rank_grid.json) oder über dieselben Auswertungsfunktionen wie in der App nachgerechnet (DEMO-PLAYBOOK §4).

Zufall nur aus SplitMix64 mit festen Seeds. Raster-Werte stehen gerundet auf drei Stellen (bzw. ein Zehntelprozent) in der README; die Vergleiche nutzen eine Toleranz von
einem halben Tausendstel. Live nachgerechnete Werte (Kurve) lassen 0,0015 zu."""

from pathlib import Path

import pytest

import prs_algorithm as PA
import prs_scenario as S
import rkr_algorithm as R
import rkr_constants as C
import rkr_evaluation as ev

README = (Path(__file__).resolve().parent.parent / "README.md").read_text(encoding="utf-8")
TOL = 0.0006


@pytest.fixture(scope="module")
def grid():
    g = ev.load_grid()
    assert g is not None
    return g


def _rel(grid, family, rho):
    return ev.find(grid["rel"], family=family, rho=rho)


def test_claim_formulas_hold_or_not(grid):
    """README: τ bei ρ = 0,6: Normal 0,410, t(3) 0,411, kontaminiert 0,412, Lognormal 0,409 (Formel 0,410); ρ_S: Normal 0,582, Lognormal 0,581 (Formel 0,582), t(3) 0,563, kontaminiert 0,573."""
    assert [_rel(grid, f, 0.6)["tau"] for f in ("normal", "t3", "kontaminiert", "lognormal1")] == pytest.approx([0.410, 0.411, 0.412, 0.409], abs=TOL)
    assert _rel(grid, "normal", 0.6)["tau_formula"] == pytest.approx(0.410, abs=TOL) and _rel(grid, "normal", 0.6)["rho_s_formula"] == pytest.approx(0.582, abs=TOL)
    assert [_rel(grid, f, 0.6)["rho_s"] for f in ("normal", "lognormal1", "t3", "kontaminiert")] == pytest.approx([0.582, 0.581, 0.563, 0.573], abs=TOL)


def test_claim_conversions(grid):
    """README: sin(πτ/2) trifft ρ in allen 12 Zellen auf höchstens 0,004, 2 sin(πρ_S/6) weicht bis 0,019 ab (t(3), ρ = 0,6: 0,581); Lognormal: ρ_S 0,288 / 0,581 / 0,891, r 0,206 / 0,466 / 0,851."""
    assert len(grid["rel"]) == 12
    assert max(abs(r["rho_from_tau"] - r["rho"]) for r in grid["rel"]) == pytest.approx(0.004, abs=0.0006)
    worst = max(grid["rel"], key=lambda r: abs(r["rho_from_spearman"] - r["rho"]))
    assert (worst["family"], worst["rho"]) == ("t3", 0.6) and abs(worst["rho_from_spearman"] - 0.6) == pytest.approx(0.019, abs=TOL) and worst["rho_from_spearman"] == pytest.approx(0.581, abs=TOL)
    logn = [_rel(grid, "lognormal1", rho) for rho in C.REL_RHOS]
    assert [r["rho_s"] for r in logn] == pytest.approx([0.288, 0.581, 0.891], abs=TOL) and [r["r"] for r in logn] == pytest.approx([0.206, 0.466, 0.851], abs=TOL)


def test_claim_curve_without_and_with_noise():
    """README (n = 100, 200 Läufe, Seeds 7 + i): ohne Rauschen r = 0,945 / 0,807 / 0,583 / 0,465 / 0,393, ρ_S = τ = 1; mit Rauschen 0,5: r 0,716 / 0,781 / 0,582 / 0,465 / 0,393, ρ_S 0,682 / 0,857 / 0,908 / 0,918 / 0,914, τ 0,501 / 0,688 / 0,774 / 0,798 / 0,798."""
    clean = [ev.curve_experiment(c, 0.0, 100, 200, 7 + i) for i, c in enumerate(C.CURVE_CS)]
    assert [r["r"] for r in clean] == pytest.approx([0.945, 0.807, 0.583, 0.465, 0.393], abs=0.0015)
    assert all(r["rho_s"] == pytest.approx(1.0, abs=1e-12) and r["tau"] == pytest.approx(1.0, abs=1e-12) for r in clean)
    noisy = [ev.curve_experiment(c, 0.5, 100, 200, 7 + i) for i, c in enumerate(C.CURVE_CS)]
    assert [r["r"] for r in noisy] == pytest.approx([0.716, 0.781, 0.582, 0.465, 0.393], abs=0.0015)
    assert [r["rho_s"] for r in noisy] == pytest.approx([0.682, 0.857, 0.908, 0.918, 0.914], abs=0.0015)
    assert [r["tau"] for r in noisy] == pytest.approx([0.501, 0.688, 0.774, 0.798, 0.798], abs=0.0015)
    assert noisy[0]["rho_s"] < noisy[0]["r"]                                          # bei c = 0,5 liegt das Rangmaß unter Pearson


def test_claim_pearson_of_the_curve_depends_on_n():
    """README: bei c = 3 ohne Rauschen fällt r von 0,638 (n = 20) über 0,458 (100) und 0,317 (500) auf 0,235 (2 000); Population 0,033."""
    got = [ev.curve_experiment(3, 0.0, n, 200, 9)["r"] for n in (20, 100, 500, 2000)]
    assert got == pytest.approx([0.638, 0.458, 0.317, 0.235], abs=0.0015)
    assert ev.curve_population_r(3) == pytest.approx(0.033, abs=0.0006)


def test_claim_leverage_point(grid):
    """README: bei k = 5 (n = 10 / 30 / 100 / 300): Pearson 84,3 / 84,7 / 52,8 / 26,5 %, Spearman 9,0 / 5,8 / 5,3 / 5,9 %, Kendall 9,1 / 6,3 / 5,7 / 5,9 %; k = 0: Pearson 4,8 / 4,6 / 5,0 / 5,2 %."""
    at5 = [ev.find(grid["lever"], n=n, k=5) for n in C.LEVER_NS]
    assert [r["share_r"] for r in at5] == pytest.approx([0.843, 0.847, 0.528, 0.265], abs=TOL)
    assert [r["share_s"] for r in at5] == pytest.approx([0.090, 0.058, 0.053, 0.059], abs=TOL)
    assert [r["share_k"] for r in at5] == pytest.approx([0.091, 0.0635, 0.057, 0.059], abs=TOL)
    assert [ev.find(grid["lever"], n=n, k=0)["share_r"] for n in C.LEVER_NS] == pytest.approx([0.048, 0.046, 0.050, 0.052], abs=TOL)
    cell = ev.find(grid["lever"], n=30, k=5)
    assert (cell["mean_r"], cell["mean_s"], cell["mean_tau"]) == pytest.approx((0.471, 0.100, 0.069), abs=TOL)


def test_claim_leverage_ranks_stop_moving(grid):
    """README: der Anteil der Rangtests ändert sich ab k ≈ 3 kaum noch."""
    for n in C.LEVER_NS:
        shares = [ev.find(grid["lever"], n=n, k=k)["share_s"] for k in (3, 4, 5, 6, 8, 10, 15)]
        assert max(shares) - min(shares) < 0.005, n


SIZE = {("t3", 30): (0.226, 0.072, 0.080), ("t3", 100): (0.323, 0.071, 0.085), ("t3", 300): (0.410, 0.075, 0.085),
        ("kontaminiert", 30): (0.300, 0.059, 0.063), ("kontaminiert", 100): (0.409, 0.063, 0.069), ("kontaminiert", 300): (0.4285, 0.068, 0.071)}


def test_claim_size_under_common_scale(grid):
    """README, Experiment 4: Pearson / Spearman / Kendall bei ρ = 0 für t(3) und kontaminiert."""
    for (fam, n), expected in SIZE.items():
        row = ev.find(grid["size"], family=fam, n=n)
        assert (row["pearson"], row["spearman"], row["kendall"]) == pytest.approx(expected, abs=TOL), (fam, n)


def test_claim_size_ranges_of_the_other_families(grid):
    """README: t(5): Pearson 13,2 bis 18,6 %; t(10): 7,2 bis 8,5 %; Rangtests bei t(5) und t(10) 5,1 bis 7,0 %; unabhängige Ränder und Lognormal: alle drei nahe 5 %; Rangtests bei gemeinsamer Streuung 5,9 bis 8,5 %."""
    t5 = [ev.find(grid["size"], family="t5", n=n)["pearson"] for n in C.SIZE_NS]
    t10 = [ev.find(grid["size"], family="t10", n=n)["pearson"] for n in C.SIZE_NS]
    assert (min(t5), max(t5)) == pytest.approx((0.132, 0.186), abs=TOL) and (min(t10), max(t10)) == pytest.approx((0.072, 0.085), abs=TOL)
    rank = [ev.find(grid["size"], family=f, n=n)[k] for f in ("t5", "t10") for n in C.SIZE_NS for k in ("spearman", "kendall")]
    assert (min(rank), max(rank)) == pytest.approx((0.051, 0.070), abs=TOL)
    common = [ev.find(grid["size"], family=f, n=n)[k] for f in ("t3", "kontaminiert") for n in C.SIZE_NS for k in ("spearman", "kendall")]
    assert (min(common), max(common)) == pytest.approx((0.059, 0.085), abs=TOL)
    assert max(ev.find(grid["size"], family=f, n=n)["pearson"] for f in ("t3", "kontaminiert") for n in C.SIZE_NS) == pytest.approx(0.428, abs=TOL)
    for fam in ("normal", "tindep3", "lognormal1"):
        for n in C.SIZE_NS:
            row = ev.find(grid["size"], family=fam, n=n)
            assert all(abs(row[k] - 0.05) < 0.01 for k in ("pearson", "spearman", "kendall")), (fam, n)


def test_claim_ties(grid):
    """README: 3 Stufen: τ-a 0,297, τ-b 0,444, τ-c 0,444, γ 0,625, ρ_S 0,493 (= Pearson der Codes); 5 Stufen 0,362 / 0,451 / 0,551; 11: 0,397 / 0,436 / 0,477; 100: 0,409 / 0,412 / 0,416."""
    rows = {r["levels"]: r for r in grid["ties"]}
    assert (rows[3]["tau_a"], rows[3]["tau_b"], rows[3]["tau_c"], rows[3]["gamma"], rows[3]["rho_s"], rows[3]["pearson_codes"]) == pytest.approx((0.297, 0.444, 0.444, 0.625, 0.493, 0.493), abs=TOL)
    for lv, vals in ((5, (0.362, 0.451, 0.551)), (11, (0.397, 0.436, 0.477)), (100, (0.409, 0.412, 0.416))):
        assert (rows[lv]["tau_a"], rows[lv]["tau_b"], rows[lv]["gamma"]) == pytest.approx(vals, abs=TOL)


def _range(grid, family, key):
    vals = [r[key] for r in grid["cover"] if r["family"] == family]
    return min(vals), max(vals)


def test_claim_coverage_ranges(grid):
    """README, Experiment 6: Kendall-Bootstrap: Normal 93,0 bis 98,1 %, Lognormal 92,2 bis 98,1 %, t(3) 92,6 bis 96,2 %, kontaminiert 93,2 bis 97,0 %; Pearson gegen ρ: Normal 93,2 bis 96,5 %, t(3) 57,3 bis 86,3 %,
    kontaminiert 54,7 bis 84,7 %, Lognormal 25,2 bis 95,8 %."""
    for fam, lo, hi in (("normal", 0.930, 0.981), ("lognormal1", 0.922, 0.981), ("t3", 0.926, 0.962), ("kontaminiert", 0.932, 0.970)):
        assert _range(grid, fam, "k_boot") == pytest.approx((lo, hi), abs=TOL), fam
    for fam, lo, hi in (("normal", 0.932, 0.965), ("t3", 0.573, 0.863), ("kontaminiert", 0.547, 0.847), ("lognormal1", 0.252, 0.958)):
        assert _range(grid, fam, "pearson") == pytest.approx((lo, hi), abs=TOL), fam


def test_claim_spearman_intervals(grid):
    """README: Bonett–Wright 93,4 bis 96,9 % (Normal), 93,5 bis 96,4 % (Lognormal); naives Fisher 90,6 bis 95,5 % bzw. 88,5 bis 95,6 %; Bonett–Wright liegt in keiner Zelle unter 93,4 %."""
    assert _range(grid, "normal", "s_bw") == pytest.approx((0.934, 0.969), abs=TOL) and _range(grid, "lognormal1", "s_bw") == pytest.approx((0.935, 0.964), abs=TOL)
    assert _range(grid, "normal", "s_naive") == pytest.approx((0.906, 0.955), abs=TOL) and _range(grid, "lognormal1", "s_naive") == pytest.approx((0.885, 0.956), abs=TOL)


def test_claim_widths_and_seed_spread(grid):
    """README: n = 100, ρ = 0,5: Normal Bonett–Wright 0,321, Kendall 0,234, umgerechnet 0,318; t(3) 0,326, 0,267, 0,362; Streuung über die Seeds bis 2,7 Punkte (Kendall, Normal) bzw. 3,7 (t(3))."""
    normal, t3 = ev.find(grid["cover"], family="normal", n=100, rho=0.5), ev.find(grid["cover"], family="t3", n=100, rho=0.5)
    assert (normal["w_s_bw"], normal["w_k_boot"], normal["w_rho_k"]) == pytest.approx((0.321, 0.234, 0.318), abs=TOL)
    assert (t3["w_s_bw"], t3["w_k_boot"], t3["w_rho_k"]) == pytest.approx((0.326, 0.267, 0.362), abs=TOL)
    assert max(r["k_boot_sd"] for r in grid["cover"] if r["family"] == "normal") == pytest.approx(0.027, abs=TOL)
    assert max(r["k_boot_sd"] for r in grid["cover"] if r["family"] == "t3") == pytest.approx(0.037, abs=TOL)


def test_claim_efficiency(grid):
    """README (n = 100, ρ = 0,6): Streuung Pearson / 2 sin(πρ_S/6) / sin(πτ/2): Normal 0,065 / 0,070 / 0,069, t(3) 0,146 / 0,083 / 0,082, kontaminiert 0,153 / 0,078 / 0,076, Lognormal 0,134 (Mittel 0,502) / 0,070 / 0,069;
    Spearman-Umrechnung bei t(3): Mittel 0,573; Verhältnis Kendall zu Pearson 0,50 bis 1,07 (ρ = 0,6) und 0,45 bis 1,12 (ρ = 0,9)."""
    for fam, vals in (("normal", (0.065, 0.070, 0.069)), ("t3", (0.146, 0.083, 0.082)), ("kontaminiert", (0.153, 0.078, 0.076)), ("lognormal1", (0.134, 0.070, 0.069))):
        row = ev.find(grid["eff"], family=fam, rho=0.6)
        assert (row["r_sd"], row["s_sd"], row["k_sd"]) == pytest.approx(vals, abs=TOL), fam
    assert ev.find(grid["eff"], family="lognormal1", rho=0.6)["r_mean"] == pytest.approx(0.502, abs=TOL)
    assert ev.find(grid["eff"], family="t3", rho=0.6)["s_mean"] == pytest.approx(0.573, abs=TOL)
    for rho, lo, hi in ((0.6, 0.50, 1.07), (0.9, 0.45, 1.12)):
        ratios = [ev.find(grid["eff"], family=f, rho=rho)["k_sd"] / ev.find(grid["eff"], family=f, rho=rho)["r_sd"] for f in C.EFF_FAMILIES if f != "lognormal1"]
        assert (min(ratios), max(ratios)) == pytest.approx((lo, hi), abs=0.006)


def test_claim_hand_values(four, tied):
    """README: ρ_S = 0,8, τ = 2/3 (C = 5, D = 1), sin(πτ/2) = 0,866; mit Gleichständen τ-a 0,667, τ-b 0,8, τ-c 0,75, γ 1, ρ_S 0,833."""
    assert R.spearman_r(*four) == pytest.approx(0.8) and R.kendall_tau(*four) == pytest.approx(2 / 3) and R.rho_from_tau(2 / 3) == pytest.approx(0.866, abs=0.0006)
    k = R.kendall_counts(*four)
    assert (k["C"], k["D"]) == (5, 1)
    assert [R.kendall_tau(*tied, v) for v in ("a", "b", "c", "gamma")] == pytest.approx([0.667, 0.8, 0.75, 1.0], abs=0.0006) and R.spearman_r(*tied) == pytest.approx(0.833, abs=0.0006)
    assert PA.pearson_r(*four) == pytest.approx(0.8)


def test_claim_preset_help_numbers(grid):
    """Hilfetexte der Presets: 94,7 % (Standard); 84,7 / 5,8 / 6,3 % (Einzelpunkt); 40,9 / 6,3 / 6,9 % (Spitzenwochen); 94,4 und 69,0 % (schwerer Rand); Lognormal: r 0,38, Pearson-Abdeckung 59,5 %, Kendall 95,5 %."""
    assert ev.find(grid["cover"], family="normal", n=30, rho=0.5)["k_boot"] == pytest.approx(0.947, abs=TOL)
    cell = ev.find(grid["lever"], n=30, k=5)
    assert (cell["share_r"], cell["share_s"], cell["share_k"]) == pytest.approx((0.847, 0.058, 0.0635), abs=TOL)
    row = ev.find(grid["size"], family="kontaminiert", n=100)
    assert (row["pearson"], row["spearman"], row["kendall"]) == pytest.approx((0.409, 0.063, 0.069), abs=TOL)
    heavy = ev.find(grid["cover"], family="t3", n=100, rho=0.5)
    assert (heavy["k_boot"], heavy["pearson"]) == pytest.approx((0.944, 0.690), abs=TOL)
    logn = ev.find(grid["cover"], family="lognormal1", n=100, rho=0.5)
    assert (logn["pearson"], logn["k_boot"], S.true_r("lognormal1", 0.5)) == pytest.approx((0.595, 0.955, 0.378), abs=0.0006)
    for name in C.PRESETS:
        assert C.PRESET_HELP[name] and C.PRESETS[name]["family"] in C.FAMILIES


def test_claim_readme_mentions_its_key_numbers():
    """Verbindet die README mit den Tests: die Zahlen, die dort als Kernbefunde stehen, kommen im Text vor."""
    for needle in ("5,9 bis 8,5 %", "84,7 %", "0,297, 0,444 und 0,625", "0,004", "0,019", "0,682", "0,235", "93,4 bis 96,9", "0,50 (kontaminiert) bis 1,07", "564 Sekunden"):
        assert needle in README, needle


def _p(v):
    return f"{100 * v:.1f}".replace(".", ",")


def test_readme_percentages_use_the_rounding_of_the_app(grid):
    """Die Prozentreihen der README entsprechen Zeichen für Zeichen der Rundung, die die App zeigt (z. B. 0,0635 wird 6,3 %)."""
    lever = [ev.find(grid["lever"], n=n, k=5) for n in C.LEVER_NS]
    for key in ("share_r", "share_s", "share_k"):
        assert " / ".join(_p(r[key]) for r in lever) in README, key
    for fam in ("t3", "kontaminiert"):
        rows = [ev.find(grid["size"], family=fam, n=n) for n in C.SIZE_NS]
        for key in ("pearson", "spearman", "kendall"):
            assert " / ".join(_p(r[key]) for r in rows) in README, (fam, key)
    k0 = [ev.find(grid["lever"], n=n, k=0)["share_r"] for n in C.LEVER_NS]
    assert " / ".join(_p(v) for v in k0) in README
