"""Das vorgerechnete Raster (data/rank_grid.json): Aufbau, Plausibilität und Übereinstimmung mit dem Erzeuger (Probezelle)."""

import pytest

import rkr_constants as C
import rkr_evaluation as ev


@pytest.fixture(scope="module")
def grid():
    g = ev.load_grid()
    assert g is not None, "data/rank_grid.json fehlt: python tools/make_grid.py"
    return g


def test_probe_cell_still_matches_the_generator(grid):
    """Ändert sich Erzeuger, Statistik oder Seed-Ableitung, passt die gespeicherte Probezelle nicht mehr: dann tools/make_grid.py neu laufen lassen."""
    probe = ev.grid_probe()
    assert probe["runs"] == grid["probe"]["runs"] and all(probe[k] == grid["probe"][k] for k in ("s_naive", "s_bw", "s_boot", "k_boot", "rho_k", "pearson"))     # Trefferanteile exakt
    assert probe == pytest.approx(grid["probe"], rel=1e-9)                       # Breiten: die Summationsreihenfolge weicht je Plattform in der 16. Stelle ab (DEMO-PLAYBOOK §4)


def test_grid_has_every_cell_exactly_once(grid):
    def keys(rows, *names):
        return [tuple(r[n] for n in names) for r in rows]

    assert sorted(keys(grid["rel"], "family", "rho")) == sorted((f, r) for f in C.REL_FAMILIES for r in C.REL_RHOS)
    assert sorted(keys(grid["lever"], "n", "k")) == sorted((n, k) for n in C.LEVER_NS for k in C.LEVER_KS)
    assert sorted(keys(grid["size"], "family", "n")) == sorted((f, n) for f in C.SIZE_FAMILIES for n in C.SIZE_NS)
    assert sorted(keys(grid["ties"], "levels")) == sorted((lv,) for lv in C.TIE_LEVELS)
    assert sorted(keys(grid["cover"], "family", "n", "rho")) == sorted((f, n, r) for f in C.COVER_FAMILIES for n in C.COVER_NS for r in C.COVER_RHOS)
    assert sorted(keys(grid["eff"], "family", "rho")) == sorted((f, r) for f in C.EFF_FAMILIES for r in C.EFF_RHOS)


def test_grid_meta_and_run_counts(grid):
    assert grid["meta"]["seeds"] == list(C.GRID_SEEDS) and grid["meta"]["boot_b"] == C.COVER_BOOT
    assert all(r["runs"] == 4 * (500 if r["n"] >= 300 else 1000) for r in grid["lever"])
    assert all(r["runs"] == 4 * (1000 if r["n"] >= 300 else 2000) for r in grid["size"])
    assert all(r["runs"] == 4 * (150 if r["n"] >= 300 else 250) and r["boot_b"] == 200 for r in grid["cover"])
    assert all(r["runs"] == C.REL_RUNS and r["n"] == C.REL_N for r in grid["rel"]) and all(r["runs"] == 1000 for r in grid["ties"]) and all(r["runs"] == 4000 for r in grid["eff"])


def test_every_rate_is_a_probability(grid):
    for r in grid["cover"]:
        assert all(0.0 <= r[k] <= 1.0 for k in ("s_naive", "s_bw", "s_boot", "k_boot", "rho_k", "pearson")) and r["w_k_boot"] > 0
    for r in grid["lever"]:
        assert all(0.0 <= r[k] <= 1.0 for k in ("share_r", "share_s", "share_k"))
    for r in grid["size"]:
        assert all(0.0 <= r[k] <= 1.0 for k in ("pearson", "spearman", "kendall"))


def test_the_spearman_target_exists_only_for_the_normal_copula(grid):
    for r in grid["cover"]:
        assert r["has_spearman_target"] == (r["family"] in ("normal", "lognormal1"))
        if not r["has_spearman_target"]:
            assert r["s_bw"] == r["s_boot"] == r["s_naive"] == 0.0


def test_kendall_conversion_is_right_in_every_family_and_the_spearman_conversion_is_not(grid):
    assert max(abs(r["rho_from_tau"] - r["rho"]) for r in grid["rel"]) < 0.006
    t3 = [r for r in grid["rel"] if r["family"] == "t3"]
    assert max(abs(r["rho_from_spearman"] - r["rho"]) for r in t3) > 0.01
    logn = [r for r in grid["rel"] if r["family"] == "lognormal1"]
    assert all(abs(r["rho_s"] - r["rho_s_formula"]) < 0.01 and r["r"] < r["rho"] - 0.04 for r in logn)           # gleiche Ränge wie Normal, aber ein ganz anderes r


def test_rank_tests_hold_up_where_pearson_does_not(grid):
    for row in grid["size"]:
        if row["family"] in ("t3", "kontaminiert"):
            assert row["pearson"] > 0.2 and row["spearman"] < 0.1 and row["kendall"] < 0.1, row
        if row["family"] in ("normal", "tindep3", "lognormal1"):
            assert all(abs(row[k] - 0.05) < 0.015 for k in ("pearson", "spearman", "kendall")), row
    for row in grid["lever"]:
        if row["k"] >= 5:
            assert row["share_s"] < 0.1 and row["share_k"] < 0.1 and row["share_r"] > 0.2, row


def test_ties_variants_are_ordered_on_a_coarse_scale(grid):
    first = min(grid["ties"], key=lambda r: r["levels"])
    assert first["tau_a"] < first["tau_b"] < first["gamma"] and first["tau_c"] == pytest.approx(first["tau_b"], abs=1e-3)


def test_kendall_interval_covers_in_every_family(grid):
    for r in grid["cover"]:
        assert r["k_boot"] > 0.9, r
    for r in grid["cover"]:
        if r["family"] in ("t3", "kontaminiert") and r["n"] >= 100:
            assert r["pearson"] < 0.85, r


def test_verdicts_run_on_every_grid_row(grid):
    assert ev.verdict_relations(grid["rel"])[0] in ("success", "info")
    for r in grid["size"]:
        assert ev.verdict_size(r)[0] in ("success", "warning")
    for n in C.LEVER_NS:
        for k in C.LEVER_KS:
            assert ev.verdict_lever(grid["lever"], n, k)[0] in ("info", "warning")
    assert ev.verdict_ties(sorted(grid["ties"], key=lambda r: r["levels"]))[0] == "info"
    for f in C.COVER_FAMILIES:
        for rho in C.COVER_RHOS:
            assert ev.verdict_cover(ev.cover_series(grid, f, rho), f)[0] in ("success", "warning")
    for rho in C.EFF_RHOS:
        assert ev.verdict_eff([ev.find(grid["eff"], family=f, rho=rho) for f in C.EFF_FAMILIES])[0] == "info"
