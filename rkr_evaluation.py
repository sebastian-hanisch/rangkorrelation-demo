"""Auswertung: Einzeldatensatz, Messreihen (E1 bis E6), Zellen des vorgerechneten Rasters, Zugriff auf das Raster und alle Verdikte als reine Funktionen."""

import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

import prs_algorithm as PA
import prs_scenario as S
import rkr_algorithm as R
import rkr_constants as C

GRID_PATH = Path(__file__).resolve().parent / "data" / "rank_grid.json"
INTERVALS = ("Pearson (Fisher-z)", "Spearman (Bonett–Wright)", "Spearman (Bootstrap)", "Kendall (Bootstrap)", "ρ aus Kendall (Bootstrap)")


# --- Einzeldatensatz -------------------------------------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class Settings:
    family: str = "normal"
    rho: float = C.DEFAULT_RHO
    n: int = C.DEFAULT_N
    lev_count: int = C.DEFAULT_LEV_COUNT
    lev_k: float = C.DEFAULT_LEV_K
    lev_dir: str = "gleichläufig"
    boot_b: int = C.DEFAULT_B
    seed: int = C.DEFAULT_SEED


@dataclass(frozen=True)
class Analysis:
    settings: Settings
    x: tuple[float, ...]
    y: tuple[float, ...]
    lever: tuple[int, ...]
    true_r: float             # was Pearson-r erwartungsgemäß misst (Familie, ρ)
    tau_true: float
    rs_true: float | None     # nur bei Normalkopula bekannt
    r: float
    rho_s: float
    tau: float                # τ-b
    p_pearson: float
    p_spearman: float
    p_kendall: float
    intervals: dict           # Name -> (unten, oben)
    targets: dict             # Name -> Zielwert oder None
    covers: dict              # Name -> enthält das Intervall den Zielwert (None, wenn der Zielwert unbekannt ist)
    infl_r: tuple[float, ...]
    infl_s: tuple[float, ...]


def affordable_boot(boot_b: int, n: int, budget: int = C.RANK_BUDGET) -> int:
    """So viele Bootstrap-Ziehungen, wie das Budget B · n² erlaubt (mindestens 50)."""
    return max(50, min(boot_b, budget // max(1, n * n)))


def analyse(s: Settings) -> Analysis:
    ds = S.make_dataset(s.family, s.n, s.rho, s.seed, s.lev_count, s.lev_k, C.LEV_DIRS[s.lev_dir])
    x, y = np.array(ds.x), np.array(ds.y)
    r, rs, tau = PA.pearson_r(x, y), R.spearman_r(x, y), R.kendall_tau(x, y, "b")
    B = affordable_boot(s.boot_b, s.n)
    rs_b, tau_b = R.rank_bootstrap(x, y, B, S.VRng(s.seed + 1))
    tau_int = PA.percentile_interval(tau_b)
    intervals = {
        INTERVALS[0]: PA.fisher_interval(r, s.n),
        INTERVALS[1]: R.bonett_wright_interval(rs, s.n),
        INTERVALS[2]: PA.percentile_interval(rs_b),
        INTERVALS[3]: tau_int,
        INTERVALS[4]: R.interval_through(R.rho_from_tau, tau_int),
    }
    rs_true = R.spearman_true(s.family, s.rho)
    targets = {INTERVALS[0]: ds.true_r, INTERVALS[1]: rs_true, INTERVALS[2]: rs_true, INTERVALS[3]: R.tau_true(s.family, s.rho), INTERVALS[4]: s.rho if s.family != C.INDEPENDENT_T3 else 0.0}
    covers = {k: (None if targets[k] is None else lo_hi[0] <= targets[k] <= lo_hi[1]) for k, lo_hi in intervals.items()}
    return Analysis(s, ds.x, ds.y, ds.lever, ds.true_r, targets[INTERVALS[3]], rs_true, r, rs, tau, PA.t_test_p(r, s.n), R.spearman_t_p(rs, s.n), R.kendall_z_p(tau, s.n),
                    intervals, targets, covers, tuple(map(float, PA.influence(x, y))), tuple(map(float, R.influence_rank(x, y))))


# --- E2: Kurve mit Rauschen (live) -------------------------------------------------------------------------------------------------------------

def affordable_curve_runs(n: int, runs: int, budget: int = 2 * 10**8) -> int:
    return max(5, min(runs, budget // max(1, n * n)))


def curve_population_r(c: float) -> float:
    """Populationskorrelation von x und e^(c·x) für standardnormales x ohne Rauschen: c/√(e^(c²) − 1) (Formel hier nur durch die Messung gestützt)."""
    return c / math.sqrt(math.exp(c * c) - 1.0)


def curve_experiment(c: float, noise: float, n: int, runs: int, seed: int) -> dict:
    """y = e^(c·x) + noise·ε mit x, ε standardnormal: Mittel von r, ρ_S, τ_b über die Läufe, dazu die Streuung von r."""
    runs = affordable_curve_runs(n, runs)
    rng = S.VRng(seed)
    X = rng.normal((runs, n))
    Y = np.exp(c * X) + noise * rng.normal((runs, n))
    r = PA.pearson_rows(X, Y)
    rs = R.spearman_rows(X, Y)
    tau = R.kendall_rows(X, Y, "b")
    return {"c": c, "noise": noise, "n": n, "runs": runs, "r": float(r.mean()), "r_sd": float(r.std()), "rho_s": float(rs.mean()), "tau": float(tau.mean()),
            "population_r": curve_population_r(c) if noise == 0 else None}


# --- Zellen des vorgerechneten Rasters -------------------------------------------------------------------------------------------------------------

def cell_seed(kind: str, family: str, n: int, rho: float, s: int) -> int:
    """Eigener Seed je Zelle und Plan-Seed s (verschieden für Art, Familie, n und ρ)."""
    kinds = ("rel", "lever", "size", "ties", "cover", "eff")
    return s * 1_000_003 + n * 101 + int(round(rho * 10)) * 7 + list(C.ALL_FAMILIES).index(family) * 1013 + kinds.index(kind) * 100_000_007


def relations_cell(family: str, rho: float, runs: int = C.REL_RUNS, n: int = C.REL_N) -> dict:
    """E1: Mittel von r, ρ_S, τ_b über `runs` Stichproben der Größe n gegen die Formeln und die beiden Umrechnungen."""
    X, Y = S.draw_pairs(family, runs, n, rho, S.VRng(cell_seed("rel", family, n, rho, 1)))
    r, rs, tau = PA.pearson_rows(X, Y), R.spearman_rows(X, Y), R.kendall_rows(X, Y, "b")
    return {"family": family, "rho": rho, "n": n, "runs": runs, "r": float(r.mean()), "rho_s": float(rs.mean()), "tau": float(tau.mean()),
            "rho_s_sd": float(rs.std(ddof=1)), "tau_sd": float(tau.std(ddof=1)), "tau_formula": R.tau_formula(rho), "rho_s_formula": R.spearman_formula(rho),
            "rho_from_tau": R.rho_from_tau(float(tau.mean())), "rho_from_spearman": R.rho_from_spearman(float(rs.mean())), "true_r": S.true_r(family, rho)}


def lever_cell(n: int, k: float, seeds=C.GRID_SEEDS, runs_per_seed: int | None = None) -> dict:
    """E3: Hebelpunkt (k, k) in n unabhängigen Normalpaaren: Anteil „signifikant“ für Pearson-t, Spearman-t und Kendall-z, mittlere r, ρ_S, τ."""
    rps = runs_per_seed or (500 if n >= 300 else 1000)
    acc = []
    for s in seeds:
        X, Y = S.draw_pairs("normal", rps, n, 0.0, S.VRng(cell_seed("lever", "normal", n, 0.0, s)))
        if k:
            X[:, 0], Y[:, 0] = k, k
        r, rs, tau = PA.pearson_rows(X, Y), R.spearman_rows(X, Y), R.kendall_rows(X, Y, "b")
        acc.append((PA.rejects_rows(r, n).mean(), R.rejects_spearman_rows(rs, n).mean(), R.rejects_kendall_rows(tau, n).mean(), r.mean(), rs.mean(), tau.mean()))
    m = np.mean(acc, axis=0)
    return {"n": n, "k": k, "runs": rps * len(seeds), "share_r": float(m[0]), "share_s": float(m[1]), "share_k": float(m[2]), "mean_r": float(m[3]), "mean_s": float(m[4]), "mean_tau": float(m[5])}


def size_cell(family: str, n: int, seeds=C.GRID_SEEDS, runs_per_seed: int | None = None) -> dict:
    """E4: Fehlalarmrate bei wahrem ρ = 0 für Pearson-t, Spearman-t und Kendall-z (Mittel über die Seeds, Streuung des Pearson-Werts über die Seeds)."""
    rps = runs_per_seed or (1000 if n >= 300 else 2000)
    acc = []
    for s in seeds:
        X, Y = S.draw_pairs(family, rps, n, 0.0, S.VRng(cell_seed("size", family, n, 0.0, s)))
        acc.append((PA.rejects_rows(PA.pearson_rows(X, Y), n).mean(), R.rejects_spearman_rows(R.spearman_rows(X, Y), n).mean(), R.rejects_kendall_rows(R.kendall_rows(X, Y, "b"), n).mean()))
    acc = np.array(acc)
    m = acc.mean(0)
    return {"family": family, "n": n, "runs": rps * len(seeds), "pearson": float(m[0]), "spearman": float(m[1]), "kendall": float(m[2]), "pearson_sd": float(acc[:, 0].std(ddof=1))}


def bin_rows(X, levels: int) -> np.ndarray:
    """Je Zeile in `levels` gleich besetzte Stufen einteilen (0 bis levels − 1), wie eine grobe Skala."""
    out = np.empty_like(X)
    qs = np.linspace(0, 1, levels + 1)[1:-1]
    for i, row in enumerate(X):
        out[i] = np.digitize(row, np.quantile(row, qs))
    return out


def ties_cell(levels: int, seeds=C.GRID_SEEDS, runs_per_seed: int = 250, n: int = C.TIE_N, rho: float = C.TIE_RHO) -> dict:
    """E5: Skala mit `levels` Stufen aus ρ = 0,6: Mittel von τ-a, τ-b, τ-c, γ, ρ_S (Durchschnittsränge) und Pearson der Codes."""
    acc = []
    for s in seeds:
        X, Y = S.draw_pairs("normal", runs_per_seed, n, rho, S.VRng(cell_seed("ties", "normal", levels, rho, s)))
        U, V = bin_rows(X, levels), bin_rows(Y, levels)
        acc.append([R.kendall_rows(U, V, v).mean() for v in ("a", "b", "c", "gamma")] + [R.spearman_rows(U, V).mean(), PA.pearson_rows(U, V).mean()])
    m = np.mean(acc, axis=0)
    return {"levels": levels, "n": n, "rho": rho, "runs": runs_per_seed * len(seeds), "tau_a": float(m[0]), "tau_b": float(m[1]), "tau_c": float(m[2]), "gamma": float(m[3]),
            "rho_s": float(m[4]), "pearson_codes": float(m[5])}


def cover_runs(family: str, n: int, rho: float, runs: int, boot_b: int, seed: int) -> dict:
    """E6: Abdeckung der Intervalle über `runs` Stichproben: Bonett–Wright und naives Fisher für ρ_S (nur Normalkopula), Bootstrap für ρ_S und τ, ρ aus Kendall, Pearson-Fisher gegen ρ."""
    tau_t, rs_t = R.tau_true(family, rho), R.spearman_true(family, rho)
    hit = dict.fromkeys(("s_naive", "s_bw", "s_boot", "k_boot", "rho_k", "pearson"), 0)
    width = dict.fromkeys(("s_bw", "k_boot", "rho_k"), 0.0)
    for sub in S.substream_seeds(seed, runs):
        rng = S.VRng(sub)
        X, Y = S.draw_pairs(family, 1, n, rho, rng)
        x, y = X[0], Y[0]
        rs, tau = R.spearman_r(x, y), R.kendall_tau(x, y, "b")
        rs_b, tau_b = R.rank_bootstrap(x, y, boot_b, rng)
        lo, hi = R.bonett_wright_interval(rs, n)
        hit["s_bw"] += rs_t is not None and lo <= rs_t <= hi
        width["s_bw"] += hi - lo
        lo, hi = R.naive_spearman_interval(rs, n)
        hit["s_naive"] += rs_t is not None and lo <= rs_t <= hi
        lo, hi = PA.percentile_interval(rs_b)
        hit["s_boot"] += rs_t is not None and lo <= rs_t <= hi
        lo, hi = PA.percentile_interval(tau_b)
        hit["k_boot"] += lo <= tau_t <= hi
        width["k_boot"] += hi - lo
        lo_r, hi_r = R.rho_from_tau(lo), R.rho_from_tau(hi)
        hit["rho_k"] += lo_r <= rho <= hi_r
        width["rho_k"] += hi_r - lo_r
        flo, fhi = PA.fisher_interval(PA.pearson_r(x, y), n)
        hit["pearson"] += flo <= rho <= fhi
    out = {"runs": runs}
    out.update({k: v / runs for k, v in hit.items()})
    out.update({"w_" + k: v / runs for k, v in width.items()})
    return out


def cover_cell(family: str, n: int, rho: float, seeds=C.GRID_SEEDS, runs_per_seed: int | None = None, boot_b: int = C.COVER_BOOT) -> dict:
    """E6: Zelle des Abdeckungsrasters: Mittel über die Seeds und Standardabweichung über die Seeds je Verfahren."""
    rps = runs_per_seed or (150 if n >= 300 else 250)
    per = [cover_runs(family, n, rho, rps, boot_b, cell_seed("cover", family, n, rho, s)) for s in seeds]
    has_s = R.spearman_true(family, rho) is not None
    row = {"family": family, "n": n, "rho": rho, "runs": rps * len(seeds), "boot_b": boot_b, "has_spearman_target": has_s}
    for key in ("s_naive", "s_bw", "s_boot", "k_boot", "rho_k", "pearson", "w_s_bw", "w_k_boot", "w_rho_k"):
        vals = [p[key] for p in per]
        row[key] = float(np.mean(vals))
        if not key.startswith("w_"):
            row[key + "_sd"] = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
    return row


def eff_cell(family: str, rho: float, seeds=C.GRID_SEEDS, runs_per_seed: int = 1000, n: int = C.EFF_N) -> dict:
    """E6: Schätzer für ρ: r, 2 sin(πρ_S/6), sin(πτ/2): Mittel, Standardabweichung, RMSE (r gegen das, was r misst; die Umrechnungen gegen ρ)."""
    est = {"r": [], "s": [], "k": []}
    for s in seeds:
        X, Y = S.draw_pairs(family, runs_per_seed, n, rho, S.VRng(cell_seed("eff", family, n, rho, s)))
        est["r"].append(PA.pearson_rows(X, Y))
        est["s"].append(2 * np.sin(np.pi * R.spearman_rows(X, Y) / 6))
        est["k"].append(np.sin(np.pi * R.kendall_rows(X, Y, "b") / 2))
    out = {"family": family, "rho": rho, "n": n, "runs": runs_per_seed * len(seeds)}
    for key, parts in est.items():
        v = np.concatenate(parts)
        target = S.true_r(family, rho) if key == "r" else rho
        out[key + "_mean"], out[key + "_sd"], out[key + "_rmse"] = float(v.mean()), float(v.std()), float(math.sqrt(np.mean((v - target) ** 2)))
    return out


def grid_probe() -> dict:
    """Kleine feste Zelle, die der Test nachrechnet: ändert sich Erzeuger oder Statistik, passt sie nicht mehr zum gespeicherten Raster."""
    return cover_runs("t3", 20, 0.5, 8, 50, 12345)


# --- Zugriff auf das Raster ------------------------------------------------------------------------------------------------------------------------

def load_grid(path: Path = GRID_PATH) -> dict | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def find(rows: list[dict], **keys) -> dict | None:
    for row in rows:
        if all(row.get(k) == v or (isinstance(v, float) and isinstance(row.get(k), (int, float)) and abs(row[k] - v) < 1e-9) for k, v in keys.items()):
            return row
    return None


def cover_series(grid: dict, family: str, rho: float) -> list[dict]:
    return sorted((r for r in grid["cover"] if r["family"] == family and abs(r["rho"] - rho) < 1e-9), key=lambda r: r["n"])


# --- Verdikte (reine Funktionen: Art, Text) -----------------------------------------------------------------------------------------------------

def verdict_main(a: Analysis) -> tuple[str, str]:
    """Kernfrage der Hauptansicht: Wo liegen die Zielwerte, und welche Intervalle fangen sie?"""
    s = a.settings
    known = [k for k in INTERVALS if a.covers[k] is not None]
    hit = [k for k in known if a.covers[k]]
    miss = [k for k in known if not a.covers[k]]
    base = f"Gemessen: r = **{a.r:.2f}**, ρ_S = **{a.rho_s:.2f}**, τ = **{a.tau:.2f}** (n = {s.n}); die Wahrheit ist ρ = **{s.rho:.2f}**, τ = **{a.tau_true:.2f}** ({C.ALL_FAMILIES[s.family]})."
    if a.lever:
        base += f" {len(a.lever)} Hebelpunkt(e) bei k = {s.lev_k:g}."
    if not known:
        return "info", f"{base} Für kein Intervall ist ein Zielwert bekannt."
    if not miss:
        return "success", f"{base} Alle {len(hit)} Intervalle mit bekanntem Ziel enthalten es. Ein Datensatz zeigt nur einen Treffer; wie oft sie treffen, zeigt das Raster unten."
    if not hit:
        return "warning", f"{base} Kein Intervall enthält sein Ziel. Das kommt bei 95 %-Intervallen etwa jedes zwanzigste Mal vor, bei Hebelpunkten oder schweren Rändern für Pearson deutlich öfter."
    return "info", f"{base} Es enthalten ihr Ziel: {', '.join(hit)}; es verfehlen es: {', '.join(miss)}."


def verdict_relations(rows: list[dict]) -> tuple[str, str]:
    """E1: Abweichung der beiden Umrechnungen von ρ über alle Zellen."""
    dk = max(rows, key=lambda r: abs(r["rho_from_tau"] - r["rho"]))
    ds = max(rows, key=lambda r: abs(r["rho_from_spearman"] - r["rho"]))
    err_k, err_s = abs(dk["rho_from_tau"] - dk["rho"]), abs(ds["rho_from_spearman"] - ds["rho"])
    text = (f"Die Kendall-Umrechnung sin(πτ/2) weicht in allen {len(rows)} Zellen höchstens um **{err_k:.3f}** von ρ ab, die Spearman-Umrechnung um bis zu **{err_s:.3f}** "
            f"(am meisten bei {C.ALL_FAMILIES[ds['family']]}, ρ = {ds['rho']:g}: {ds['rho_from_spearman']:.3f}).")
    return ("success" if err_k < 0.01 else "info"), text


def verdict_curve(res: dict) -> tuple[str, str]:
    """E2: Mittel von r, ρ_S, τ der Kurve und der Abstand zum Rangmaß."""
    text = (f"Bei c = {res['c']:g}, Rauschen {res['noise']:g}, n = {res['n']} ({res['runs']} Läufe) ist r im Mittel **{res['r']:.2f}** (Streuung {res['r_sd']:.2f}), ρ_S **{res['rho_s']:.2f}**, "
            f"τ **{res['tau']:.2f}**.")
    if res["population_r"] is not None:
        text += f" Die Populationskorrelation ohne Rauschen wäre {res['population_r']:.2f}."
    return ("info" if res["r"] <= res["rho_s"] else "warning"), text


def verdict_lever(rows: list[dict], n: int, k: float) -> tuple[str, str]:
    """E3: Anteil „signifikant“ je Test in der Zelle (n, k)."""
    cell = find(rows, n=n, k=k)
    if k == 0:
        return "info", (f"Ohne Hebelpunkt schlagen bei n = {n} Pearson-t, Spearman-t und Kendall-z in {100 * cell['share_r']:.1f} / {100 * cell['share_s']:.1f} / {100 * cell['share_k']:.1f} % "
                        f"der {cell['runs']} Läufe an (erwartet 5 %).")
    kind = "warning" if cell["share_r"] >= 0.5 else "info"
    return kind, (f"Ein Punkt bei {k:g} Standardabweichungen lässt bei n = {n} den Pearson-Test in **{100 * cell['share_r']:.1f} %**, den Spearman-Test in **{100 * cell['share_s']:.1f} %** "
                  f"und den Kendall-Test in **{100 * cell['share_k']:.1f} %** der {cell['runs']} Läufe anschlagen (mittleres r {cell['mean_r']:.2f}, ρ_S {cell['mean_s']:.2f}, τ {cell['mean_tau']:.2f}).")


def verdict_size(row: dict) -> tuple[str, str]:
    """E4: Fehlalarmrate bei wahrem ρ = 0; Warnung, wenn Pearson mehr als drei Standardfehler über 5 % liegt."""
    se = math.sqrt(0.05 * 0.95 / row["runs"])
    name = C.ALL_FAMILIES[row["family"]]
    text = (f"Bei wahrem ρ = 0 ({name}, n = {row['n']}) schlagen Pearson-t, Spearman-t und Kendall-z in **{100 * row['pearson']:.1f} / {100 * row['spearman']:.1f} / {100 * row['kendall']:.1f} %** "
            f"von {row['runs']} Läufen an.")
    if row["pearson"] > 0.05 + 3 * se:
        return "warning", text + " Pearson ist hier zu oft auffällig, die Rangtests nur geringfügig."
    return "success", text + " Alle drei liegen nahe 5 %."


def verdict_ties(rows: list[dict]) -> tuple[str, str]:
    """E5: Spannweite der Varianten bei der gröbsten Skala gegen den stetigen Wert (letzte Zeile)."""
    first, last = rows[0], rows[-1]
    vals = {"τ-a": first["tau_a"], "τ-b": first["tau_b"], "γ": first["gamma"]}
    lo, hi = min(vals.values()), max(vals.values())
    return "info", (f"Bei {first['levels']} Stufen ergibt dieselbe Skala **{vals['τ-a']:.2f} (τ-a), {vals['τ-b']:.2f} (τ-b) und {vals['γ']:.2f} (γ)**: eine Spanne von {hi - lo:.2f}; "
                    f"bei {last['levels']} Stufen liegen sie zwischen {min(last['tau_a'], last['tau_b'], last['gamma']):.2f} und {max(last['tau_a'], last['tau_b'], last['gamma']):.2f}.")


def verdict_cover(series: list[dict], label: str) -> tuple[str, str]:
    """E6: Abdeckung des Kendall-Bootstrap-Intervalls und des Pearson-Intervalls (gegen ρ) bei der kleinsten und der größten Stichprobe der Reihe."""
    first, last = series[0], series[-1]
    text = (f"{label}, ρ = {last['rho']:g}: Das Kendall-Intervall deckt bei n = {first['n']} zu **{100 * first['k_boot']:.1f} %** und bei n = {last['n']} zu **{100 * last['k_boot']:.1f} %**; "
            f"das Pearson-Intervall (Fisher-z, gegen ρ) {100 * first['pearson']:.1f} % und {100 * last['pearson']:.1f} %.")
    if last["k_boot"] < 0.90:
        return "warning", f"{text} Auch das Kendall-Intervall hält sein Versprechen hier nicht."
    return "success", f"{text} Das Kendall-Intervall hält hier annähernd sein Versprechen."


def verdict_eff(rows: list[dict]) -> tuple[str, str]:
    """E6: Streuung der Schätzer für ρ; genannt wird das Verhältnis Kendall zu Pearson bei der Zeile mit dem kleinsten und der mit dem größten Verhältnis."""
    ratios = [(r["k_sd"] / r["r_sd"], r) for r in rows if r["family"] != "lognormal1"]
    lo, rlo = min(ratios, key=lambda t: t[0])
    hi, rhi = max(ratios, key=lambda t: t[0])
    short = lambda family: C.ALL_FAMILIES[family].split(" (")[0]          # „kontaminiert (5 % …)“ → „kontaminiert“
    return "info", (f"Die Streuung der Kendall-Umrechnung ist **{lo:.2f}-mal** (bei {short(rlo['family'])}, ρ = {rlo['rho']:g}) bis **{hi:.2f}-mal** (bei {short(rhi['family'])}, ρ = {rhi['rho']:g}) "
                    f"so groß wie die von r (n = {rlo['n']}); unter 1 heißt: genauer als Pearson.")
