"""Plotly-Abbildungen. Alle Achsen sind gesperrt (fixedrange), damit Touch-Geräte beim Scrollen nicht am Diagramm hängen bleiben."""

import numpy as np
import plotly.graph_objects as go

import rkr_algorithm as R
import rkr_constants as C

INTERVAL_COLORS = ["#4c78a8", "#f58518", "#e8a05a", "#54a24b", "#2f7d4b"]


def lock_axes(fig):
    fig.update_xaxes(fixedrange=True)
    fig.update_yaxes(fixedrange=True)
    return fig


def _base(fig, height=380):
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=10, b=10), legend=dict(orientation="h", y=-0.22), plot_bgcolor="rgba(0,0,0,0)")
    return lock_axes(fig)


def _split(values, lever):
    mask = np.zeros(len(values), dtype=bool)
    mask[list(lever)] = True
    return mask


def build_scatter(x, y, lever=(), slope=None, intercept=None, height=340):
    """Streudiagramm der Messwerte mit Regressionsgerade; Hebelpunkte in Orange."""
    x, y = np.asarray(x), np.asarray(y)
    mask = _split(x, lever)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x[~mask], y=y[~mask], mode="markers", name="Beobachtungen", marker=dict(color=C.COLOR_POINTS, size=7, opacity=0.8),
                             hovertemplate="x = %{x:.2f}, y = %{y:.2f}<extra></extra>"))
    if mask.any():
        fig.add_trace(go.Scatter(x=x[mask], y=y[mask], mode="markers", name="Hebelpunkt", marker=dict(color=C.COLOR_LEVER, size=11, symbol="diamond"),
                                 hovertemplate="x = %{x:.2f}, y = %{y:.2f}<extra></extra>"))
    if slope is not None:
        xs = np.array([x.min(), x.max()])
        fig.add_trace(go.Scatter(x=xs, y=intercept + slope * xs, mode="lines", name="Regressionsgerade", line=dict(color=C.COLOR_LINE, width=2.5), hoverinfo="skip"))
    fig.update_xaxes(title_text="x: Abweichung des Auftragsvolumens vom Plan")
    fig.update_yaxes(title_text="y: Überstunden")
    return _base(fig, height)


def build_rank_scatter(x, y, lever=(), height=340):
    """Dieselben Punkte als Ränge (Durchschnittsränge): was Spearman und Kendall sehen."""
    rx, ry = R.rank_rows(x)[0], R.rank_rows(y)[0]
    mask = _split(rx, lever)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=rx[~mask], y=ry[~mask], mode="markers", name="Beobachtungen", marker=dict(color=C.COLOR_SPEARMAN, size=7, opacity=0.8),
                             hovertemplate="Rang x = %{x:.1f}, Rang y = %{y:.1f}<extra></extra>"))
    if mask.any():
        fig.add_trace(go.Scatter(x=rx[mask], y=ry[mask], mode="markers", name="Hebelpunkt", marker=dict(color=C.COLOR_LEVER, size=11, symbol="diamond"),
                                 hovertemplate="Rang x = %{x:.1f}, Rang y = %{y:.1f}<extra></extra>"))
    fig.update_xaxes(title_text="Rang von x")
    fig.update_yaxes(title_text="Rang von y")
    return _base(fig, height)


def build_intervals(intervals, estimates, targets):
    """Fünf Intervalle als waagerechte Balken; Punkt: Schätzer, Raute: Zielwert (nur wo er bekannt ist)."""
    names = list(intervals)
    fig = go.Figure()
    for color, name in zip(INTERVAL_COLORS, names):
        lo, hi = intervals[name]
        fig.add_trace(go.Scatter(x=[lo, hi], y=[name, name], mode="lines+markers", line=dict(color=color, width=5), marker=dict(symbol="line-ns-open", size=14, color=color),
                                 showlegend=False, hovertemplate=f"{name}: %{{x:.3f}}<extra></extra>"))
    fig.add_trace(go.Scatter(x=[estimates[n] for n in names], y=names, mode="markers", name="Schätzer", marker=dict(color=C.COLOR_TRUE, size=9), hovertemplate="Schätzer %{x:.3f}<extra></extra>"))
    known = [n for n in names if targets[n] is not None]
    fig.add_trace(go.Scatter(x=[targets[n] for n in known], y=known, mode="markers", name="Zielwert", marker=dict(color="#d62728", size=11, symbol="diamond-open", line=dict(width=2)),
                             hovertemplate="Ziel %{x:.3f}<extra></extra>"))
    fig.update_xaxes(title_text="Korrelation", range=[-1.05, 1.05])
    fig.update_yaxes(autorange="reversed", categoryorder="array", categoryarray=names)
    return _base(fig, 300)


def build_influence(infl_r, infl_s):
    """r − r ohne Punkt i und ρ_S − ρ_S ohne Punkt i (zwei Balken je Beobachtung)."""
    idx = list(range(1, len(infl_r) + 1))
    fig = go.Figure()
    fig.add_trace(go.Bar(x=idx, y=list(infl_r), name="Pearson r", marker_color=C.COLOR_PEARSON, hovertemplate="Beobachtung %{x}: %{y:+.3f}<extra>r</extra>"))
    fig.add_trace(go.Bar(x=idx, y=list(infl_s), name="Spearman ρ_S", marker_color=C.COLOR_SPEARMAN, hovertemplate="Beobachtung %{x}: %{y:+.3f}<extra>ρ_S</extra>"))
    fig.update_layout(barmode="group")
    fig.update_xaxes(title_text="Beobachtung", type="category" if len(infl_r) <= 40 else "linear")
    fig.update_yaxes(title_text="Änderung beim Weglassen")
    return _base(fig, 300)


def build_curve(results):
    """Mittleres r, ρ_S und τ der Kurve e^(c·x) mit Rauschen für mehrere c (gruppierte Balken)."""
    labels = [f"c = {r['c']:g}" for r in results]
    fig = go.Figure()
    fig.add_trace(go.Bar(x=labels, y=[r["r"] for r in results], name="Pearson r", marker_color=C.COLOR_PEARSON))
    fig.add_trace(go.Bar(x=labels, y=[r["rho_s"] for r in results], name="Spearman ρ_S", marker_color=C.COLOR_SPEARMAN))
    fig.add_trace(go.Bar(x=labels, y=[r["tau"] for r in results], name="Kendall τ", marker_color=C.COLOR_KENDALL))
    fig.update_layout(barmode="group")
    fig.update_yaxes(title_text="Mittel über die Läufe", range=[0, 1.02])
    return _base(fig, 320)


def build_lever(rows, n):
    """Anteil „signifikant“ gegen die Lage k, je ein Test eine Linie (für ein n)."""
    sub = sorted((r for r in rows if r["n"] == n), key=lambda r: r["k"])
    fig = go.Figure()
    for key, name, color in (("share_r", "Pearson-t", C.COLOR_PEARSON), ("share_s", "Spearman-t", C.COLOR_SPEARMAN), ("share_k", "Kendall-z", C.COLOR_KENDALL)):
        fig.add_trace(go.Scatter(x=[r["k"] for r in sub], y=[100 * r[key] for r in sub], mode="lines+markers", name=name, line=dict(color=color, width=3),
                                 hovertemplate="k = %{x}: %{y:.1f} %<extra>" + name + "</extra>"))
    fig.add_hline(y=5, line=dict(color=C.COLOR_TRUE, dash="dash", width=1.5))
    fig.update_xaxes(title_text="Lage des Punktes k (Standardabweichungen)")
    fig.update_yaxes(title_text="Anteil „signifikant“ (%)", range=[0, 102])
    return _base(fig)


def build_size(rows):
    """Fehlalarmrate bei wahrem ρ = 0 je Familie für die drei Tests."""
    labels = [C.ALL_FAMILIES[r["family"]].split(",")[0].split(" (")[0] if r["family"] != "tindep3" else "t(3) unabhängig" for r in rows]
    fig = go.Figure()
    for key, name, color in (("pearson", "Pearson-t", C.COLOR_PEARSON), ("spearman", "Spearman-t", C.COLOR_SPEARMAN), ("kendall", "Kendall-z", C.COLOR_KENDALL)):
        fig.add_trace(go.Bar(x=labels, y=[100 * r[key] for r in rows], name=name, marker_color=color))
    fig.add_hline(y=5, line=dict(color=C.COLOR_TRUE, dash="dash", width=1.5))
    fig.update_layout(barmode="group")
    fig.update_yaxes(title_text="Fehlalarm bei wahrem ρ = 0 (%)")
    return _base(fig)


def build_ties(rows):
    """Die Varianten gegen die Zahl der Skalenstufen."""
    labels = [str(r["levels"]) for r in rows]
    fig = go.Figure()
    for key, name, color in (("tau_a", "τ-a", "#9ecae1"), ("tau_b", "τ-b", C.COLOR_KENDALL), ("gamma", "γ", C.COLOR_GAMMA), ("rho_s", "ρ_S (Durchschnittsränge)", C.COLOR_SPEARMAN)):
        fig.add_trace(go.Scatter(x=labels, y=[r[key] for r in rows], mode="lines+markers", name=name, line=dict(color=color, width=3), hovertemplate="%{x} Stufen: %{y:.3f}<extra>" + name + "</extra>"))
    fig.update_xaxes(title_text="Zahl der Skalenstufen", type="category")
    fig.update_yaxes(title_text="Mittel über die Läufe")
    return _base(fig)


def build_cover(series, family_label):
    """Abdeckung gegen n (Raster): Kendall-Intervall, Spearman (Bonett–Wright, Bootstrap; nur bei Normalkopula) und Pearson-Fisher gegen ρ."""
    ns = [r["n"] for r in series]
    fig = go.Figure()
    specs = [("k_boot", "Kendall (Bootstrap)", C.COLOR_KENDALL), ("pearson", "Pearson (Fisher-z) gegen ρ", C.COLOR_PEARSON)]
    if series and series[0]["has_spearman_target"]:
        specs = [("s_bw", "Spearman (Bonett–Wright)", C.COLOR_SPEARMAN), ("s_boot", "Spearman (Bootstrap)", "#e8a05a")] + specs
    for key, name, color in specs:
        fig.add_trace(go.Scatter(x=ns, y=[100 * r[key] for r in series], mode="lines+markers", name=name, line=dict(color=color, width=3),
                                 error_y=dict(type="data", array=[100 * r[key + "_sd"] for r in series], visible=True, thickness=1), hovertemplate="n = %{x}: %{y:.1f} %<extra>" + name + "</extra>"))
    fig.add_hline(y=95, line=dict(color=C.COLOR_TRUE, dash="dash", width=1.5))
    fig.update_xaxes(title_text=f"Stichprobengröße n ({family_label})", type="log", tickvals=ns, ticktext=[str(n) for n in ns])
    fig.update_yaxes(title_text="Abdeckung (%)", range=[40, 100])
    return _base(fig)


def build_eff(rows):
    """Streuung der drei Schätzer für ρ je Familie (gruppierte Balken, kleiner ist besser)."""
    labels = [C.ALL_FAMILIES[r["family"]].split(",")[0].split(" (")[0] for r in rows]
    fig = go.Figure()
    for key, name, color in (("r_sd", "Pearson r", C.COLOR_PEARSON), ("s_sd", "2 sin(πρ_S/6)", C.COLOR_SPEARMAN), ("k_sd", "sin(πτ/2)", C.COLOR_KENDALL)):
        fig.add_trace(go.Bar(x=labels, y=[r[key] for r in rows], name=name, marker_color=color))
    fig.update_layout(barmode="group")
    fig.update_yaxes(title_text="Standardabweichung des Schätzers")
    return _base(fig)
