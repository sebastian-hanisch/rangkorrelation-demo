"""Rangkorrelation: Spearman, Kendall und was Ränge ändern - interaktive Konzept-Demo
Sebastian Hanisch - Operations Research und Machine Learning

Zweites Stück der Korrelations-Reihe. Dieselben simulierten Standort-Wochen wie im ersten Stück (bekannte wahre Korrelation): Spearman-ρ und Kendall-τ als Pearson der Ränge
und als Paarzählung, ihre Umrechnungen, Tests und Intervalle, und was sich bei Hebelpunkten, gemeinsamer Streuung, monotoner Kurve und groben Skalen ändert.

Lauffähig mit: streamlit run app.py
"""

import numpy as np
import streamlit as st

import rkr_algorithm as R
import rkr_constants as C
import rkr_evaluation as ev
from rkr_layout import dunkles_design, kopf_css
from rkr_pdf_export import build_pdf
from rkr_presets import (
    bounds,
    init_session_state_defaults,
    load_permalink_settings,
    randomize_seed,
    sidebar_presets,
    sync_query_params,
)
from rkr_visualization import (
    build_cover,
    build_curve,
    build_eff,
    build_influence,
    build_intervals,
    build_lever,
    build_rank_scatter,
    build_scatter,
    build_size,
    build_ties,
)

st.set_page_config(page_title="Rangkorrelation – Sebastian Hanisch", layout="wide")


@st.cache_data(show_spinner=False)
def _analysis(settings):
    return ev.analyse(settings)


@st.cache_data(show_spinner=False)
def _grid():
    return ev.load_grid()


# Fester Kopf: nur der Titel bleibt beim Scrollen oben (Stylesheet in rkr_layout.py).
st.html(kopf_css(dunkles_design()))
with st.container(key="kopf"):
    st.title("🔢 Rangkorrelation: Spearman, Kendall und was Ränge ändern")
st.markdown(
    """
**Zweites Stück der Korrelations-Reihe** und **die erste Demo des Portfolios mit Rangmaßen als Thema**. Dieselben simulierten Standort-Wochen wie im ersten Stück
(*Pearson: eine Zahl, viele Daten*): das **Auftragsvolumen** gegen den Plan und die **Überstunden**, die **wahre Korrelation ist bekannt**. Das erste Stück zeigte, wann r kippt: ein
**einzelner Punkt**, **schwere Ränder**, **gemeinsame Streuung**. Hier die Frage, ob **Ränge** das beheben, und was sie kosten: **Spearman-ρ** ist die Korrelation der Ränge,
**Kendall-τ** zählt Paare.

Weiter unten: sechs Experimente (**🔬**), **🚧 Wo die Annahmen enden** und der Expander **📐 Mathematische Formulierung**. Der Abstand zweier **Rangfolgen** (Turniertabellen) steht in
`rangdistanz-demo`, Ursachen in der Linie *Kausale Diagramme und Adjustierung*, nichtmonotone Zusammenhänge im nächsten Stück.
"""
)

with st.expander("So funktioniert die Rangkorrelation", expanded=True):
    st.markdown(
        """
1. **Ränge:** jeden Wert durch seinen Platz in der Reihe ersetzen (kleinster Wert = 1); gleiche Werte bekommen das Mittel ihrer Plätze.
2. **Spearman-ρ:** die Pearson-Korrelation der beiden Rangreihen. Sie misst, wie gut eine **monoton** steigende (oder fallende) Beziehung passt, nicht die Geradlinigkeit.
3. **Kendall-τ:** alle Paare (i, j) vergleichen: **konkordant** (beide Werte steigen), **diskordant** (einer steigt, einer fällt); τ = (C − D) / Zahl der Paare. Bei Gleichständen gibt es Varianten.
4. **Umrechnung auf ρ:** bei elliptischen Verteilungen (Normal, t(ν), kontaminiert) gilt τ = 2/π · arcsin ρ, also ρ = sin(πτ/2); bei der Normalverteilung auch ρ = 2 sin(πρ_S/6).

| Variante bei Gleichständen | Rechnung | Wirkung |
|---|---|---|
| **τ-a** | (C − D) / alle Paare | Gleichstände zählen als „kein Zusammenhang“: τ schrumpft |
| **τ-b** | (C − D) / √((Paare − T_x)(Paare − T_y)) | Gleichstände aus dem Nenner nehmen |
| **τ-c** | 2(C − D) / (n² (m − 1)/m), m = kleinere Stufenzahl | für Tabellen unterschiedlicher Größe |
| **γ** | (C − D) / (C + D) | ignoriert alle Gleichstände, wird bei groben Skalen groß |
        """
    )

st.caption("🔗 Die Adresszeile oben spiegelt Ihre aktuelle Konfiguration wider – einfach kopieren, um ein Szenario zu teilen.")

load_permalink_settings()
init_session_state_defaults()

with st.sidebar:
    sidebar_presets()  # Schnellstart oben, das aktive Preset ist hervorgehoben
    st.header("⚙️ Einstellungen")
    st.markdown("**Daten**")
    family = st.selectbox("Verteilung", list(C.FAMILIES), format_func=C.FAMILIES.get, key="family_select",
                          help="Die Familie der Paare. t(ν) und kontaminiert: beide Kennzahlen teilen sich einen Skalenfaktor (Spitzenwochen heben beide zugleich). Lognormal: Mengen, schief; Ränge sind dieselben wie bei Normal.")
    rho = st.slider("Wahre Korrelation ρ (Parameter)", *bounds("rho_slider"), step=0.05, key="rho_slider",
                    help="Parameter der Normalfelder. Bei Lognormal weicht die Pearson-Korrelation davon ab, Kendall-τ und Spearman-ρ nicht.")
    n = st.slider("Beobachtungen n", *bounds("n_slider"), key="n_slider", help="Standort-Wochen im Datensatz (höchstens 500: Kendall zählt alle Paare).")
    st.markdown("**Hebelpunkte**")
    lev_count = st.slider("Zahl der Hebelpunkte", *bounds("lev_count_slider"), key="lev_count_slider",
                          help="Ersetzt so viele der ersten Beobachtungen durch weit entfernte Punkte (mindestens drei Beobachtungen bleiben unverändert).")
    lev_k = st.slider("Lage k (Standardabweichungen)", *bounds("lev_k_slider"), step=0.5, key="lev_k_slider", disabled=lev_count == 0,
                      help="Wie weit der Punkt vom Rest liegt, in Standardabweichungen der übrigen Daten. Wirkt nur mit mindestens einem Hebelpunkt.")
    lev_dir = st.selectbox("Richtung", list(C.LEV_DIRS), key="lev_dir_select", disabled=lev_count == 0,
                           help="gleichläufig: Punkt bei (k, k), er hebt r; gegenläufig: (k, −k), er senkt r.")
    st.markdown("**Verfahren**")
    boot_b = st.slider("Bootstrap-Ziehungen B", *bounds("boot_slider"), step=100, key="boot_slider",
                       help="Wie oft die Paare neu gezogen werden. Bei großem n begrenzt das Budget B · n² die Zahl (Kendall zählt in jeder Ziehung alle Paare).")
    seed = st.number_input("Zufalls-Seed", *bounds("seed_input"), key="seed_input", step=1)
    st.button("🎲 Neuen Datensatz ziehen", width="stretch", on_click=randomize_seed)

sync_query_params({"family_select": family, "rho_slider": rho, "n_slider": int(n), "lev_count_slider": int(lev_count), "lev_k_slider": lev_k, "lev_dir_select": lev_dir,
                   "boot_slider": int(boot_b), "seed_input": int(seed)})

settings = ev.Settings(family, round(float(rho), 2), int(n), int(lev_count), float(lev_k), lev_dir, int(boot_b), int(seed))
a = _analysis(settings)
x_arr, y_arr = np.array(a.x), np.array(a.y)
slope = float(np.polyfit(x_arr, y_arr, 1)[0])
intercept = float(y_arr.mean() - slope * x_arr.mean())

# --- Ergebnis ------------------------------------------------------------------------------------------------------------------------------

st.markdown("## 🎯 Ergebnis")
m1, m2, m3 = st.columns(3)
m1.metric("Pearson r", f"{a.r:.2f}", delta=f"{a.r - a.true_r:+.2f} gegen {a.true_r:.2f}", delta_color="off")
m2.metric("Spearman ρ_S", f"{a.rho_s:.2f}", delta=(f"{a.rho_s - a.rs_true:+.2f} gegen {a.rs_true:.2f}" if a.rs_true is not None else "Ziel unbekannt (keine Normalkopula)"), delta_color="off")
m3.metric("Kendall τ-b", f"{a.tau:.2f}", delta=f"{a.tau - a.tau_true:+.2f} gegen {a.tau_true:.2f}", delta_color="off")
st.caption(f"p-Werte für ρ = 0 (Näherungen): Pearson-t {a.p_pearson:.3f} · Spearman-t {a.p_spearman:.3f} · Kendall-z {a.p_kendall:.3f}. Die Rangtests setzen bei kleinem n und Gleichständen nur näherungsweise "
           "die Normal- bzw. t-Verteilung voraus.")
left, right = st.columns(2)
with left:
    st.plotly_chart(build_scatter(a.x, a.y, a.lever, slope, intercept), width="stretch", key="main_scatter")
    st.caption("Messwerte: Pearson sieht die Geradlinigkeit (grüne Gerade).")
with right:
    st.plotly_chart(build_rank_scatter(a.x, a.y, a.lever), width="stretch", key="rank_scatter")
    st.caption("Dieselben Punkte als Ränge: Spearman und Kendall sehen nur diese Ordnung.")
estimates = {INT: v for INT, v in zip(ev.INTERVALS, (a.r, a.rho_s, a.rho_s, a.tau, R.rho_from_tau(a.tau)))}
st.plotly_chart(build_intervals(a.intervals, estimates, a.targets), width="stretch", key="main_intervals")
st.caption("Fünf 95 %-Intervalle; schwarzer Punkt: Schätzer, rote Raute: Zielwert (Pearson: was r erwartungsgemäß misst; Spearman: nur bei Normalkopula bekannt; Kendall: τ; unten: ρ). Die Ziele der Zeilen sind verschieden: "
           "gelesen wird, ob die Raute im Balken liegt.")
if st.button("📄 PDF erstellen", key="pdf_make"):
    st.session_state["pdf_bytes"] = (settings, build_pdf(a))
pdf = st.session_state.get("pdf_bytes")
if pdf and pdf[0] == settings:
    st.download_button("⬇️ PDF herunterladen", pdf[1], file_name="rangkorrelation-demo.pdf", mime="application/pdf", key="pdf_download")

st.markdown("## ❓ Kernfrage: Wo liegen die Zielwerte, und wer fängt sie?")
kind, text = ev.verdict_main(a)
getattr(st, kind)(text)
st.plotly_chart(build_influence(a.infl_r, a.infl_s), width="stretch", key="influence_chart")
top_r, top_s = int(np.argmax(np.abs(a.infl_r))), int(np.argmax(np.abs(a.infl_s)))
st.caption(f"Einfluss jeder Beobachtung: um wie viel sich r bzw. ρ_S ändert, wenn sie fehlt. Der größte Einfluss auf r liegt bei Nummer {top_r + 1} ({a.infl_r[top_r]:+.3f}), "
           f"der auf ρ_S bei Nummer {top_s + 1} ({a.infl_s[top_s]:+.3f}).")

st.markdown("---")

grid = _grid()
if grid is None:
    st.warning("Das vorgerechnete Raster (data/rank_grid.json) fehlt; es entsteht mit `python tools/make_grid.py`. Die Experimente 1 und 3 bis 6 zeigen es.")

# --- E1 Beziehungen und Umrechnungen -------------------------------------------------------------------------------------------------------

st.subheader("🔬 Experiment 1: Zwei Zahlen, eine Wahrheit? Beziehungen und Umrechnungen")
st.caption("Mittel über 40 Stichproben mit n = 2 000 je Zelle. Formeln: ρ_S = 6/π · arcsin(ρ/2) (Normalkopula), τ = 2/π · arcsin ρ (elliptisch).")
if grid is not None:
    rel = sorted(grid["rel"], key=lambda r: (C.REL_FAMILIES.index(r["family"]), r["rho"]))
    tab = ["| Verteilung | ρ | r | ρ_S | Formel ρ_S | τ | Formel τ | ρ aus τ | ρ aus ρ_S |", "|---|---|---|---|---|---|---|---|---|"]
    for r in rel:
        tab.append(f"| {C.ALL_FAMILIES[r['family']].split(',')[0]} | {r['rho']:g} | {r['r']:.3f} | {r['rho_s']:.3f} | {r['rho_s_formula']:.3f} | {r['tau']:.3f} | {r['tau_formula']:.3f} | "
                   f"{r['rho_from_tau']:.3f} | {r['rho_from_spearman']:.3f} |")
    st.markdown("\n".join(tab))
    kind, text = ev.verdict_relations(rel)
    getattr(st, kind)(text)
    st.caption("Lognormal hat dieselben Ränge wie Normal (Ränge sehen die Ränder nicht), aber ein ganz anderes r. Bei t(3) und kontaminiert gilt die τ-Formel, die ρ_S-Formel nicht.")

st.markdown("---")

# --- E2 Monotone Kurve ---------------------------------------------------------------------------------------------------------------------

st.subheader("🔬 Experiment 2: Eine monotone Kurve")
st.caption("y = e^(c·x) plus Rauschen, x standardnormal. Spearman und Kendall kennen nur die Reihenfolge; Pearson hängt an c, am Rauschen und an n.")
c1, c2 = st.columns(2)
curve_noise = c1.slider("Rauschen (Standardabweichung)", 0.0, C.CURVE_NOISE_MAX, C.DEFAULT_CURVE_NOISE, step=0.25, key="curve_noise")
curve_n = c2.slider("n der Kurve", C.CURVE_N_MIN, C.CURVE_N_MAX, C.DEFAULT_CURVE_N, step=20, key="curve_n")
if st.button("🔁 Kurve für c = 0,5 bis 4 rechnen", key="btn_curve"):
    with st.spinner("Rechne die Kurven …"):
        st.session_state["curve_result"] = ((curve_noise, curve_n), [ev.curve_experiment(c, curve_noise, curve_n, C.CURVE_RUNS, 7 + i) for i, c in enumerate(C.CURVE_CS)])
curve = st.session_state.get("curve_result")
if curve and curve[0] == (curve_noise, curve_n):
    st.plotly_chart(build_curve(curve[1]), width="stretch", key="curve_chart")
    pick = st.selectbox("c für das Verdikt", list(C.CURVE_CS), index=2, key="curve_pick")
    res = next(r for r in curve[1] if r["c"] == pick)
    kind, text = ev.verdict_curve(res)
    getattr(st, kind)(text)
    if res["runs"] < C.CURVE_RUNS:
        st.caption(f"Das Budget erlaubt bei n = {res['n']} nur {res['runs']} von {C.CURVE_RUNS} Läufen.")
    st.caption("Ohne Rauschen sind ρ_S und τ gleich 1; die Populationskorrelation von Pearson ist c/√(e^(c²) − 1) (0,94 / 0,76 / 0,27 / 0,03 für c = 0,5 / 1 / 2 / 3): r streut bei großem c stark und fällt mit n.")

st.markdown("---")

# --- E3 Hebelpunkt -------------------------------------------------------------------------------------------------------------------------

st.subheader("🔬 Experiment 3: Ein einzelner Punkt")
st.caption("n unabhängige Standardnormalpaare, ein Punkt bei (k, k). Anteil der Läufe, in denen der Test „signifikant“ meldet (vorgerechnet: 4 · 1 000 Läufe je Zelle, n = 300: 4 · 500).")
if grid is not None:
    lever_rows = grid["lever"]
    sel_n = st.selectbox("n", list(C.LEVER_NS), index=1, key="lev_n")
    st.plotly_chart(build_lever(lever_rows, sel_n), width="stretch", key="lever_chart")
    sel_k = st.selectbox("k für das Verdikt", list(C.LEVER_KS), index=C.LEVER_KS.index(5), key="lev_k_sel")
    kind, text = ev.verdict_lever(lever_rows, sel_n, sel_k)
    getattr(st, kind)(text)
    st.caption("Ein Punkt kann bei Rängen nur begrenzt ziehen: ab etwa k = 4 ändert sich der Anteil der Rangtests nicht mehr.")

st.markdown("---")

# --- E4 Gemeinsame Streuung ----------------------------------------------------------------------------------------------------------------

st.subheader("🔬 Experiment 4: Unkorreliert, aber nicht unabhängig")
st.markdown(
    "Im ersten Stück schlug der Test auf ρ = 0 bei gemeinsamer Streuung (Spitzenwochen) viel zu oft an. Wie schlagen sich die Rangtests?"
)
if grid is not None:
    sn = st.selectbox("n", list(C.SIZE_NS), index=1, key="size_n")
    size_rows = [r for f in C.SIZE_FAMILIES if (r := ev.find(grid["size"], family=f, n=sn))]
    st.plotly_chart(build_size(size_rows), width="stretch", key="size_chart")
    sfam = st.selectbox("Verteilung für das Verdikt", list(C.SIZE_FAMILIES), format_func=C.ALL_FAMILIES.get, index=2, key="size_family")
    kind, text = ev.verdict_size(ev.find(grid["size"], family=sfam, n=sn))
    getattr(st, kind)(text)
    st.caption("Die Rangtests sind nicht exakt 5 %: bei t(3) liegen sie bei 6 bis 8 %. Ränge ändern die Verteilung der Statistik, weil die Streuung nicht mehr in die Gewichte eingeht.")

st.markdown("---")

# --- E5 Gleichstände -----------------------------------------------------------------------------------------------------------------------

st.subheader("🔬 Experiment 5: Gleichstände und grobe Skalen")
st.caption("Eine Skala mit k Stufen aus bivariater Normalverteilung (ρ = 0,6, n = 500). Stetig wären τ = 0,410, ρ_S = 0,582 und r = 0,600.")
if grid is not None:
    tie_rows = sorted(grid["ties"], key=lambda r: r["levels"])
    st.plotly_chart(build_ties(tie_rows), width="stretch", key="ties_chart")
    tab = ["| Stufen | τ-a | τ-b | τ-c | γ | ρ_S (Durchschnittsränge) | Pearson der Codes |", "|---|---|---|---|---|---|---|"]
    tab += [f"| {r['levels']} | {r['tau_a']:.3f} | {r['tau_b']:.3f} | {r['tau_c']:.3f} | {r['gamma']:.3f} | {r['rho_s']:.3f} | {r['pearson_codes']:.3f} |" for r in tie_rows]
    st.markdown("\n".join(tab))
    kind, text = ev.verdict_ties(tie_rows)
    getattr(st, kind)(text)
    st.caption("τ-c ist hier gleich τ-b, weil beide Skalen gleich viele Stufen haben. Dass ρ_S und der Pearson der Codes übereinstimmen, gilt für diese Skala, nicht allgemein.")

st.markdown("---")

# --- E6 Intervalle und Preis ---------------------------------------------------------------------------------------------------------------

st.subheader("🔬 Experiment 6: Intervalle für Rangmaße und der Preis in Genauigkeit")
st.markdown(
    "**Abdeckung** der 95 %-Intervalle (Bootstrap mit 200 Ziehungen, vier Seeds, Fehlerbalken: Streuung über die Seeds). Bonett–Wright ist das Fisher-Intervall mit dem Standardfehler "
    "√((1 + ρ_S²/2)/(n − 3)). Spearman-Ziele sind nur bei Normalkopula bekannt (Normal, Lognormal); τ und ρ (über sin(πτ/2)) bei allen Familien."
)
if grid is not None:
    g1, g2 = st.columns(2)
    cfam = g1.selectbox("Verteilung", list(C.COVER_FAMILIES), format_func=C.ALL_FAMILIES.get, index=2, key="cover_family")
    crho = g2.selectbox("ρ", list(C.COVER_RHOS), index=1, key="cover_rho")
    series = ev.cover_series(grid, cfam, crho)
    st.plotly_chart(build_cover(series, C.ALL_FAMILIES[cfam].split(",")[0]), width="stretch", key="cover_chart")
    kind, text = ev.verdict_cover(series, C.ALL_FAMILIES[cfam])
    getattr(st, kind)(text)
    st.markdown("**Preis in Genauigkeit:** Streuung der drei Schätzer für ρ bei n = 100 (4 · 1 000 Läufe).")
    erho = st.selectbox("ρ für die Schätzer", list(C.EFF_RHOS), index=1, key="eff_rho")
    eff_rows = [r for f in C.EFF_FAMILIES if (r := ev.find(grid["eff"], family=f, rho=erho))]
    st.plotly_chart(build_eff(eff_rows), width="stretch", key="eff_chart")
    tab = ["| Verteilung | Pearson r: Mittel / sd | 2 sin(πρ_S/6): Mittel / sd | sin(πτ/2): Mittel / sd |", "|---|---|---|---|"]
    tab += [f"| {C.ALL_FAMILIES[r['family']].split(',')[0]} | {r['r_mean']:.3f} / {r['r_sd']:.3f} | {r['s_mean']:.3f} / {r['s_sd']:.3f} | {r['k_mean']:.3f} / {r['k_sd']:.3f} |" for r in eff_rows]
    st.markdown("\n".join(tab))
    kind, text = ev.verdict_eff(eff_rows)
    getattr(st, kind)(text)
    st.caption(f"Das wahre ρ ist {erho:g}. Pearson misst bei Lognormal nicht ρ (Mittel weicht ab); die Zeile zählt im Verdikt nicht. Die Spearman-Umrechnung ist bei t(3) und kontaminiert verschoben.")

st.markdown("---")

# --- Grenzen -------------------------------------------------------------------------------------------------------------------------------

st.subheader("🚧 Wo die Annahmen enden")
st.markdown(
    """
| Annahme | Was passiert, wenn sie verletzt ist | Wer setzt an |
|---|---|---|
| **Monotonie:** Ränge kennen nur die Reihenfolge | eine Parabel oder ein Kreis hat ρ_S ≈ 0 trotz Abhängigkeit; Spearman ist dort blind | nichtlineare Abhängigkeit (nächstes Stück) |
| **Ränge helfen immer** | bei einer schwach gekrümmten Kurve mit Rauschen liegt ρ_S unter Pearson (Experiment 2, c = 0,5) | – |
| **Rangtests sind exakt** | die Näherungen (t, Normal) sind bei kleinem n und Gleichständen ungenau; bei gemeinsamer Streuung liegen sie bei 6 bis 8 % (Experiment 4) | exakte und Permutationstests |
| **Eine Variante genügt** | bei groben Skalen unterscheiden sich τ-a, τ-b und γ um mehr als die Hälfte (Experiment 5) | Skala und Variante vorher festlegen |
| **Die Umrechnung gilt überall** | 2 sin(πρ_S/6) ist nur bei Normalkopula richtig, sin(πτ/2) bei elliptischen Verteilungen (Experiment 1) | Kopula-Stück (Randabhängigkeit) |
| **Ränge sehen alles Wichtige** | Ränge schneiden die Ränder ab: zwei Datensätze mit gleichen Rängen sind für Spearman gleich, auch wenn die Extreme verschieden sind | Randabhängigkeit (Kopula) |
| **Beobachtungen sind unabhängig** | Reihen in der Zeit verletzen das | Scheinkorrelation in Zeitreihen |
| **Ursachen** | Rangmaße sagen nichts darüber, welche Kennzahl die andere bewegt | Linie *Kausale Diagramme und Adjustierung* |
"""
)
st.caption("Familien und Zahlen sind simuliert; der gemeinsame Skalenfaktor ist eine Modellwahl für Spitzenwochen, kein Befund über echte Standorte. Die Formel von Bonett–Wright ist hier nur durch die Messungen gestützt.")

st.markdown("---")

with st.expander("📐 Mathematische Formulierung"):
    st.markdown(
        r"""
**Lesehilfe.** Gegeben sind $n$ Paare $(x_i, y_i)$; gesucht ist die Stärke des monotonen Zusammenhangs und ein Intervall für $\rho$, $\rho_S$ oder $\tau$.

| Symbol | Bedeutung |
|---|---|
| $R_i, S_i$ | Ränge von $x_i$ und $y_i$ (bei Gleichständen das Mittel der Plätze) |
| $\rho_S$ | Spearman-Korrelation |
| $\tau$ | Kendall-Korrelation |
| $C, D$ | Zahl konkordanter und diskordanter Paare |
| $T_x, T_y$ | Zahl der Paare mit Gleichstand in $x$ bzw. $y$ |
| $\rho$ | Parameter der Normalfelder (Pearson-Korrelation bei Normalverteilung) |

**Spearman.**

$$ \rho_S = \frac{\sum_i (R_i-\bar R)(S_i-\bar S)}{\sqrt{\sum_i (R_i-\bar R)^2 \,\sum_i (S_i-\bar S)^2}} $$

**Kendall (Varianten).** Mit $n_0 = n(n-1)/2$:

$$ \tau_a = \frac{C-D}{n_0}, \qquad \tau_b = \frac{C-D}{\sqrt{(n_0-T_x)(n_0-T_y)}}, \qquad \gamma = \frac{C-D}{C+D} $$

**Beziehung zur Pearson-Korrelation** (elliptische Verteilung bzw. Normalverteilung):

$$ \tau = \frac{2}{\pi}\arcsin\rho, \qquad \rho_S = \frac{6}{\pi}\arcsin\frac{\rho}{2} \;\;(\text{nur Normalkopula}) $$

**Umrechnung auf $\rho$.**

$$ \hat\rho_K = \sin\frac{\pi\hat\tau}{2}, \qquad \hat\rho_S = 2\sin\frac{\pi\hat\rho_S}{6} \;\;(\text{nur Normalkopula}) $$

**Näherungstests unter Unabhängigkeit** (ohne Gleichstandskorrektur):

$$ t = \rho_S\sqrt{\frac{n-2}{1-\rho_S^2}} \sim t_{n-2}, \qquad z = \tau\sqrt{\frac{9n(n-1)}{2(2n+5)}} \sim N(0,1) $$

**Bonett–Wright-Intervall** (Näherung, hier durch Messung gestützt):

$$ \tanh\!\left(\operatorname{artanh}\rho_S \pm \frac{1{,}96\sqrt{1+\rho_S^2/2}}{\sqrt{n-3}}\right) $$

**Quellen.** In dieser Fassung sind keine Literaturangaben ausgewiesen; die Formeln stützen die Messungen der Experimente.
        """
    )

st.markdown("---")
st.caption(
    "Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – "
    "Operations Research und Machine Learning ([Über mich](https://sebastianhanisch.net/ueber-mich.html)). "
    "Mehr zu den Demos: [Interaktive Demos](https://sebastianhanisch.net/demos.html)."
)
