"""Konstanten: Regler-Grenzen, Raster der Messreihen, Farben und Presets.

Familien, Zufallserzeuger und Pearson-Kern kommen als wörtliche Kopien aus pearson-demo (prs_constants.py, prs_scenario.py, prs_algorithm.py, prs_rng.py).
Komplexität am Regler-Maximum (DEMO-PLAYBOOK §1c): Kendall braucht O(n²) je Datensatz; ein Bootstrap mit B Ziehungen kostet O(B·n²). Bei n = 500 und B = 1 000
sind das 2,5·10^8 Vorzeichenprodukte (rund 2 s); Wiederholungsexperimente stehen vorgerechnet im Raster."""

from prs_constants import ALL_FAMILIES, FAMILIES, INDEPENDENT_T3, LEV_DIRS, LEVEL, SEED_MAX  # noqa: F401  (Wiederverwendung der Kopie)

N_MIN, N_MAX, DEFAULT_N = 5, 500, 30
RHO_MIN, RHO_MAX, DEFAULT_RHO = -0.95, 0.95, 0.5
LEV_COUNT_MAX, LEV_K_MAX = 2, 15.0
DEFAULT_LEV_COUNT, DEFAULT_LEV_K = 0, 5.0
B_MIN, B_MAX, DEFAULT_B = 200, 1000, 400
DEFAULT_SEED = 35
RANK_BUDGET = 3 * 10**8          # B · n² höchstens (Kendall-Bootstrap des Einzeldatensatzes)

# E2 (Kurve mit Rauschen, live)
CURVE_CS = (0.5, 1, 2, 3, 4)
CURVE_N_MIN, CURVE_N_MAX, DEFAULT_CURVE_N = 20, 2000, 100
CURVE_NOISE_MAX, DEFAULT_CURVE_NOISE = 1.0, 0.5
CURVE_RUNS = 200

# Raster (tools/make_grid.py)
GRID_SEEDS = (101, 102, 103, 104)
REL_FAMILIES = ("normal", "t3", "kontaminiert", "lognormal1")      # E1: Beziehungen
REL_RHOS = (0.3, 0.6, 0.9)
REL_N, REL_RUNS = 2000, 40
LEVER_NS = (10, 30, 100, 300)                                       # E3
LEVER_KS = (0, 2, 3, 4, 5, 6, 8, 10, 15)
SIZE_NS = (30, 100, 300)                                            # E4
SIZE_FAMILIES = ("normal", "tindep3", "t3", "t5", "t10", "kontaminiert", "lognormal1")
TIE_LEVELS = (3, 5, 7, 11, 21, 100)                                 # E5 (n = 500, ρ = 0,6)
TIE_N, TIE_RHO = 500, 0.6
COVER_FAMILIES = ("normal", "lognormal1", "t3", "kontaminiert")     # E6: Intervalle
COVER_NS = (10, 20, 30, 50, 100, 300)
COVER_RHOS = (0.0, 0.5, 0.9)
COVER_BOOT = 200
EFF_FAMILIES = ("normal", "t5", "t3", "kontaminiert", "lognormal1")  # E6: Schätzer
EFF_RHOS = (0.3, 0.6, 0.9)
EFF_N = 100

COLOR_POINTS = "#4c78a8"
COLOR_LEVER = "#d95f02"
COLOR_LINE = "#2F6B65"
COLOR_TRUE = "#222222"
COLOR_PEARSON = "#4c78a8"
COLOR_SPEARMAN = "#f58518"
COLOR_KENDALL = "#54a24b"
COLOR_GAMMA = "#b279a2"

PRESETS = {
    "Standardfall (Voreinstellung)": {"family": "normal", "rho": 0.5, "n": 30, "lev_count": 0, "lev_k": 5.0, "seed": DEFAULT_SEED},
    "Ein einzelner Punkt (k = 5, n = 30)": {"family": "normal", "rho": 0.0, "n": 30, "lev_count": 1, "lev_k": 5.0, "seed": DEFAULT_SEED},
    "Spitzenwochen (kontaminiert, n = 100)": {"family": "kontaminiert", "rho": 0.0, "n": 100, "lev_count": 0, "lev_k": 5.0, "seed": DEFAULT_SEED},
    "Schwerer Rand (t(3), n = 100)": {"family": "t3", "rho": 0.5, "n": 100, "lev_count": 0, "lev_k": 5.0, "seed": DEFAULT_SEED},
    "Mengen (Lognormal, n = 100)": {"family": "lognormal1", "rho": 0.5, "n": 100, "lev_count": 0, "lev_k": 5.0, "seed": DEFAULT_SEED},
}
# Die Hilfetexte nennen Zahlen des vorgerechneten Rasters (data/rank_grid.json); tests/test_claims.py prüft jede davon.
PRESET_HELP = {
    "Standardfall (Voreinstellung)": "Normalverteilung, ρ = 0,5, n = 30: Pearson, Spearman und Kendall erzählen dieselbe Geschichte; das Kendall-Intervall deckt in der Messreihe zu 94,7 %.",
    "Ein einzelner Punkt (k = 5, n = 30)": "Unabhängige Daten (ρ = 0) und ein Punkt bei fünf Standardabweichungen, n = 30: der Pearson-Test schlägt in der Messreihe in 84,7 % der Läufe an, der Spearman-Test in 5,8 %, der Kendall-Test in 6,3 %.",
    "Spitzenwochen (kontaminiert, n = 100)": "5 % der Wochen mit 5-facher Streuung, ρ = 0, n = 100: der Pearson-Test schlägt in der Messreihe in 40,9 % der Läufe an statt in 5 %, der Spearman-Test in 6,3 %, der Kendall-Test in 6,9 %.",
    "Schwerer Rand (t(3), n = 100)": "t(3) mit gemeinsamer Skala, ρ = 0,5, n = 100: das Kendall-Intervall deckt in der Messreihe zu 94,4 %, das Fisher-Intervall von Pearson nur zu 69,0 %.",
    "Mengen (Lognormal, n = 100)": "Lognormal σ = 1, ρ = 0,5, n = 100: dieselben Ränge wie bei Normal, aber Pearson misst nur 0,38; das Fisher-Intervall von Pearson deckt ρ zu 59,5 %, das Kendall-Intervall zu 95,5 %.",
}
