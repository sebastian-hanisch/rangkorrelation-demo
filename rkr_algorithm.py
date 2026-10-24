"""Kern: Ränge mit Gleichständen, Spearman-ρ, Kendall-τ (a, b, c) und Goodman-Kruskal-γ, ihre Tests, Intervalle und Umrechnungen.

Pearson, Student- und Normalverteilung, Perzentil-Intervall und Bootstrap-Grundlagen kommen aus prs_algorithm.py (wörtliche Kopie). Kendall wird über
Vorzeichenmatrizen gezählt (O(n²) je Datensatz, Speicher in Blöcken). Zufall nur über einen übergebenen VRng (prs_scenario)."""

import math

import numpy as np

import prs_algorithm as A
import rkr_constants as C


# --- Ränge und Spearman -----------------------------------------------------------------------------------------------------------------------

def rank_rows(X) -> np.ndarray:
    """Ränge je Zeile, Gleichstände mit dem Mittel ihrer Plätze: (Zahl der Werte darunter + Zahl der Werte bis einschließlich + 1) / 2."""
    X = np.atleast_2d(np.asarray(X, dtype=float))
    out = np.empty_like(X)
    for i, row in enumerate(X):
        s = np.sort(row)
        out[i] = (np.searchsorted(s, row, side="left") + np.searchsorted(s, row, side="right") + 1) / 2.0
    return out


def spearman_rows(X, Y) -> np.ndarray:
    """Spearman-ρ je Zeile: Pearson-Korrelation der Durchschnittsränge. Konstante Zeilen geben NaN."""
    return A.pearson_rows(rank_rows(X), rank_rows(Y))


def spearman_r(x, y) -> float:
    """Spearman-ρ eines Datensatzes; Fehler bei konstanter Kennzahl."""
    if len(x) != len(y):
        raise ValueError("x und y müssen gleich lang sein")
    return A.pearson_r(rank_rows(x)[0], rank_rows(y)[0])


# --- Kendall ----------------------------------------------------------------------------------------------------------------------------------

def _sgn(X) -> np.ndarray:
    return np.sign(X[:, :, None] - X[:, None, :]).astype(np.int8)


def kendall_counts(x, y) -> dict:
    """Paarzählung: C konkordante, D diskordante Paare, T_x / T_y Paare mit Gleichstand in x bzw. y (jeweils auch die, die in beiden gleich sind), n0 = n(n−1)/2."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n = len(x)
    if n != len(y) or n < 2:
        raise ValueError("mindestens zwei Beobachtungspaare gleicher Länge nötig")
    a, b = _sgn(x[None])[0], _sgn(y[None])[0]
    prod = a * b
    return {"C": int((prod == 1).sum()) // 2, "D": int((prod == -1).sum()) // 2, "Tx": (int((a == 0).sum()) - n) // 2, "Ty": (int((b == 0).sum()) - n) // 2,
            "n0": n * (n - 1) // 2, "n": n, "mx": len(np.unique(x)), "my": len(np.unique(y))}


def kendall_tau(x, y, variant: str = "b") -> float:
    """Kendall-τ: a = (C − D)/n0 (ignoriert Gleichstände), b = (C − D)/√((n0 − T_x)(n0 − T_y)), c = 2(C − D)/(n²(m − 1)/m) mit m = kleinere Stufenzahl; γ = (C − D)/(C + D)."""
    k = kendall_counts(x, y)
    cd = k["C"] - k["D"]
    if variant == "a":
        return cd / k["n0"]
    if variant == "b":
        den = math.sqrt((k["n0"] - k["Tx"]) * (k["n0"] - k["Ty"]))
        if den == 0:
            raise ValueError("eine Kennzahl ist konstant: τ-b ist nicht definiert")
        return cd / den
    if variant == "c":
        m = min(k["mx"], k["my"])
        if m < 2:
            raise ValueError("eine Kennzahl ist konstant: τ-c ist nicht definiert")
        return 2.0 * cd / (k["n"] ** 2 * (m - 1) / m)
    if variant == "gamma":
        if k["C"] + k["D"] == 0:
            raise ValueError("kein Paar ohne Gleichstand: γ ist nicht definiert")
        return cd / (k["C"] + k["D"])
    raise ValueError(f"unbekannte Variante: {variant}")


def kendall_rows(X, Y, variant: str = "b") -> np.ndarray:
    """Kendall-τ (b, a, c oder γ) je Zeile, in Blöcken über Vorzeichenmatrizen; Zeilen mit nicht definiertem Nenner geben NaN."""
    X = np.asarray(X, dtype=float)
    Y = np.asarray(Y, dtype=float)
    reps, n = X.shape
    n0 = n * (n - 1) / 2.0
    step = max(1, int(1.2e7 // (n * n)))
    out = []
    for s in range(0, reps, step):
        a, b = _sgn(X[s:s + step]), _sgn(Y[s:s + step])
        prod = a * b
        cd = prod.sum((1, 2), dtype=np.int64) / 2.0
        with np.errstate(invalid="ignore", divide="ignore"):
            if variant == "a":
                out.append(cd / n0)
            elif variant == "b":
                n1 = ((a == 0).sum((1, 2)) - n) / 2.0
                n2 = ((b == 0).sum((1, 2)) - n) / 2.0
                out.append(cd / np.sqrt((n0 - n1) * (n0 - n2)))
            elif variant == "gamma":
                out.append(cd / (np.abs(prod).sum((1, 2), dtype=np.int64) / 2.0))
            elif variant == "c":
                m = np.minimum([len(np.unique(x)) for x in X[s:s + step]], [len(np.unique(y)) for y in Y[s:s + step]])
                out.append(np.where(m > 1, 2.0 * cd / (n * n * (m - 1) / np.maximum(m, 1)), np.nan))
            else:
                raise ValueError(f"unbekannte Variante: {variant}")
    return np.concatenate(out)


# --- Tests --------------------------------------------------------------------------------------------------------------------------------------

def spearman_t_p(rs: float, n: int) -> float:
    """Zweiseitiger p-Wert der t-Näherung für ρ_S = 0 (n − 2 Freiheitsgrade; bei kleinem n und Gleichständen nur eine Näherung)."""
    if n < 3:
        raise ValueError("der Test braucht mindestens drei Beobachtungen")
    if abs(rs) >= 1.0:
        return 0.0
    return A.t_two_sided_p(rs * math.sqrt((n - 2) / (1.0 - rs * rs)), n - 2)


def kendall_z(tau: float, n: int) -> float:
    """z = τ √(9n(n − 1) / (2(2n + 5))): Normalnäherung unter Unabhängigkeit, ohne Gleichstandskorrektur."""
    if n < 2:
        raise ValueError("mindestens zwei Beobachtungen nötig")
    return tau * math.sqrt(9.0 * n * (n - 1) / (2.0 * (2.0 * n + 5.0)))


def kendall_z_p(tau: float, n: int) -> float:
    return math.erfc(abs(kendall_z(tau, n)) / math.sqrt(2.0))


def rejects_spearman_rows(rs, n: int, alpha: float = 0.05) -> np.ndarray:
    """Verwirft die t-Näherung ρ_S = 0 (zweiseitig), je Zeile."""
    rs = np.asarray(rs, dtype=float)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.abs(rs) * np.sqrt((n - 2) / (1.0 - rs * rs)) > A.t_critical(n - 2, alpha)


def rejects_kendall_rows(tau, n: int, alpha: float = 0.05) -> np.ndarray:
    """Verwirft die Normalnäherung τ = 0 (zweiseitig), je Zeile."""
    tau = np.asarray(tau, dtype=float)
    return np.abs(tau) * math.sqrt(9.0 * n * (n - 1) / (2.0 * (2.0 * n + 5.0))) > A.norm_ppf(1.0 - alpha / 2.0)


def permutation_p_rank(x, y, B: int, rng, measure: str = "spearman") -> float:
    """Zweiseitiger Permutationstest auf Unabhängigkeit mit |ρ_S| oder |τ_b|: (1 + #{|T*| ≥ |T|}) / (B + 1)."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n = len(x)
    perm = np.argsort(rng.uniform((B, n)), axis=1)
    if measure == "spearman":
        obs = abs(spearman_r(x, y))
        rx = np.broadcast_to(rank_rows(x)[0], (B, n))
        stat = np.abs(A.pearson_rows(rx, rank_rows(y)[0][perm]))
    elif measure == "kendall":
        obs = abs(kendall_tau(x, y, "b"))
        stat = np.abs(kendall_rows(np.broadcast_to(x, (B, n)), y[perm], "b"))
    else:
        raise ValueError(f"unbekanntes Maß: {measure}")
    return float((1 + int((stat >= obs - 1e-12).sum())) / (B + 1))


# --- Intervalle ---------------------------------------------------------------------------------------------------------------------------------

def bonett_wright_interval(rs: float, n: int, level: float = C.LEVEL) -> tuple[float, float]:
    """Fisher-Intervall für ρ_S mit dem Standardfehler √((1 + ρ_S²/2)/(n − 3)) (Näherung; hier nur durch die Messungen gestützt)."""
    if n < 4:
        raise ValueError("das Intervall braucht mindestens vier Beobachtungen")
    z = A.norm_ppf(0.5 + 0.5 * level)
    zr = math.atanh(max(-1.0 + 1e-12, min(1.0 - 1e-12, rs)))
    se = math.sqrt((1.0 + rs * rs / 2.0) / (n - 3))
    return math.tanh(zr - z * se), math.tanh(zr + z * se)


def naive_spearman_interval(rs: float, n: int, level: float = C.LEVEL) -> tuple[float, float]:
    """Fisher-Intervall für ρ_S mit dem Pearson-Standardfehler 1/√(n − 3) (so wird es oft angewendet; zu eng bei hohem ρ_S)."""
    return A.fisher_interval(rs, n, level)


def rank_bootstrap(x, y, B: int, rng) -> tuple[np.ndarray, np.ndarray]:
    """B Bootstrap-Werte (ρ_S*, τ_b*): Paare mit Zurücklegen ziehen, Durchschnittsränge und Paarzählung je Ziehung neu."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n = len(x)
    idx = rng.below(n, (B, n))
    xb, yb = x[idx], y[idx]
    return spearman_rows(xb, yb), kendall_rows(xb, yb, "b")


def interval_through(f, interval: tuple[float, float]) -> tuple[float, float]:
    """Intervall durch eine monoton steigende Abbildung f (z. B. sin(πτ/2)) schicken."""
    return f(interval[0]), f(interval[1])


# --- Formeln und Umrechnungen ------------------------------------------------------------------------------------------------------------------

def tau_formula(rho: float) -> float:
    """Kendall-τ der elliptischen Verteilungen (auch der Normalverteilung): 2/π · arcsin ρ."""
    return 2.0 / math.pi * math.asin(rho)


def spearman_formula(rho: float) -> float:
    """Spearman-ρ der Normalkopula: 6/π · arcsin(ρ/2) (gilt nicht für t(ν) und kontaminiert)."""
    return 6.0 / math.pi * math.asin(rho / 2.0)


def rho_from_tau(tau: float) -> float:
    """ρ aus Kendall-τ: sin(πτ/2), für alle elliptischen Verteilungen und die Normalkopula."""
    return math.sin(math.pi * tau / 2.0)


def rho_from_spearman(rs: float) -> float:
    """ρ aus Spearman-ρ: 2 sin(πρ_S/6), nur bei Normalkopula richtig."""
    return 2.0 * math.sin(math.pi * rs / 6.0)


NORMAL_COPULA_FAMILIES = ("normal", "lognormal05", "lognormal1")


def tau_true(family: str, rho: float) -> float:
    """Wahres Kendall-τ der Familie bei Parameter ρ (alle Familien sind elliptisch oder haben die Normalkopula; „tindep3“: 0)."""
    return 0.0 if family == C.INDEPENDENT_T3 else tau_formula(rho)


def spearman_true(family: str, rho: float) -> float | None:
    """Wahres Spearman-ρ, wo es bekannt ist (Normalkopula); sonst None (t(ν), kontaminiert: keine geschlossene Formel)."""
    if family == C.INDEPENDENT_T3:
        return 0.0
    return spearman_formula(rho) if family in NORMAL_COPULA_FAMILIES else None


def influence_rank(x, y) -> np.ndarray:
    """ρ_S − ρ_S ohne Beobachtung i, für alle i (jeweils neue Ränge)."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n = len(x)
    keep = np.array([np.delete(np.arange(n), i) for i in range(n)])
    with np.errstate(invalid="ignore"):
        return spearman_r(x, y) - spearman_rows(x[keep], y[keep])
