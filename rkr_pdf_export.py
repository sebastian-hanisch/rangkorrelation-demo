"""PDF-Export (fpdf2, Kernschriften): Einstellungen, Ergebnis und Intervalle der Hauptansicht. Kein Gedankenstrich und kein Euro-Zeichen (Kernschriften)."""

from fpdf import FPDF

import rkr_constants as C
import rkr_evaluation as ev


def _text(value: str) -> str:
    """Zeichen, die die Kernschriften nicht kennen, ersetzen (Gedankenstrich, Pfeile, Fettdruck-Markierung, griechische und mathematische Zeichen)."""
    table = {"–": "-", "—": "-", "→": "->", "**": "", "€": "EUR", "·": "*", "≤": "<=", "≥": ">=", "−": "-", "×": "x", "„": '"', "“": '"', "ρ": "rho", "σ": "sigma", "τ": "tau", "γ": "gamma",
             "ν": "nu", "κ": "kappa", "²": "^2", "√": "sqrt ", "…": "...", "Φ": "Phi", "π": "pi"}
    for old, new in table.items():
        value = value.replace(old, new)
    return value.encode("latin-1", "replace").decode("latin-1")


def _para(pdf: FPDF, height: float, text: str) -> None:
    """Absatz über die volle Breite; danach steht der Cursor wieder am linken Rand."""
    pdf.multi_cell(0, height, _text(text), new_x="LMARGIN", new_y="NEXT")


def build_pdf(a: ev.Analysis) -> bytes:
    s = a.settings
    pdf = FPDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.set_font("Helvetica", "B", 15)
    pdf.cell(0, 9, _text("Rangkorrelation: Spearman, Kendall und was Ränge ändern"), new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 10)
    _para(pdf, 5.5, f"Familie: {C.ALL_FAMILIES[s.family]}   rho: {s.rho:.2f}   n: {s.n}   Hebelpunkte: {len(a.lever)}   Zufalls-Seed: {s.seed}")
    pdf.ln(1)
    pdf.set_font("Helvetica", "B", 11)
    pdf.cell(0, 7, "Ergebnis", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 10)
    _para(pdf, 5.5, ev.verdict_main(a)[1])
    _para(pdf, 5.5, f"r = {a.r:.3f}, Spearman = {a.rho_s:.3f}, Kendall tau-b = {a.tau:.3f}; p (Pearson-t / Spearman-t / Kendall-z) = {a.p_pearson:.4f} / {a.p_spearman:.4f} / {a.p_kendall:.4f}")
    pdf.ln(1)
    pdf.set_font("Helvetica", "B", 11)
    pdf.cell(0, 7, "95 %-Intervalle", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 10)
    for name, (lo, hi) in a.intervals.items():
        target = a.targets[name]
        cover = "Ziel unbekannt" if target is None else f"Ziel {target:.3f}: {'ja' if a.covers[name] else 'nein'}"
        _para(pdf, 5.5, f"{name}: [{lo:.3f}; {hi:.3f}]   {cover}")
    pdf.ln(2)
    pdf.set_font("Helvetica", "I", 8)
    _para(pdf, 4.5, "Simulierte Daten mit bekannter wahrer Korrelation. Sebastian Hanisch - Operations Research und Machine Learning.")
    return bytes(pdf.output())
