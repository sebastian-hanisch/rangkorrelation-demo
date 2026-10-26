# Rangkorrelation: Spearman, Kendall und was Ränge ändern – Streamlit-Demo

<!-- TODO nach dem Deploy: direkt hierher den Live-Link setzen (Leerzeile davor und danach):
**[→ Demo live ausprobieren](https://sebastianhanisch-rangkorrelation-demo.streamlit.app/)** -->

Zweites Stück der Korrelations-Reihe in der „Konzepte“-Reihe für die Website „Sebastian Hanisch – Operations Research und Machine Learning“, Nachfolger von [pearson-demo](https://github.com/sebastian-hanisch/pearson-demo) (Kante `fix`: dort kippt r durch einen einzelnen Punkt, wird bei schweren Rändern unsicher und sieht nur Geraden). Es ist die **erste Demo des Portfolios, die Rangmaße als Zusammenhang zweier Messreihen zum Gegenstand macht**: [rangdistanz-demo](https://github.com/sebastian-hanisch/rangdistanz-demo) behandelt Kendall und Spearman als **Abstand zweier Rangfolgen** (Turniertabellen), hier geht es um Messwerte, Streuung, Hebelpunkte, Gleichstände, Tests und Intervalle. Vehikel sind dieselben **simulierten Standort-Wochen** wie im ersten Stück (x: Abweichung des Auftragsvolumens vom Plan, y: Überstunden) mit **bekannter wahrer Korrelation**.

## Warum dieses Problem

Das erste Stück zeigte drei Schwächen von Pearson-r. Die Demo fragt, ob Ränge sie beheben, und was sie kosten. Befunde: **Rangtests liegen bei gemeinsamer Streuung (t(3), kontaminiert) bei 5,9 bis 8,5 % Fehlalarm statt bei bis zu 43 %**, ein **einzelner Punkt** bei fünf Standardabweichungen lässt den Rangtest bei n = 30 in 5,8 % statt 84,7 % anschlagen, die **Kendall-Umrechnung** sin(πτ/2) trifft ρ in allen vier untersuchten Familien im Rahmen des Messfehlers (und streut bei t(3) etwa halb so stark wie r), und **die Variante entscheidet die Zahl** (bei drei gleich besetzten Stufen, Normalverteilung, ρ = 0,6 und n = 500 ergeben τ-a, τ-b und γ 0,297, 0,444 und 0,625). Ehrlich dazu: bei einer schwach gekrümmten Kurve mit Rauschen liegt das Rangmaß **unter** Pearson, und die Rangtests sind bei gemeinsamer Streuung nicht ganz exakt.

## Befunde und Korrekturen gegenüber dem Plan

Der Einzelplan (`korrelation-planung/STUECK-2-rangkorrelation.md`, nicht in diesem Repo) stützte sich auf Wegwerfskripte mit dem Zufallsgenerator von numpy und `scipy`-Rangfunktionen. Gebaut ist alles mit SplitMix64 und eigenem Kern; die Zahlen weichen leicht ab, die Befunde bleiben:

- **Hebelpunkt (k = 5, n = 30):** Pearson / Spearman / Kendall 84,7 / 5,8 / 6,3 % (Plan: 83,3 / 6,4 / 7,1 %).
- **Fehlalarm bei gemeinsamer Streuung (t(3), n = 100):** 32,3 / 7,1 / 8,5 % (Plan: 32,2 / 7,0 / 8,2 %); kontaminiert, n = 100: 40,9 / 6,3 / 6,9 % (Plan: 41,7 / 6,4 / 6,9 %).
- **Gleichstände (3 Stufen):** τ-a 0,297, τ-b 0,444, γ 0,625 (Plan: 0,298 / 0,445 / 0,627).
- **Pearson der Kurve e^(3x) ohne Rauschen** bei n = 20 / 100 / 500 / 2 000: 0,638 / 0,458 / 0,317 / 0,235 (Plan: 0,639 / 0,453 / 0,316 / 0,233).
- **Bonett–Wright:** Abdeckung 93,4 bis 96,9 % (Plan: 94,9 bis 97,0 %; die kleinste Zelle liegt bei ρ = 0,9, n = 100). Kendall-Bootstrap: 92,2 bis 98,1 % über alle Familien (Plan: 92,3 bis 98,5 %).
- **Presets:** „Schwerer Rand“ und „Mengen“ nutzen ρ = 0,5 (Plan: 0,6), damit ihre Hilfetexte Zahlen des Rasters nennen.
- **Neu gegenüber dem Plan:** Rangtests mit Permutation (im Kern, nicht in der App), die Zeile „Pearson der Codes“ in der Gleichstand-Tabelle.

## Modell

Paare (x, y) aus sieben Familien (wörtliche Kopien aus `pearson-demo`: `prs_scenario.py`, `prs_constants.py`, `prs_algorithm.py`, `prs_rng.py`): **Normal**; **t(10), t(5), t(3)** und **kontaminiert** mit **gemeinsamem Skalenfaktor** (Spitzenwochen heben beide Kennzahlen zugleich; ρ bleibt, aber die Kennzahlen sind nicht unabhängig); **Lognormal σ = 0,5 und 1** (Mengen: dieselben Ränge wie bei Normal, aber eine ganz andere Pearson-Korrelation). Für Experiment 4 gibt es t(3) mit unabhängigen Rändern. Formeln im 📐-Expander der App.

## Methodik

- **Ränge mit Gleichständen** (Durchschnittsränge), **Spearman-ρ** als Pearson der Ränge, **Kendall** über Vorzeichenmatrizen in O(n²) (τ-a, τ-b, τ-c, γ), dazu Näherungstests (t, Normal) und Permutationstests, Bonett–Wright-Intervall, Perzentil-Bootstrap, Umrechnungen sin(πτ/2) und 2 sin(πρ_S/6). Alles in `rkr_algorithm.py`, ohne scipy zur Laufzeit.
- **Experimente:** (1) Beziehungen und Umrechnungen, (2) monotone Kurve (live), (3) Hebelpunkt, (4) gemeinsame Streuung, (5) Gleichstände, (6) Intervalle und Preis in Genauigkeit.
- **Vorgerechnetes Raster** (`tools/make_grid.py`, `data/rank_grid.json`, 564 Sekunden auf 14 Prozessen): Experiment 1 (12 Zellen, 40 Stichproben mit n = 2 000), 3 (36 Zellen, 4 · 1 000 Läufe, n = 300: 4 · 500), 4 (21 Zellen, 4 · 2 000, n = 300: 4 · 1 000), 5 (6 Zellen, n = 500, 4 · 250), 6 (72 Intervallzellen, Bootstrap mit 200 Ziehungen, 4 · 250 Läufe, n = 300: 4 · 150; 15 Schätzerzellen, n = 100, 4 · 1 000).

## Befunde (gemessen, keine Behauptungen)

Alle Messzahlen stehen in `tests/test_claims.py` (Raster-Werte gegen die JSON-Datei, Kurve live über dieselben Funktionen wie in der App); Laufzeit- und Mutationsangaben sind dort nicht erfasst.

| Frage | Befund |
|---|---|
| Gelten die Formeln? (Experiment 1) | **τ = 2/π · arcsin ρ gilt in allen vier Familien:** bei ρ = 0,6 Normal 0,410, t(3) 0,411, kontaminiert 0,412, Lognormal 0,409 (Formel 0,410). **ρ_S = 6/π · arcsin(ρ/2) gilt nur bei Normalkopula:** Normal 0,582, Lognormal 0,581 (Formel 0,582), aber t(3) 0,563 und kontaminiert 0,573. |
| Welche Umrechnung stimmt? | **sin(πτ/2) trifft ρ in allen 12 Zellen im Rahmen des Messfehlers** (größte Abweichung im Raster 0,004, Messfehler etwa ±0,005), 2 sin(πρ_S/6) weicht bis **0,019** ab (t(3), ρ = 0,6: 0,581). Lognormal hat dieselben Ränge wie Normal (ρ_S 0,288 / 0,581 / 0,891 bei ρ = 0,3 / 0,6 / 0,9), aber ein ganz anderes r: gemessen **0,206 / 0,466 / 0,851** (Mittel aus 40 Stichproben; der wahre Wert (e^ρ − 1)/(e − 1) ist 0,204 / 0,479 / 0,849). |
| Wie verhält sich eine monotone Kurve? (Experiment 2, y = e^(cx), n = 100) | Ohne Rauschen sind ρ_S und τ **genau 1,000**, Pearson bei c = 0,5 / 1 / 2 / 3 / 4 nur **0,945 / 0,807 / 0,583 / 0,465 / 0,393** (200 Läufe je c: die dritte Stelle ist nicht belastbar, mit mehr Läufen liegt Pearson bei c = 1 eher bei 0,82). Mit Rauschen (Standardabweichung 0,5): r 0,716 / 0,781 / 0,582 / 0,465 / 0,393, ρ_S 0,682 / 0,857 / 0,908 / 0,918 / 0,914, τ 0,501 / 0,688 / 0,774 / 0,798 / 0,798. **Bei c = 0,5 liegt ρ_S (0,682) unter r (0,716).** |
| Hängt Pearson an n? | Ja: bei c = 3 ohne Rauschen fällt r von 0,638 (n = 20) über 0,458 (100) und 0,317 (500) auf 0,235 (2 000); die Populationskorrelation ist c/√(e^(c²) − 1) = 0,033 (Formel nur durch die Messung gestützt). Die Ränge bleiben bei 1,000. |
| Was macht ein einzelner Punkt? (Experiment 3, k = 5, n = 10 / 30 / 100 / 300) | Pearson-Test schlägt in **84,3 / 84,7 / 52,8 / 26,5 %** der Läufe an, Spearman in **9,0 / 5,8 / 5,3 / 5,9 %**, Kendall in **9,1 / 6,3 / 5,7 / 5,9 %**. Ohne Punkt (k = 0): 4,8 / 4,6 / 5,0 / 5,2 % für Pearson. Bei n = 30, k = 5 ist r im Mittel 0,471, ρ_S 0,100, τ 0,069. Der Anteil der Rangtests ändert sich ab k ≈ 3 kaum noch (im Rahmen des Rauschens von etwa ±0,4 Punkten): ein Punkt ist für Ränge ein Punkt, für Pearson ein Gewicht. |
| Schlagen Rangtests bei gemeinsamer Streuung zu oft an? (Experiment 4, ρ = 0, n = 30 / 100 / 300) | **t(3):** Pearson 22,6 / 32,3 / 41,0 %, Spearman 7,2 / 7,1 / 7,5 %, Kendall 8,0 / 8,5 / 8,5 %. **Kontaminiert:** Pearson 30,0 / 40,9 / 42,9 %, Spearman 5,9 / 6,3 / 6,8 %, Kendall 6,3 / 6,9 / 7,1 %. t(5) und t(10): Pearson 13,2 bis 18,6 % und 7,2 bis 8,5 %, die Rangtests 5,1 bis 7,0 %. **Unabhängige Ränder und Lognormal: alle drei Tests nahe 5 %.** Die Rangtests sind also **fast**, nicht ganz gerettet (bis 8,5 % statt 5 %). |
| Wie sehr ändert die Variante die Zahl? (Experiment 5, gleich besetzte Stufen, bivariate Normalverteilung, n = 500, ρ = 0,6, stetig τ = 0,410, ρ_S = 0,582) | Bei **3 Stufen** τ-a 0,297, τ-b 0,444, τ-c 0,444, γ 0,625, ρ_S 0,493 (gleich dem Pearson der Codes); bei 5 Stufen 0,362 / 0,451 / 0,551 (τ-a / τ-b / γ); bei 11 Stufen 0,397 / 0,436 / 0,477; bei 100 Stufen 0,409 / 0,412 / 0,416. |
| Halten die Intervalle? (Experiment 6, 18 Zellen je Familie: n = 10 bis 300 · ρ = 0 / 0,5 / 0,9) | **Kendall-Bootstrap:** Normal 93,0 bis 98,1 %, Lognormal 92,2 bis 98,1 %, **t(3) 92,6 bis 96,2 %, kontaminiert 93,2 bis 97,0 %**. **Pearson (Fisher-z) gegen ρ:** Normal 93,2 bis 96,5 %, t(3) **57,3 bis 86,3 %**, kontaminiert **54,7 bis 84,7 %**, Lognormal **25,2 bis 95,8 %**. **Spearman:** Bonett–Wright 93,4 bis 96,9 % (Normal) und 93,5 bis 96,4 % (Lognormal), naives Fisher 90,6 bis 95,5 % bzw. 88,5 bis 95,6 %. |
| Wie breit sind sie? (n = 100, ρ = 0,5) | Normal: Bonett–Wright 0,321, Kendall 0,234 (in ρ umgerechnet 0,318). t(3): 0,326 (nur Normalkopula belegt), 0,267, umgerechnet 0,362. |
| Was kostet die Umrechnung an Genauigkeit? (n = 100, 4 · 1 000 Läufe, ρ = 0,6) | Streuung Pearson / 2 sin(πρ_S/6) / sin(πτ/2): **Normal 0,065 / 0,070 / 0,069**, t(3) **0,146 / 0,083 / 0,082**, kontaminiert 0,153 / 0,078 / 0,076, Lognormal 0,134 (Mittel 0,502, kein Schätzer für ρ) / 0,070 / 0,069. Verhältnis Kendall zu Pearson, gemessen **0,50 (kontaminiert) bis 1,07 (Normal)** bei ρ = 0,6 und **0,45 (t(3)) bis 1,12 (Normal)** bei ρ = 0,9, also etwa 0,5 bis 1,1 (bei schweren Rändern streut r selbst stark, die zweite Stelle ist dort nicht belastbar). Die Spearman-Umrechnung ist bei t(3) verschoben (Mittel 0,573 statt 0,6). |
| Handwerte | (1, 1), (2, 3), (3, 2), (4, 4): ρ_S = 0,8, τ = 2/3 (C = 5, D = 1), sin(πτ/2) = 0,866; mit Gleichständen x = (1, 1, 2, 3), y = (1, 2, 2, 3): τ-a 0,667, τ-b 0,8, τ-c 0,75, γ 1, ρ_S = 0,833. |

## Ehrliche Grenzen

- Alles ist **simuliert**; die Familien sind Beispiele. Der gemeinsame Skalenfaktor ist eine Modellwahl für Spitzenwochen, kein Befund über echte Standorte. Die **wahre Korrelation ist nur in der Simulation bekannt**.
- Die Spearman-Wahrheit ist nur bei **Normalkopula** bekannt (Normal, Lognormal); bei t(ν) und kontaminiert fehlt sie, dort gibt es für Spearman keine Abdeckung. τ und ρ (über sin(πτ/2)) sind für alle Familien bekannt.
- Die **Rangtests sind Näherungen** (t-Näherung, Normalnäherung ohne Gleichstandskorrektur): bei n = 10 und groben Skalen ungenau, bei gemeinsamer Streuung liegen sie bei 5,9 bis 8,5 %. Die App zeigt die Näherungs-p-Werte; ein exakter Test oder eine Gleichstandskorrektur fehlen.
- **Bonett–Wright** (Standardfehler √((1 + ρ_S²/2)/(n − 3))) ist hier **nur durch die Messung gestützt**, nicht aus einer Quelle belegt; seine kleinste Abdeckung liegt bei 93,4 % (das Minimum über verrauschte Zellen, Standardfehler 0,7 bis 1,0 Punkte).
- Die Raster-Zahlen haben vier Seeds und Bootstrap mit 200 Ziehungen; die Streuung der Abdeckung über die Seeds beträgt bis 2,7 Punkte (Kendall-Bootstrap, Normal) bzw. 3,7 Punkte (t(3)).
- **Ränge sind nicht immer besser** (schwach gekrümmte Kurve mit Rauschen) und **sehen keine nichtmonotonen Beziehungen** (Parabel, Kreis: nächstes Stück) und **nicht die Ränder** (Extreme gehen verloren: Kopula-Stück).
- Es gibt **keine Literaturangaben**: Zahlen stammen ausschließlich aus den eigenen Messungen, Formeln aus dem Gedächtnis sind durch Messung und Orakel gestützt.

## Verifikation

- **Kern:** Handrechnung an Mini-Instanzen (`tests/conftest.py`: τ-a, τ-b, τ-c, γ, Durchschnittsränge), Fehlerfälle (konstante Kennzahl, zu wenige Paare), jede Variante, `tests/test_algorithm.py`.
- **Orakel** (`tests/test_oracle_rangkorrelation.py`): Spearman, Kendall-b und -c und der p-Wert gegen scipy (30 Datensätze), Paarzählung und alle Varianten gegen die dreifache Schleife (30 Datensätze mit Gleichständen), `rank_rows` gegen `scipy.stats.rankdata`, Resamples und Leave-one-out gegen Schleifen, Permutationstests gegen die vollständige Aufzählung (n = 3 bis 6), Bonett–Wright und Normalnäherung gegen die Definition, Formeln gegen große Stichproben.
- **Kopien** (`tests/test_copies.py`): vier wörtliche Kopien aus `pearson-demo` gegen das Original.
- **Messreihen und Verdikte:** `tests/test_evaluation.py`, `tests/test_verdicts.py` (jede Schwelle und jede Zahl im Text an handgesetzten Werten), `tests/test_grid.py` (Raster vollständig, Plausibilität, **Probezelle gegen den Erzeuger**, Trefferanteile exakt, Breiten mit Toleranz), `tests/test_eigenschaften_rangkorrelation.py` (Invarianz unter monotonen Abbildungen, Spiegelung, Gleichstandsordnung), `tests/test_edge_cases.py`.
- **Oberfläche:** AppTest-Rauchtest (Vorlage mit einer Ausnahme: gesperrte Regler lässt sie aus, `tests/test_app_experiments.py` prüft sie mit einem und zwei Hebelpunkten), jedes Preset, Permalink, jeder Experiment-Knopf, Layout, Footer, Laufzeit-Importe.
- **Zahlen:** `tests/test_claims.py` rechnet die Messzahlen dieser README nach (nicht erfasst: Laufzeit, Mutationszahlen), dazu prüft ein Test, dass die Prozentreihen Zeichen für Zeichen der Rundung der App entsprechen (0,0635 wird 6,3 %).
- **Mutationsprüfung** (`tools/mutate_demo.py`, die vier Kopien aus `pearson-demo` ausgenommen; Stand in `tests/mutation_state.json`): 159 Stellen in `rkr_algorithm.py` laut Zustandsdatei, **alle gesichert** (153 gefangen, 6 als rechnerisch gleichwertig festgehalten), **0 offen**. Die sechs gleichwertigen sind die Kappung `−1 + 1e-12` im Bonett–Wright-Intervall, die Blockgröße von `kendall_rows`, `m > 1` und `np.maximum(m, 1)` bei τ-c und `>` gegen `≥` an den beiden Testschwellen. Der erste Lauf ließ 17 Mutanten überleben, davon elf echte Lücken (τ-c bei zwei Stufen, Schwellen der beiden Rangtests, die jetzt Randtests haben).

## Dateistruktur

| Datei | Zweck |
|---|---|
| `.github/workflows/tests.yml` | CI: Push, Pull Request, wöchentlicher Canary |
| `.gitignore` | venv, Caches, Streamlit-Secrets |
| `.streamlit/config.toml` | Streamlit-Einstellungen (keine Nutzungsstatistik) |
| `app.py` | Streamlit-App: Hauptansicht, sechs Experimente, Grenzen, Formeln |
| `data/rank_grid.json` | vorgerechnetes Raster (Experimente 1, 3 bis 6, Probezelle) |
| `prs_algorithm.py`, `prs_constants.py`, `prs_rng.py`, `prs_scenario.py` | **wörtliche Kopien** aus `pearson-demo` (Pearson-Kern, Familien, SplitMix64 vektorisiert) |
| `rkr_algorithm.py` | Kern: Ränge, Spearman, Kendall (a, b, c, γ), Tests, Intervalle, Umrechnungen |
| `rkr_constants.py` | Regler-Grenzen, Raster, Farben, Presets mit Hilfetexten |
| `rkr_evaluation.py` | Einzeldatensatz, Kurve, Raster-Zellen, Verdikte |
| `rkr_layout.py` | Fester Kopf (Stylesheet) |
| `rkr_pdf_export.py` | PDF-Export der Hauptansicht (fpdf2) |
| `rkr_presets.py` | Permalink, Presets (Schnellstart in der Seitenleiste, Hervorhebung), Zufalls-Seed-Button |
| `rkr_visualization.py` | Plotly-Abbildungen (Achsen gesperrt) |
| `requirements-dev.txt` | Laufzeit plus pytest und scipy (nur Gegenproben) |
| `requirements.txt` | Laufzeit-Bibliotheken (ungepinnt) |
| `tools/make_grid.py` | rechnet das Raster vor (`python tools/make_grid.py`, rund 10 Minuten) |
| `tests/` | siehe Verifikation |

## Bewusst nicht umgesetzt

- **Nichtlineare Abhängigkeit** (Parabel, Kreis; Hoeffding, Distanzkorrelation: nächstes Stück), **Randabhängigkeit und Kopula** (Stück 9), **Zeitreihen**, **viele Variablen**, **Kausalität**.
- Exakte Rangtests und Gleichstandskorrektur der Kendall-Varianz, Somers-d, **partielle Rangkorrelation** (`strukturelle-loecher-demo`), ein O(n log n)-Zähler für Kendall (n bleibt unter 501).
- Eine Herleitung der Näherungsformeln (Bonett–Wright, Varianz von τ) und echte Daten.

## Lokal ausführen

```bash
python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

pip install -r requirements.txt
streamlit run app.py
```

Das vorgerechnete Raster liegt im Repo; neu erzeugen mit `python tools/make_grid.py`. Die Demo nutzt nur Sprachmerkmale, die auch in der CI-Python-Version laufen; die Version der Streamlit Cloud steht in den App-Einstellungen.

## Tests ausführen

```bash
pip install -r requirements-dev.txt
pytest tests/ -v
```

Gebaut mit Streamlit, Plotly, NumPy und fpdf2.

---

Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – Operations Research und Machine Learning ([Über mich](https://sebastianhanisch.net/ueber-mich.html)). Mehr zu den Demos: [Interaktive Demos](https://sebastianhanisch.net/demos.html).
