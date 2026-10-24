"""App: jedes Preset, der Permalink, jeder Experiment-Knopf einzeln und die Eckfall-Kombinationen (DEMO-PLAYBOOK §1d: jeder Knopf einzeln im AppTest)."""

import os
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

import rkr_constants as C

APP = str(Path(__file__).resolve().parent.parent / "app.py")
TIMEOUT = int(os.environ.get("APPTEST_TIMEOUT", "300"))


def _app() -> AppTest:
    at = AppTest.from_file(APP, default_timeout=TIMEOUT)
    at.run()
    assert not at.exception, [str(e) for e in at.exception]
    return at


def _texts(at):
    return " ".join([m.value for m in at.markdown] + [s.value for s in at.success] + [i.value for i in at.info] + [w.value for w in at.warning])


@pytest.mark.parametrize("name", list(C.PRESETS))
def test_every_preset_runs_and_shows_its_data(name):
    at = _app()
    at.sidebar.button(key=f"preset_{name}").click().run()
    assert not at.exception, [str(e) for e in at.exception]
    preset = C.PRESETS[name]
    assert at.session_state["n_slider"] == preset["n"] and at.session_state["family_select"] == preset["family"]
    assert len(at.get("plotly_chart")) >= 9


def test_permalink_is_loaded_and_clamped():
    at = AppTest.from_file(APP, default_timeout=TIMEOUT)
    at.query_params["fam"] = "t3"
    at.query_params["n"] = "99999"
    at.query_params["rho"] = "-0.4"
    at.query_params["ld"] = "quer"          # ungültig: wird ignoriert
    at.run()
    assert not at.exception
    assert at.session_state["family_select"] == "t3" and at.session_state["n_slider"] == C.N_MAX
    assert at.session_state["rho_slider"] == -0.4 and at.session_state["lev_dir_select"] == "gleichläufig"


def test_leverage_widgets_are_disabled_without_leverage_points():
    at = _app()
    assert at.sidebar.slider(key="lev_k_slider").proto.disabled and at.sidebar.selectbox(key="lev_dir_select").proto.disabled
    at.sidebar.slider(key="lev_count_slider").set_value(1).run()
    assert not at.sidebar.slider(key="lev_k_slider").proto.disabled and not at.sidebar.selectbox(key="lev_dir_select").proto.disabled


@pytest.mark.parametrize("direction", list(C.LEV_DIRS))
@pytest.mark.parametrize("edge", ["min", "max"])
def test_leverage_position_at_its_limits_with_one_and_two_points(edge, direction):
    for count in (1, 2):
        at = _app()
        at.sidebar.slider(key="lev_count_slider").set_value(count).run()
        w = at.sidebar.slider(key="lev_k_slider")
        w.set_value(w.min if edge == "min" else w.max).run()
        at.sidebar.selectbox(key="lev_dir_select").set_value(direction).run()
        assert not at.exception, (count, edge, direction, [str(e) for e in at.exception])


def test_a_leverage_point_changes_pearson_but_hardly_spearman():
    at = _app()
    r0, s0 = float(at.metric[0].value), float(at.metric[1].value)
    at.sidebar.selectbox(key="family_select").set_value("normal").run()
    at.sidebar.slider(key="rho_slider").set_value(0.0).run()
    at.sidebar.slider(key="lev_count_slider").set_value(1).run()
    at.sidebar.slider(key="lev_k_slider").set_value(8.0).run()
    assert not at.exception and float(at.metric[0].value) > 0.4 > abs(float(at.metric[1].value)) + 0.1


def test_curve_button_shows_chart_and_verdict_and_resets_with_the_settings():
    at = _app()
    assert "Bei c = " not in _texts(at)
    at.slider(key="curve_n").set_value(60).run()
    at.button(key="btn_curve").click().run()
    assert not at.exception, [str(e) for e in at.exception]
    assert "Bei c = 2, Rauschen 0.5, n = 60" in _texts(at)
    at.slider(key="curve_noise").set_value(0.0).run()
    assert "Bei c = " not in _texts(at)


def test_pdf_button_then_download():
    at = _app()
    at.button(key="pdf_make").click().run()
    assert not at.exception and len(at.get("download_button")) == 1


def test_experiment_selectors_run():
    at = _app()
    for key, value in (("lev_n", 300), ("lev_k_sel", 15), ("lev_k_sel", 0), ("size_n", 300), ("size_family", "kontaminiert"), ("size_family", "normal"), ("cover_family", "lognormal1"),
                       ("cover_family", "normal"), ("cover_rho", 0.9), ("cover_rho", 0.0), ("eff_rho", 0.9)):
        at.selectbox(key=key).set_value(value).run()
        assert not at.exception, (key, [str(e) for e in at.exception])


def test_the_experiment_sections_show_their_verdicts():
    text = _texts(_app())
    for needle in ("Die Kendall-Umrechnung sin(πτ/2)", "lässt bei n = 30 den Pearson-Test", "Bei wahrem ρ = 0", "Bei 3 Stufen ergibt dieselbe Skala", "Das Kendall-Intervall deckt", "Die Streuung der Kendall-Umrechnung"):
        assert needle in text, needle


def test_corner_combination_of_all_minima_and_all_maxima():
    for edge in ("min", "max"):
        at = _app()
        w = at.sidebar.slider(key="lev_count_slider")
        w.set_value(int(w.min if edge == "min" else w.max)).run()
        for key in ("rho_slider", "n_slider", "lev_k_slider", "boot_slider"):
            w = at.sidebar.slider(key=key)
            if w.proto.disabled:
                continue
            value = w.min if edge == "min" else w.max
            w.set_value(int(value) if isinstance(w.value, int) else value)
        at.sidebar.selectbox(key="family_select").set_value("t3" if edge == "min" else "lognormal1")
        at.run()
        assert not at.exception, (edge, [str(e) for e in at.exception])
