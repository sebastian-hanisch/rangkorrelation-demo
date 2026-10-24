"""Presets: Vollständigkeit, gültige Werte, Standard-Preset = Standardeinstellungen, Hilfetexte."""

import rkr_constants as C
import rkr_evaluation as ev
import rkr_presets as P


def test_every_preset_has_help_and_all_keys():
    assert set(C.PRESETS) == set(C.PRESET_HELP)
    for name, preset in C.PRESETS.items():
        assert set(preset) == set(P.PRESET_KEYS), name
        assert C.PRESET_HELP[name] and C.PRESET_HELP[name] != name       # ein echter Text, nicht der Name


def test_preset_values_are_inside_the_slider_bounds():
    for preset in C.PRESETS.values():
        assert preset["family"] in C.FAMILIES
        assert C.RHO_MIN <= preset["rho"] <= C.RHO_MAX and C.N_MIN <= preset["n"] <= C.N_MAX
        assert 0 <= preset["lev_count"] <= C.LEV_COUNT_MAX and 0 <= preset["lev_k"] <= C.LEV_K_MAX and 0 <= preset["seed"] <= C.SEED_MAX


def test_default_preset_equals_the_default_settings():
    p = C.PRESETS["Standardfall (Voreinstellung)"]
    assert ev.Settings(family=p["family"], rho=p["rho"], n=p["n"], lev_count=p["lev_count"], lev_k=p["lev_k"], seed=p["seed"]) == ev.Settings()


def test_setting_specs_cover_every_preset_key_and_the_defaults_match():
    assert set(P.PRESET_KEYS.values()) <= set(P.SETTING_SPECS)
    s = ev.Settings()
    assert [P.SETTING_SPECS[k].default for k in ("family_select", "rho_slider", "n_slider", "lev_count_slider", "lev_k_slider", "lev_dir_select", "boot_slider", "seed_input")] == [
        s.family, s.rho, s.n, s.lev_count, s.lev_k, s.lev_dir, s.boot_b, s.seed]


def test_url_parameters_are_unique():
    params = [s.url_param for s in P.SETTING_SPECS.values()]
    assert len(params) == len(set(params))


def test_clean_value_for_the_permalink():
    spec = P.SETTING_SPECS
    assert P.clean_value(spec["n_slider"], "99999") == C.N_MAX and P.clean_value(spec["n_slider"], "-4") == C.N_MIN
    assert P.clean_value(spec["n_slider"], "abc") is None and P.clean_value(spec["n_slider"], None) is None
    assert P.clean_value(spec["rho_slider"], "0.1500000001") == 0.15 and P.clean_value(spec["rho_slider"], "7") == C.RHO_MAX
    assert P.clean_value(spec["family_select"], "t3") == "t3" and P.clean_value(spec["family_select"], "tindep3") is None
    assert P.clean_value(spec["lev_dir_select"], "gegenläufig") == "gegenläufig" and P.clean_value(spec["lev_dir_select"], "quer") is None
    assert P.clean_value(spec["boot_slider"], "50") == C.B_MIN and P.clean_value(spec["seed_input"], "12.0") is None
