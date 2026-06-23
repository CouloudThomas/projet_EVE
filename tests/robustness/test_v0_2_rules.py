from __future__ import annotations

from pathlib import Path

import pandas as pd

from eve.episodes.baselines import load_structured_config
from eve.robustness.build_v0_2_episode_store import run
from eve.robustness.degraded_mode import satellite_gap_state
from eve.robustness.phenology import (
    phenology_interpretation_risk,
    phenology_stage,
)
from eve.robustness.physical_costs import physical_cost_for_action
from eve.robustness.scores import consistency_score
from eve.robustness.simple_baselines import add_simple_baseline_columns
from eve.robustness.typology import classify_episode


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RULES_CONFIG = PROJECT_ROOT / "config" / "eve_rules_v0_2.yaml"
COSTS_CONFIG = PROJECT_ROOT / "config" / "eve_costs_v0_2.yaml"
SIMPLE_CONFIG = PROJECT_ROOT / "config" / "eve_simple_baselines_v0_2.yaml"


def _rules() -> dict:
    return load_structured_config(RULES_CONFIG)


def _costs() -> dict:
    return load_structured_config(COSTS_CONFIG)


def _simple() -> dict:
    return load_structured_config(SIMPLE_CONFIG)


def test_low_ndmi_dry_climate_becomes_hydroclimatic_candidate() -> None:
    primary, flags, confidence = classify_episode(
        pd.Series(
            {
                "episode_quality_risk_score_max": 0.0,
                "episode_vegetation_anomaly_score_max": 1.0,
                "episode_climatic_pressure_score_max": 0.85,
                "episode_consistency_score_max": 0.85,
                "episode_phenology_stage_main": "summer_stress_window",
                "highest_vpd_state": "high",
            }
        ),
        _rules(),
    )

    assert primary == "hydroclimatic_candidate"
    assert "hydroclimatic_candidate" in flags
    assert confidence >= 0.80


def test_low_ndmi_normal_hydro_does_not_become_hydroclimatic_candidate() -> None:
    primary, flags, _ = classify_episode(
        pd.Series(
            {
                "episode_quality_risk_score_max": 0.0,
                "episode_vegetation_anomaly_score_max": 1.0,
                "episode_climatic_pressure_score_max": 0.15,
                "episode_consistency_score_max": 0.15,
                "episode_phenology_stage_main": "summer_stress_window",
                "highest_vpd_state": "normal",
            }
        ),
        _rules(),
    )

    assert primary == "unexplained_vegetation_anomaly"
    assert "hydroclimatic_candidate" not in flags


def test_april_episode_has_high_phenology_interpretation_risk() -> None:
    rules = _rules()
    stage = phenology_stage("2026-04-21", rules)

    assert stage == "early_season"
    assert phenology_interpretation_risk(stage, rules) == "high"


def test_poor_quality_overrides_agronomic_interpretation() -> None:
    primary, flags, _ = classify_episode(
        pd.Series(
            {
                "episode_quality_risk_score_max": 1.0,
                "episode_vegetation_anomaly_score_max": 1.0,
                "episode_climatic_pressure_score_max": 1.0,
                "episode_consistency_score_max": 1.0,
                "episode_phenology_stage_main": "summer_stress_window",
                "highest_vpd_state": "very_high",
            }
        ),
        _rules(),
    )

    assert primary == "quality_or_artifact_suspect"
    assert "hydroclimatic_candidate" in flags


def test_obsolete_satellite_data_triggers_degraded_mode() -> None:
    assert satellite_gap_state(13, _rules()) == "obsolete"


def test_null_travel_distance_does_not_generate_zero_co2() -> None:
    cost = physical_cost_for_action("field_check", _costs())

    assert cost["physical_travel_km_low"] is None
    assert cost["physical_travel_km_high"] is None
    assert cost["estimated_kgco2e_low"] is None
    assert cost["estimated_kgco2e_high"] is None


def test_consistency_score_is_limited_by_weakest_signal() -> None:
    assert consistency_score(1.0, 0.25, _rules()) == 0.25


def test_simple_baseline_flags_are_row_local() -> None:
    frame = pd.DataFrame(
        {
            "hydro_context": ["very_dry", "normal"],
            "vpd_state": ["very_high", "normal"],
            "phenology_stage": ["summer_stress_window", "summer_stress_window"],
            "ndmi_percentile": [8.0, 8.0],
            "is_candidate": [True, True],
        }
    )

    result = add_simple_baseline_columns(frame, _simple())

    assert result["simple_weather_alert"].tolist() == [True, False]
    assert result["simple_vegetation_alert"].tolist() == [True, True]
    assert result["simple_combined_alert"].tolist() == [True, False]


def test_v0_2_does_not_modify_v0_1_inputs(tmp_path: Path) -> None:
    decision_file = tmp_path / "eve_decision_points_site_999.csv"
    episode_file = tmp_path / "eve_episodes_site_999.csv"
    pd.DataFrame(
        {
            "decision_point_id": ["site_999_20220801"],
            "parcel_id": ["site_999"],
            "acquisition_date": ["2022-08-01"],
            "in_scope_season": [True],
            "eligible_for_analysis": [True],
            "eligibility_reason": ["eligible"],
            "ndmi_percentile": [4.0],
            "hydro_context": ["very_dry"],
            "vpd_state": ["very_high"],
            "satellite_quality": ["good"],
            "valid_pixel_ratio": [1.0],
            "recommended_information_action": ["priority_field_check"],
            "is_candidate": [True],
            "manual_review_possible": [False],
            "delay_cost_estimated": [False],
            "human_decision_available": [False],
            "episode_id": ["site_999_episode_20220801_01"],
            "eve_decision": ["candidate_priority_field_check"],
        }
    ).to_csv(decision_file, index=False)
    pd.DataFrame(
        {
            "episode_id": ["site_999_episode_20220801_01"],
            "parcel_id": ["site_999"],
            "episode_start_date": ["2022-08-01"],
            "episode_end_date": ["2022-08-01"],
            "peak_date": ["2022-08-01"],
            "duration_days": [0],
            "decision_point_count": [1],
            "lowest_ndmi_percentile": [4.0],
            "worst_ndmi_anomaly": [-0.1],
            "dominant_hydro_context": ["very_dry"],
            "highest_vpd_state": ["very_high"],
            "maximum_urgency": ["high"],
            "proposed_action": ["priority_field_check"],
        }
    ).to_csv(episode_file, index=False)
    before_decision = decision_file.read_text(encoding="utf-8")
    before_episode = episode_file.read_text(encoding="utf-8")

    run(
        site_id="site_999",
        processed_dir=tmp_path,
        rules_config_file=RULES_CONFIG,
        costs_config_file=COSTS_CONFIG,
        simple_baselines_config_file=SIMPLE_CONFIG,
    )

    assert decision_file.read_text(encoding="utf-8") == before_decision
    assert episode_file.read_text(encoding="utf-8") == before_episode
    assert (tmp_path / "eve_decision_points_site_999_v0_2.csv").exists()
    assert (tmp_path / "eve_episodes_site_999_v0_2.csv").exists()
