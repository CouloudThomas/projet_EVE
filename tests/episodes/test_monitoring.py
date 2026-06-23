from __future__ import annotations

import pandas as pd

from eve.episodes.monitoring import build_monitoring_report


def test_monitoring_report_checks_core_invariants() -> None:
    decision_points = pd.DataFrame(
        {
            "decision_point_id": ["site_004_20220801", "site_004_20220811"],
            "acquisition_date": ["2022-08-01", "2022-08-11"],
            "in_scope_season": [True, True],
            "eligible_for_analysis": [True, True],
            "eligibility_reason": ["eligible", "eligible"],
            "satellite_quality": ["good", "good"],
            "ndmi_percentile": [4.0, 60.0],
            "rule_id": ["R6", "R2"],
            "recommended_information_action": [
                "priority_field_check",
                "none",
            ],
            "urgency": ["high", "none"],
            "is_candidate": [True, False],
            "episode_id": ["site_004_episode_20220801_01", pd.NA],
        }
    )
    episodes = pd.DataFrame(
        {
            "episode_id": ["site_004_episode_20220801_01"],
            "episode_start_date": ["2022-08-01"],
            "episode_end_date": ["2022-08-01"],
            "duration_days": [0],
            "decision_point_count": [1],
            "proposed_action": ["priority_field_check"],
            "dominant_hydro_context": ["very_dry"],
            "highest_vpd_state": ["very_high"],
            "persistence_state": ["single"],
            "lowest_ndmi_percentile": [4.0],
            "worst_ndmi_anomaly": [-0.15],
            "estimated_time_min_low": [60],
            "estimated_time_min_high": [120],
            "estimated_cost_eur_low": [20.0],
            "estimated_cost_eur_high": [70.0],
        }
    )

    report = build_monitoring_report(
        decision_points,
        episodes,
        site_id="site_004",
        generated_at="2026-06-23T12:00:00+00:00",
    )

    assert report["status"] == "ok"
    assert report["summary"]["decision_points"] == 2
    assert report["summary"]["candidates"] == 1
    assert report["summary"]["episodes"] == 1
    assert report["invariants"]["candidate_without_episode"] == 0
    assert report["invariants"]["poor_satellite_candidate"] == 0
    assert report["episode_costs"]["priority_field_check"]["time_min_high"] == 120
