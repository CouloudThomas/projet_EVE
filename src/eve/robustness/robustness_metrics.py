from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


def _value_counts(series: pd.Series) -> dict[str, int]:
    return {
        str(key): int(value)
        for key, value in series.fillna("missing").astype(str).value_counts().items()
    }


def build_robustness_summary(
    decision_points_v0_2: pd.DataFrame,
    episodes_v0_2: pd.DataFrame,
    manual_audit: pd.DataFrame,
    *,
    site_id: str,
    generated_at: str | None = None,
) -> dict[str, Any]:
    generated_at = generated_at or datetime.now(timezone.utc).replace(
        microsecond=0
    ).isoformat()
    eligible = decision_points_v0_2[
        decision_points_v0_2["eligible_for_analysis"] == True  # noqa: E712
    ]
    candidates = decision_points_v0_2[
        decision_points_v0_2["is_candidate"] == True  # noqa: E712
    ]
    poor_abstentions = decision_points_v0_2[
        decision_points_v0_2["eligibility_reason"].eq(
            "insufficient_satellite_observation"
        )
    ]
    wait_decisions = decision_points_v0_2[
        decision_points_v0_2["eve_decision"].eq("wait")
    ]

    summary = {
        "site_id": site_id,
        "version": "eve_v0_2",
        "generated_at": generated_at,
        "eligible_points": int(len(eligible)),
        "candidate_dates": int(len(candidates)),
        "episodes": int(len(episodes_v0_2)),
        "manual_audit_episodes": int(len(manual_audit)),
        "poor_quality_abstentions": int(len(poor_abstentions)),
        "wait_decisions": int(len(wait_decisions)),
        "manual_review_candidates": int(
            episodes_v0_2["proposed_action"].isin(
                ["review_raster", "field_check", "priority_field_check"]
            ).sum()
        ),
        "priority_candidates": int(
            episodes_v0_2["proposed_action"].eq("priority_field_check").sum()
        ),
        "offline_decision_possible_count": int(
            decision_points_v0_2["offline_decision_possible"].sum()
        ),
        "digital_dependency_distribution": _value_counts(
            episodes_v0_2["digital_dependency_level"]
        ),
        "episode_type_distribution": _value_counts(
            episodes_v0_2["episode_type_primary"]
        ),
        "simple_baseline_relation_distribution": _value_counts(
            episodes_v0_2["eve_vs_simple_baseline_relation"]
        ),
        "potential_observations_avoided_vs_all_eligible_review_proxy": int(
            max(len(eligible) - len(manual_audit), 0)
        ),
        "interventions_avoided_real": None,
        "water_saved_real": None,
        "yield_preserved_real": None,
        "drone_flights_avoided_real": None,
        "co2e_saved_real": None,
        "note": (
            "Robustness metrics are proxies unless externally validated. "
            "The avoided-observation metric compares eligible Sentinel decision "
            "points to a manual-audit episode shortlist and is not a measured gain."
        ),
    }
    return summary


def write_robustness_summary(summary: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(summary, file, ensure_ascii=False, indent=2)
        file.write("\n")
