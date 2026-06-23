from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from eve.episodes.baselines import load_structured_config, parse_boolean_series
from eve.robustness.degraded_mode import add_degraded_mode_columns
from eve.robustness.phenology import (
    main_stage,
    phenology_interpretation_risk,
    phenology_stage,
    strongest_risk,
)
from eve.robustness.physical_costs import physical_cost_for_action
from eve.robustness.robustness_metrics import (
    build_robustness_summary,
    write_robustness_summary,
)
from eve.robustness.scores import (
    climatic_pressure_score,
    consistency_score,
    quality_risk_score,
    vegetation_anomaly_score,
)
from eve.robustness.simple_baselines import (
    add_simple_baseline_columns,
    relation_to_simple_baseline,
)
from eve.robustness.typology import classify_episode


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
DEFAULT_RULES_CONFIG = PROJECT_ROOT / "config" / "eve_rules_v0_2.yaml"
DEFAULT_COSTS_CONFIG = PROJECT_ROOT / "config" / "eve_costs_v0_2.yaml"
DEFAULT_SIMPLE_BASELINES_CONFIG = (
    PROJECT_ROOT / "config" / "eve_simple_baselines_v0_2.yaml"
)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Construit la couche EVE V0.2 de robustesse/interprétation."
    )
    parser.add_argument("--site", default="site_004")
    parser.add_argument("--processed-dir", type=Path, default=DEFAULT_PROCESSED_DIR)
    parser.add_argument("--rules-config", type=Path, default=DEFAULT_RULES_CONFIG)
    parser.add_argument("--costs-config", type=Path, default=DEFAULT_COSTS_CONFIG)
    parser.add_argument(
        "--simple-baselines-config",
        type=Path,
        default=DEFAULT_SIMPLE_BASELINES_CONFIG,
    )
    return parser.parse_args()


def _load_v0_1_artifacts(
    processed_dir: Path,
    site_id: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    decision_file = processed_dir / f"eve_decision_points_{site_id}.csv"
    episode_file = processed_dir / f"eve_episodes_{site_id}.csv"
    if not decision_file.exists():
        raise FileNotFoundError(f"Points de décision V0.1 introuvables : {decision_file}")
    if not episode_file.exists():
        raise FileNotFoundError(f"Épisodes V0.1 introuvables : {episode_file}")

    decision_points = pd.read_csv(decision_file, parse_dates=["acquisition_date"])
    episodes = pd.read_csv(
        episode_file,
        parse_dates=["episode_start_date", "episode_end_date", "peak_date"],
    )
    for column in [
        "in_scope_season",
        "eligible_for_analysis",
        "is_candidate",
        "manual_review_possible",
        "delay_cost_estimated",
        "human_decision_available",
    ]:
        if column in decision_points.columns:
            decision_points[column] = parse_boolean_series(decision_points[column])

    return decision_points, episodes


def enrich_decision_points(
    decision_points: pd.DataFrame,
    *,
    rules_config: dict[str, Any],
    simple_config: dict[str, Any],
    generated_at: str,
) -> pd.DataFrame:
    result = decision_points.copy()
    result["eve_v0_2_rules_version"] = rules_config["rules_version"]
    result["eve_v0_2_generated_at"] = generated_at
    result["source_rules_version"] = rules_config["source_rules_version"]

    result["phenology_stage"] = result["acquisition_date"].map(
        lambda value: phenology_stage(value, rules_config)
    )
    result["phenology_interpretation_risk"] = result["phenology_stage"].map(
        lambda stage: phenology_interpretation_risk(stage, rules_config)
    )
    result["vegetation_anomaly_score"] = result["ndmi_percentile"].map(
        lambda value: vegetation_anomaly_score(value, rules_config)
    )
    result["climatic_pressure_score"] = result.apply(
        lambda row: climatic_pressure_score(
            row["hydro_context"],
            row["vpd_state"],
            rules_config,
        ),
        axis=1,
    )
    result["consistency_score"] = result.apply(
        lambda row: consistency_score(
            row["vegetation_anomaly_score"],
            row["climatic_pressure_score"],
            rules_config,
        ),
        axis=1,
    )
    result["quality_risk_score"] = result.apply(
        lambda row: quality_risk_score(
            row["satellite_quality"],
            row["valid_pixel_ratio"],
            rules_config,
        ),
        axis=1,
    )
    result = add_degraded_mode_columns(result, rules_config)
    result = add_simple_baseline_columns(result, simple_config)
    return result


def _risk_order_value(value: str) -> int:
    return {"unknown": 0, "low": 1, "medium": 2, "high": 3}.get(str(value), 0)


def _manual_audit_reasons(row: pd.Series, rules_config: dict[str, Any]) -> str:
    reasons: list[str] = []
    manual_config = rules_config["manual_audit"]
    if row["proposed_action"] == "priority_field_check":
        reasons.append("priority_field_check")
    if row["proposed_action"] == "field_check":
        reasons.append("field_check")
    if int(row["duration_days"]) >= int(manual_config["duration_days_gte"]):
        reasons.append(f"duration_gte_{manual_config['duration_days_gte']}d")
    if float(row["lowest_ndmi_percentile"]) <= float(
        manual_config["lowest_ndmi_percentile_lte"]
    ):
        reasons.append(
            f"lowest_ndmi_percentile_lte_{manual_config['lowest_ndmi_percentile_lte']:g}"
        )
    type_flags = set(str(row.get("episode_type_flags", "")).split("|"))
    for episode_type in manual_config["include_episode_types"]:
        if episode_type in type_flags:
            reasons.append(episode_type)
    if (
        row.get("dominant_hydro_context") == "normal"
        and float(row["lowest_ndmi_percentile"])
        <= float(manual_config["lowest_ndmi_percentile_lte"])
    ):
        reasons.append("normal_hydro_extreme_ndmi")
    return "|".join(dict.fromkeys(reasons))


def enrich_episodes(
    episodes: pd.DataFrame,
    decision_points_v0_2: pd.DataFrame,
    *,
    rules_config: dict[str, Any],
    costs_config: dict[str, Any],
    generated_at: str,
) -> pd.DataFrame:
    if episodes.empty:
        return episodes.copy()

    candidate_points = decision_points_v0_2[
        decision_points_v0_2["episode_id"].notna()
    ].copy()
    grouped = candidate_points.groupby("episode_id", sort=False)
    aggregates = grouped.agg(
        episode_vegetation_anomaly_score_max=("vegetation_anomaly_score", "max"),
        episode_climatic_pressure_score_max=("climatic_pressure_score", "max"),
        episode_consistency_score_max=("consistency_score", "max"),
        episode_quality_risk_score_max=("quality_risk_score", "max"),
        simple_weather_alert_any=("simple_weather_alert", "any"),
        simple_vegetation_alert_any=("simple_vegetation_alert", "any"),
        simple_combined_alert_any=("simple_combined_alert", "any"),
        offline_decision_possible_any=("offline_decision_possible", "any"),
        degraded_mode_recommendation_main=(
            "degraded_mode_recommendation",
            lambda values: values.value_counts().index[0],
        ),
    )
    aggregates["episode_phenology_stage_main"] = grouped[
        "phenology_stage"
    ].agg(main_stage)
    aggregates["episode_phenology_interpretation_risk_max"] = grouped[
        "phenology_interpretation_risk"
    ].agg(strongest_risk)
    aggregates = aggregates.reset_index()

    result = episodes.merge(aggregates, on="episode_id", how="left")
    result["eve_v0_2_rules_version"] = rules_config["rules_version"]
    result["eve_v0_2_generated_at"] = generated_at

    classifications = result.apply(
        lambda row: classify_episode(row, rules_config),
        axis=1,
    )
    result["episode_type_primary"] = [item[0] for item in classifications]
    result["episode_type_flags"] = [item[1] for item in classifications]
    result["episode_type_confidence"] = [item[2] for item in classifications]
    result["eve_alert"] = True
    result["eve_vs_simple_baseline_relation"] = result[
        "simple_combined_alert_any"
    ].map(lambda simple: relation_to_simple_baseline(True, bool(simple)))

    costs = result["proposed_action"].map(
        lambda action: physical_cost_for_action(str(action), costs_config)
    )
    for column in costs.iloc[0].keys():
        result[column] = [cost[column] for cost in costs]

    result["manual_audit_reasons"] = result.apply(
        lambda row: _manual_audit_reasons(row, rules_config),
        axis=1,
    )
    result["requires_manual_audit_v0_2"] = result["manual_audit_reasons"].ne("")
    result = result.sort_values(
        [
            "requires_manual_audit_v0_2",
            "maximum_urgency",
            "lowest_ndmi_percentile",
            "duration_days",
            "episode_start_date",
        ],
        ascending=[False, True, True, False, True],
    )
    result["episode_phenology_interpretation_risk_order"] = result[
        "episode_phenology_interpretation_risk_max"
    ].map(_risk_order_value)
    return result


def build_manual_audit_table(episodes_v0_2: pd.DataFrame) -> pd.DataFrame:
    audit = episodes_v0_2[
        episodes_v0_2["requires_manual_audit_v0_2"] == True  # noqa: E712
    ].copy()
    action_order = {
        "priority_field_check": 1,
        "field_check": 2,
        "review_raster": 3,
    }
    audit["_action_order"] = audit["proposed_action"].map(action_order).fillna(99)
    audit = audit.sort_values(
        [
            "_action_order",
            "episode_type_primary",
            "lowest_ndmi_percentile",
            "duration_days",
            "episode_start_date",
        ],
        ascending=[True, True, True, False, True],
    )
    return audit.drop(columns=["_action_order"])


def write_outputs(
    decision_points_v0_2: pd.DataFrame,
    episodes_v0_2: pd.DataFrame,
    manual_audit: pd.DataFrame,
    robustness_summary: dict[str, Any],
    *,
    processed_dir: Path,
    site_id: str,
) -> tuple[Path, Path, Path, Path]:
    processed_dir.mkdir(parents=True, exist_ok=True)
    decision_file = processed_dir / f"eve_decision_points_{site_id}_v0_2.csv"
    episode_file = processed_dir / f"eve_episodes_{site_id}_v0_2.csv"
    audit_file = processed_dir / f"episodes_for_manual_audit_{site_id}_v0_2.csv"
    summary_file = processed_dir / f"eve_robustness_summary_{site_id}.json"

    decision_output = decision_points_v0_2.copy()
    decision_output["acquisition_date"] = pd.to_datetime(
        decision_output["acquisition_date"]
    ).dt.date
    decision_output.to_csv(decision_file, index=False, encoding="utf-8")
    episodes_v0_2.to_csv(episode_file, index=False, encoding="utf-8")
    manual_audit.to_csv(audit_file, index=False, encoding="utf-8")
    write_robustness_summary(robustness_summary, summary_file)
    return decision_file, episode_file, summary_file, audit_file


def run(
    *,
    site_id: str,
    processed_dir: Path,
    rules_config_file: Path,
    costs_config_file: Path,
    simple_baselines_config_file: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    generated_at = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    rules_config = load_structured_config(rules_config_file)
    costs_config = load_structured_config(costs_config_file)
    simple_config = load_structured_config(simple_baselines_config_file)
    decision_points, episodes = _load_v0_1_artifacts(processed_dir, site_id)
    decision_points_v0_2 = enrich_decision_points(
        decision_points,
        rules_config=rules_config,
        simple_config=simple_config,
        generated_at=generated_at,
    )
    episodes_v0_2 = enrich_episodes(
        episodes,
        decision_points_v0_2,
        rules_config=rules_config,
        costs_config=costs_config,
        generated_at=generated_at,
    )
    manual_audit = build_manual_audit_table(episodes_v0_2)
    summary = build_robustness_summary(
        decision_points_v0_2,
        episodes_v0_2,
        manual_audit,
        site_id=site_id,
        generated_at=generated_at,
    )
    write_outputs(
        decision_points_v0_2,
        episodes_v0_2,
        manual_audit,
        summary,
        processed_dir=processed_dir,
        site_id=site_id,
    )
    return decision_points_v0_2, episodes_v0_2, manual_audit, summary


def main() -> None:
    args = parse_arguments()
    decision_points, episodes, manual_audit, summary = run(
        site_id=args.site,
        processed_dir=args.processed_dir,
        rules_config_file=args.rules_config,
        costs_config_file=args.costs_config,
        simple_baselines_config_file=args.simple_baselines_config,
    )
    print("EVE V0.2 terminé.")
    print(f"Site : {args.site}")
    print(f"Points V0.2 : {len(decision_points)}")
    print(f"Épisodes V0.2 : {len(episodes)}")
    print(f"Épisodes audit manuel V0.2 : {len(manual_audit)}")
    print("Typologie :")
    for label, count in episodes["episode_type_primary"].value_counts().items():
        print(f"- {label}: {count}")
    print(
        "Proxy observations évitées vs revue naïve : "
        f"{summary['potential_observations_avoided_vs_all_eligible_review_proxy']}"
    )


if __name__ == "__main__":
    main()
