from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from eve.episodes.baselines import (
    ReferenceDistribution,
    build_baseline_artifact,
    build_reference_cache,
    distribution_is_sufficient,
    load_structured_config,
    parse_boolean_series,
    percentile_rank,
)
from eve.episodes.grouping import (
    add_causal_persistence,
    assign_episode_ids,
    build_episode_table,
)
from eve.episodes.monitoring import (
    build_monitoring_report,
    write_monitoring_artifacts,
)
from eve.episodes.rules import (
    build_justification_codes,
    build_justification_text,
    classify_hydro_context,
    classify_low_state,
    classify_ndmi,
    classify_precipitation,
    classify_satellite_quality,
    classify_vpd,
    cost_for_action,
    propose_decision,
)


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
DEFAULT_RULES_CONFIG = PROJECT_ROOT / "config" / "episode_rules_v0_1.yaml"
DEFAULT_COSTS_CONFIG = (
    PROJECT_ROOT / "config" / "verification_costs_v0_1.yaml"
)

ACTION_EXPORTS = {
    "review_raster": "eve_episodes_review_raster_{site_id}.csv",
    "field_check": "eve_episodes_field_check_{site_id}.csv",
    "priority_field_check": "eve_episodes_priority_field_check_{site_id}.csv",
}

AUDIT_SHORTLIST_FILE = "eve_episodes_audit_shortlist_{site_id}.csv"
INFORMATION_QUEUE_FILE = "eve_episodes_information_queue_{site_id}.csv"
AUDIT_LONG_DURATION_DAYS = 30
AUDIT_EXTREME_NDMI_PERCENTILE = 5.0


DECISION_POINT_COLUMNS = [
    "decision_point_id",
    "parcel_id",
    "acquisition_date",
    "evaluation_mode",
    "rules_version",
    "baseline_version",
    "baseline_reference_years",
    "baseline_start_date",
    "baseline_end_date",
    "generated_at",
    "in_scope_season",
    "eligible_for_analysis",
    "eligibility_reason",
    "ndmi_value",
    "ndmi_monthly_median",
    "ndmi_anomaly_value",
    "ndmi_percentile",
    "ndmi_reference_count",
    "ndmi_state",
    "valid_pixel_ratio",
    "satellite_quality",
    "water_balance_30d_value",
    "water_balance_30d_monthly_median",
    "water_balance_30d_anomaly_value",
    "water_balance_30d_percentile",
    "water_balance_reference_count",
    "water_balance_state",
    "soil_moisture_30d_value",
    "soil_moisture_30d_monthly_median",
    "soil_moisture_30d_anomaly_value",
    "soil_moisture_30d_percentile",
    "soil_moisture_reference_count",
    "soil_moisture_state",
    "vpd_7d_value",
    "vpd_7d_monthly_median",
    "vpd_7d_anomaly_value",
    "vpd_7d_percentile",
    "vpd_reference_count",
    "vpd_state",
    "precipitation_30d_value",
    "precipitation_30d_monthly_median",
    "precipitation_30d_anomaly_value",
    "precipitation_30d_percentile",
    "precipitation_reference_count",
    "precipitation_state",
    "hydro_context",
    "low_signal_count_15d",
    "persistence_state",
    "rule_id",
    "eve_decision",
    "recommended_information_action",
    "urgency",
    "is_candidate",
    "manual_review_possible",
    "justification_codes",
    "justification_text",
    "probable_context",
    "context_confidence",
    "context_evidence",
    "cost_model_version",
    "estimated_time_min_low",
    "estimated_time_min_high",
    "estimated_cost_eur_low",
    "estimated_cost_eur_high",
    "cost_status",
    "delay_cost_estimated",
    "verification_status",
    "verification_method",
    "verification_result",
    "human_decision_available",
    "human_final_decision",
    "episode_id",
]


def _sort_episode_queue(episodes: pd.DataFrame) -> pd.DataFrame:
    """Trie les épisodes dans un ordre utile pour l'audit humain."""

    if episodes.empty:
        return episodes.copy()

    action_order = {
        "priority_field_check": 1,
        "field_check": 2,
        "review_raster": 3,
    }
    urgency_order = {"high": 1, "medium": 2, "low": 3, "none": 4}
    result = episodes.copy()
    result["_action_order"] = result["proposed_action"].map(
        action_order
    ).fillna(99)
    result["_urgency_order"] = result["maximum_urgency"].map(
        urgency_order
    ).fillna(99)
    sort_columns = [
        "_action_order",
        "_urgency_order",
        "lowest_ndmi_percentile",
        "duration_days",
        "episode_start_date",
    ]
    ascending = [True, True, True, False, True]
    result = result.sort_values(sort_columns, ascending=ascending)
    return result.drop(columns=["_action_order", "_urgency_order"])


def _audit_reasons(row: pd.Series) -> str:
    reasons: list[str] = []
    if row["proposed_action"] == "priority_field_check":
        reasons.append("priority_field_check")
    if row["proposed_action"] == "field_check":
        reasons.append("field_check")
    if int(row["duration_days"]) >= AUDIT_LONG_DURATION_DAYS:
        reasons.append(f"duration_gte_{AUDIT_LONG_DURATION_DAYS}d")
    if float(row["lowest_ndmi_percentile"]) <= AUDIT_EXTREME_NDMI_PERCENTILE:
        reasons.append(f"lowest_ndmi_percentile_lte_{AUDIT_EXTREME_NDMI_PERCENTILE:g}")
    return "|".join(reasons)


def build_episode_export_tables(
    episodes: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    """Construit les files d'audit opérationnel à partir des épisodes.

    Les exports séparent les actions pour éviter qu'un épisode terrain non
    prioritaire disparaisse d'une short-list raster. La short-list d'audit
    inclut explicitement les épisodes priority_field_check, field_check,
    les épisodes longs et les épisodes NDMI très extrêmes.
    """

    if episodes.empty:
        return {
            "information_queue": episodes.copy(),
            "audit_shortlist": episodes.copy(),
            "review_raster": episodes.copy(),
            "field_check": episodes.copy(),
            "priority_field_check": episodes.copy(),
        }

    queue = episodes[
        episodes["proposed_action"].isin(ACTION_EXPORTS.keys())
    ].copy()
    queue = _sort_episode_queue(queue)

    audit_mask = (
        queue["proposed_action"].isin(
            ["priority_field_check", "field_check"]
        )
        | (queue["duration_days"] >= AUDIT_LONG_DURATION_DAYS)
        | (queue["lowest_ndmi_percentile"] <= AUDIT_EXTREME_NDMI_PERCENTILE)
    )
    audit_shortlist = queue[audit_mask].copy()
    if not audit_shortlist.empty:
        audit_shortlist.insert(
            0,
            "audit_selection_reasons",
            audit_shortlist.apply(_audit_reasons, axis=1),
        )

    exports = {
        "information_queue": queue,
        "audit_shortlist": audit_shortlist,
    }
    for action in ACTION_EXPORTS:
        exports[action] = _sort_episode_queue(
            episodes[episodes["proposed_action"].eq(action)].copy()
        )
    return exports


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Construit les points de décision et épisodes EVE-Sense V0.1."
        )
    )
    parser.add_argument("--site", default="site_004")
    parser.add_argument("--processed-dir", type=Path, default=DEFAULT_PROCESSED_DIR)
    parser.add_argument("--rules-config", type=Path, default=DEFAULT_RULES_CONFIG)
    parser.add_argument("--costs-config", type=Path, default=DEFAULT_COSTS_CONFIG)
    parser.add_argument(
        "--evaluation-mode",
        choices=("retrospective_loyo", "prospective"),
        default=None,
    )
    return parser.parse_args()


def _required_columns(config: dict[str, Any]) -> tuple[set[str], set[str]]:
    columns = config["columns"]
    vegetation_columns = {
        "site_id",
        "acquisition_date",
        columns["ndmi"],
        columns["valid_pixel_ratio"],
        columns["is_usable"],
        columns["water_balance"],
        columns["soil_moisture"],
        columns["vpd"],
        columns["precipitation"],
    }
    weather_columns = {
        "site_id",
        "date",
        columns["water_balance"],
        columns["soil_moisture"],
        columns["vpd"],
        columns["precipitation"],
    }
    return vegetation_columns, weather_columns


def _validate_columns(
    dataframe: pd.DataFrame,
    expected: set[str],
    dataset_name: str,
) -> None:
    missing = sorted(expected.difference(dataframe.columns))
    if missing:
        raise ValueError(
            f"Colonnes absentes dans {dataset_name} : " + ", ".join(missing)
        )


def load_sources(
    vegetation_file: Path,
    weather_file: Path,
    config: dict[str, Any],
    site_id: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not vegetation_file.exists():
        raise FileNotFoundError(f"Fichier végétation introuvable : {vegetation_file}")
    if not weather_file.exists():
        raise FileNotFoundError(f"Fichier météo introuvable : {weather_file}")

    vegetation = pd.read_csv(vegetation_file, parse_dates=["acquisition_date"])
    weather = pd.read_csv(weather_file, parse_dates=["date"])
    vegetation_expected, weather_expected = _required_columns(config)
    _validate_columns(vegetation, vegetation_expected, "végétation-météo")
    _validate_columns(weather, weather_expected, "météo quotidienne")

    vegetation_sites = set(vegetation["site_id"].dropna().astype(str).unique())
    weather_sites = set(weather["site_id"].dropna().astype(str).unique())
    if vegetation_sites != {site_id}:
        raise ValueError(
            f"Sites inattendus dans le fichier fusionné : {sorted(vegetation_sites)}"
        )
    if weather_sites != {site_id}:
        raise ValueError(
            f"Sites inattendus dans le fichier météo : {sorted(weather_sites)}"
        )

    columns = config["columns"]
    vegetation[columns["is_usable"]] = parse_boolean_series(
        vegetation[columns["is_usable"]]
    )
    numeric_columns = [
        columns["ndmi"],
        columns["valid_pixel_ratio"],
        columns["water_balance"],
        columns["soil_moisture"],
        columns["vpd"],
        columns["precipitation"],
    ]
    for column in numeric_columns:
        vegetation[column] = pd.to_numeric(vegetation[column], errors="coerce")
    for column in numeric_columns[2:]:
        weather[column] = pd.to_numeric(weather[column], errors="coerce")

    if vegetation["acquisition_date"].isna().any():
        raise ValueError("Dates Sentinel invalides.")
    if weather["date"].isna().any():
        raise ValueError("Dates météo invalides.")
    if vegetation["acquisition_date"].duplicated().any():
        raise ValueError("Dates Sentinel dupliquées.")
    if weather["date"].duplicated().any():
        raise ValueError("Dates météo dupliquées.")

    return (
        vegetation.sort_values("acquisition_date").reset_index(drop=True),
        weather.sort_values("date").reset_index(drop=True),
    )


def _distribution_features(
    value: float | None,
    distribution: ReferenceDistribution | None,
    minimum_count: int,
) -> tuple[float | None, float | None, float | None, int]:
    if distribution is None:
        return None, None, None, 0

    count = distribution.count
    if not distribution_is_sufficient(distribution, minimum_count):
        return distribution.median, None, None, count

    anomaly = (
        float(value) - float(distribution.median)
        if value is not None
        and pd.notna(value)
        and distribution.median is not None
        else None
    )
    return (
        distribution.median,
        anomaly,
        percentile_rank(value, distribution.values),
        count,
    )


def build_decision_points(
    vegetation: pd.DataFrame,
    weather_daily: pd.DataFrame,
    *,
    site_id: str,
    rules_config: dict[str, Any],
    costs_config: dict[str, Any],
    evaluation_mode: str,
    generated_at: str,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    columns = rules_config["columns"]
    season_months = set(rules_config["season_months"])
    minimum_count = int(
        rules_config["baseline"]["minimum_reference_count"]
    )
    baseline_version = (
        f"{rules_config['baseline']['version']}_{site_id}_{evaluation_mode}"
    )
    target_years = vegetation["acquisition_date"].dt.year.unique()
    cache = build_reference_cache(
        vegetation=vegetation,
        weather_daily=weather_daily,
        target_years=target_years,
        config=rules_config,
        evaluation_mode=evaluation_mode,
    )
    baseline_artifact = build_baseline_artifact(
        cache,
        site_id=site_id,
        config=rules_config,
        evaluation_mode=evaluation_mode,
        baseline_version=baseline_version,
        generated_at=generated_at,
    )

    records: list[dict[str, Any]] = []

    for row in vegetation.itertuples(index=False):
        row_values = row._asdict()
        acquisition_date = pd.Timestamp(row_values["acquisition_date"])
        year = int(acquisition_date.year)
        month = int(acquisition_date.month)
        in_scope = month in season_months
        monthly_cache = cache.get(year, {}).get(month) if in_scope else None
        reference_years = (
            monthly_cache["reference_years"] if monthly_cache else []
        )

        ndmi_value = row_values[columns["ndmi"]]
        valid_ratio = row_values[columns["valid_pixel_ratio"]]
        is_usable = bool(row_values[columns["is_usable"]])
        water_value = row_values[columns["water_balance"]]
        soil_value = row_values[columns["soil_moisture"]]
        vpd_value = row_values[columns["vpd"]]
        precipitation_value = row_values[columns["precipitation"]]

        ndmi_features = _distribution_features(
            ndmi_value,
            monthly_cache["ndmi"] if monthly_cache else None,
            minimum_count,
        )
        water_features = _distribution_features(
            water_value,
            monthly_cache["water_balance"] if monthly_cache else None,
            minimum_count,
        )
        soil_features = _distribution_features(
            soil_value,
            monthly_cache["soil_moisture"] if monthly_cache else None,
            minimum_count,
        )
        vpd_features = _distribution_features(
            vpd_value,
            monthly_cache["vpd"] if monthly_cache else None,
            minimum_count,
        )
        precipitation_features = _distribution_features(
            precipitation_value,
            monthly_cache["precipitation"] if monthly_cache else None,
            minimum_count,
        )

        satellite_quality = classify_satellite_quality(
            valid_ratio,
            is_usable,
            rules_config["satellite_quality"],
        )
        ndmi_state = classify_ndmi(ndmi_features[2], rules_config["ndmi"])
        low_threshold = float(
            rules_config["hydro_context"]["low_percentile_lte"]
        )
        water_state = classify_low_state(water_features[2], low_threshold)
        soil_state = classify_low_state(soil_features[2], low_threshold)
        hydro_context = classify_hydro_context(water_state, soil_state)
        vpd_state = classify_vpd(vpd_features[2], rules_config["vpd"])
        precipitation_state = classify_precipitation(
            precipitation_features[2],
            rules_config["precipitation"],
        )

        if not in_scope:
            eligible = False
            eligibility_reason = "out_of_scope_season"
        elif satellite_quality == "poor":
            eligible = False
            eligibility_reason = "insufficient_satellite_observation"
        elif ndmi_state == "unknown":
            eligible = False
            eligibility_reason = "insufficient_ndmi_reference"
        elif hydro_context == "unknown" or vpd_state == "unknown":
            eligible = True
            eligibility_reason = "eligible_with_incomplete_weather_context"
        else:
            eligible = True
            eligibility_reason = "eligible"

        records.append(
            {
                "decision_point_id": (
                    f"{site_id}_{acquisition_date:%Y%m%d}"
                ),
                "parcel_id": site_id,
                "acquisition_date": acquisition_date,
                "evaluation_mode": evaluation_mode,
                "rules_version": rules_config["rules_version"],
                "baseline_version": baseline_version,
                "baseline_reference_years": "|".join(map(str, reference_years)),
                "baseline_start_date": (
                    f"{min(reference_years)}-01-01" if reference_years else pd.NA
                ),
                "baseline_end_date": (
                    f"{max(reference_years)}-12-31" if reference_years else pd.NA
                ),
                "generated_at": generated_at,
                "in_scope_season": in_scope,
                "eligible_for_analysis": eligible,
                "eligibility_reason": eligibility_reason,
                "ndmi_value": ndmi_value,
                "ndmi_monthly_median": ndmi_features[0],
                "ndmi_anomaly_value": ndmi_features[1],
                "ndmi_percentile": ndmi_features[2],
                "ndmi_reference_count": ndmi_features[3],
                "ndmi_state": ndmi_state,
                "valid_pixel_ratio": valid_ratio,
                "satellite_quality": satellite_quality,
                "water_balance_30d_value": water_value,
                "water_balance_30d_monthly_median": water_features[0],
                "water_balance_30d_anomaly_value": water_features[1],
                "water_balance_30d_percentile": water_features[2],
                "water_balance_reference_count": water_features[3],
                "water_balance_state": water_state,
                "soil_moisture_30d_value": soil_value,
                "soil_moisture_30d_monthly_median": soil_features[0],
                "soil_moisture_30d_anomaly_value": soil_features[1],
                "soil_moisture_30d_percentile": soil_features[2],
                "soil_moisture_reference_count": soil_features[3],
                "soil_moisture_state": soil_state,
                "vpd_7d_value": vpd_value,
                "vpd_7d_monthly_median": vpd_features[0],
                "vpd_7d_anomaly_value": vpd_features[1],
                "vpd_7d_percentile": vpd_features[2],
                "vpd_reference_count": vpd_features[3],
                "vpd_state": vpd_state,
                "precipitation_30d_value": precipitation_value,
                "precipitation_30d_monthly_median": precipitation_features[0],
                "precipitation_30d_anomaly_value": precipitation_features[1],
                "precipitation_30d_percentile": precipitation_features[2],
                "precipitation_reference_count": precipitation_features[3],
                "precipitation_state": precipitation_state,
                "hydro_context": hydro_context,
            }
        )

    decision_points = pd.DataFrame(records)
    persistence = rules_config["persistence"]
    decision_points = add_causal_persistence(
        decision_points,
        lookback_days=int(persistence["lookback_days"]),
        persistent_count=int(persistence["persistent_count_gte"]),
        strong_persistent_count=int(
            persistence["strong_persistent_count_gte"]
        ),
    )

    decisions = decision_points.apply(propose_decision, axis=1)
    decision_points["rule_id"] = [decision.rule_id for decision in decisions]
    decision_points["eve_decision"] = [
        decision.eve_decision for decision in decisions
    ]
    decision_points["recommended_information_action"] = [
        decision.recommended_information_action for decision in decisions
    ]
    decision_points["urgency"] = [decision.urgency for decision in decisions]
    decision_points["is_candidate"] = [
        decision.is_candidate for decision in decisions
    ]
    decision_points["manual_review_possible"] = decision_points[
        "rule_id"
    ].eq("R1")
    decision_points["probable_context"] = [
        decision.probable_context for decision in decisions
    ]
    decision_points["context_confidence"] = [
        decision.context_confidence for decision in decisions
    ]
    decision_points["context_evidence"] = [
        decision.context_evidence for decision in decisions
    ]
    decision_points["justification_codes"] = decision_points.apply(
        lambda row: "|".join(build_justification_codes(row)),
        axis=1,
    )
    decision_points["justification_text"] = decision_points.apply(
        build_justification_text,
        axis=1,
    )

    costs = decision_points["recommended_information_action"].map(
        lambda action: cost_for_action(action, costs_config)
    )
    decision_points["cost_model_version"] = costs_config[
        "cost_model_version"
    ]
    decision_points["estimated_time_min_low"] = [
        cost["time_min_low"] for cost in costs
    ]
    decision_points["estimated_time_min_high"] = [
        cost["time_min_high"] for cost in costs
    ]
    decision_points["estimated_cost_eur_low"] = [
        cost["cost_eur_low"] for cost in costs
    ]
    decision_points["estimated_cost_eur_high"] = [
        cost["cost_eur_high"] for cost in costs
    ]
    decision_points["cost_status"] = costs_config["cost_status"]
    decision_points["delay_cost_estimated"] = [
        cost["delay_cost_estimated"] for cost in costs
    ]
    decision_points["verification_status"] = "not_started"
    decision_points["verification_method"] = "none"
    decision_points["verification_result"] = "not_available"
    decision_points["human_decision_available"] = False
    decision_points["human_final_decision"] = pd.NA

    decision_points = assign_episode_ids(
        decision_points,
        maximum_gap_days=int(
            rules_config["episode_grouping"]["maximum_gap_days"]
        ),
    )
    decision_points = decision_points[DECISION_POINT_COLUMNS].copy()

    return decision_points, baseline_artifact


def write_episode_exports(
    episodes: pd.DataFrame,
    *,
    output_dir: Path,
    site_id: str,
) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    tables = build_episode_export_tables(episodes)
    output_files = {
        "information_queue": output_dir
        / INFORMATION_QUEUE_FILE.format(site_id=site_id),
        "audit_shortlist": output_dir
        / AUDIT_SHORTLIST_FILE.format(site_id=site_id),
    }
    output_files.update(
        {
            action: output_dir / filename.format(site_id=site_id)
            for action, filename in ACTION_EXPORTS.items()
        }
    )

    for name, table in tables.items():
        table.to_csv(output_files[name], index=False, encoding="utf-8")

    return output_files


def write_artifacts(
    decision_points: pd.DataFrame,
    episodes: pd.DataFrame,
    baseline_artifact: dict[str, Any],
    *,
    output_dir: Path,
    site_id: str,
) -> tuple[Path, Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    decision_file = output_dir / f"eve_decision_points_{site_id}.csv"
    episode_file = output_dir / f"eve_episodes_{site_id}.csv"
    baseline_file = output_dir / f"eve_baseline_{site_id}.json"

    decision_output = decision_points.copy()
    decision_output["acquisition_date"] = pd.to_datetime(
        decision_output["acquisition_date"]
    ).dt.date
    decision_output.to_csv(decision_file, index=False, encoding="utf-8")
    episodes.to_csv(episode_file, index=False, encoding="utf-8")
    with baseline_file.open("w", encoding="utf-8") as file:
        json.dump(baseline_artifact, file, ensure_ascii=False, indent=2)
        file.write("\n")
    write_episode_exports(episodes, output_dir=output_dir, site_id=site_id)

    return decision_file, episode_file, baseline_file


def run(
    *,
    site_id: str,
    processed_dir: Path,
    rules_config_file: Path,
    costs_config_file: Path,
    evaluation_mode: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    rules_config = load_structured_config(rules_config_file)
    costs_config = load_structured_config(costs_config_file)
    selected_mode = evaluation_mode or rules_config["evaluation_mode"]
    generated_at = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    vegetation_file = processed_dir / f"vegetation_weather_{site_id}.csv"
    weather_file = processed_dir / f"weather_daily_{site_id}.csv"
    vegetation, weather = load_sources(
        vegetation_file,
        weather_file,
        rules_config,
        site_id,
    )
    decision_points, baseline_artifact = build_decision_points(
        vegetation,
        weather,
        site_id=site_id,
        rules_config=rules_config,
        costs_config=costs_config,
        evaluation_mode=selected_mode,
        generated_at=generated_at,
    )
    episodes = build_episode_table(
        decision_points,
        cost_model_version=costs_config["cost_model_version"],
        cost_status=costs_config["cost_status"],
    )
    write_artifacts(
        decision_points,
        episodes,
        baseline_artifact,
        output_dir=processed_dir,
        site_id=site_id,
    )
    monitoring_report = build_monitoring_report(
        decision_points,
        episodes,
        site_id=site_id,
        generated_at=generated_at,
    )
    write_monitoring_artifacts(
        monitoring_report,
        output_dir=processed_dir,
        site_id=site_id,
    )
    return decision_points, episodes, baseline_artifact


def main() -> None:
    args = parse_arguments()
    decision_points, episodes, baseline = run(
        site_id=args.site,
        processed_dir=args.processed_dir,
        rules_config_file=args.rules_config,
        costs_config_file=args.costs_config,
        evaluation_mode=args.evaluation_mode,
    )
    decision_counts = decision_points["recommended_information_action"].value_counts()

    print("EVE-Sense V0.1 terminé.")
    print(f"Site : {args.site}")
    print(f"Mode : {baseline['evaluation_mode']}")
    print(f"Points de décision : {len(decision_points)}")
    print(f"Candidats : {int(decision_points['is_candidate'].sum())}")
    print(f"Épisodes : {len(episodes)}")
    print("Actions proposées :")
    for action, count in decision_counts.items():
        print(f"- {action}: {count}")

    candidate_years = (
        decision_points.loc[decision_points["is_candidate"]]
        .assign(
            year=lambda frame: pd.to_datetime(
                frame["acquisition_date"]
            ).dt.year
        )
        .groupby("year")
        .size()
    )
    print("Candidats par année :")
    for year, count in candidate_years.items():
        print(f"- {int(year)}: {int(count)}")

    if not episodes.empty:
        episode_actions = episodes["proposed_action"].value_counts()
        print("Épisodes par action maximale :")
        for action, count in episode_actions.items():
            print(f"- {action}: {count}")


if __name__ == "__main__":
    main()
