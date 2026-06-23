from __future__ import annotations

from typing import Any

import pandas as pd


USABLE_QUALITY_STATES = {"usable", "good"}


def satellite_gap_state(days: object, config: dict[str, Any]) -> str:
    if pd.isna(days):
        return "unknown"
    value = int(days)
    degraded = config["degraded_mode"]
    if value <= int(degraded["fresh_days_lte"]):
        return "fresh"
    if value <= int(degraded["stale_days_lte"]):
        return "stale"
    return "obsolete"


def degraded_mode_recommendation(row: pd.Series, config: dict[str, Any]) -> str:
    degraded = config["degraded_mode"]
    if row.get("satellite_quality") == "poor":
        return "no_satellite_conclusion_wait_or_use_ground_observation"

    if row.get("satellite_gap_state") == "obsolete":
        if row.get("phenology_stage") == "summer_stress_window":
            return "do_not_conclude_absence_of_risk_apply_low_tech_field_rule"
        return "satellite_information_too_old_use_local_observation"

    climate = row.get("climatic_pressure_score")
    vegetation = row.get("vegetation_anomaly_score")
    if (
        climate is not None
        and pd.notna(climate)
        and vegetation is not None
        and pd.notna(vegetation)
        and float(climate) >= float(degraded["high_climatic_pressure_gte"])
        and float(vegetation) < float(degraded["low_visible_response_lt"])
    ):
        return "climate_exposure_without_visible_response_monitor_locally"

    return "normal_digital_mode"


def add_degraded_mode_columns(
    decision_points: pd.DataFrame,
    config: dict[str, Any],
) -> pd.DataFrame:
    """Ajoute des colonnes de dépendance numérique sans utiliser le futur."""

    result = decision_points.sort_values(
        ["parcel_id", "acquisition_date"]
    ).copy()
    result["last_usable_observation_date"] = pd.NaT

    for _, group in result.groupby("parcel_id", sort=False):
        last_usable: pd.Timestamp | None = None
        for index, row in group.iterrows():
            current_date = pd.Timestamp(row["acquisition_date"])
            if str(row.get("satellite_quality")) in USABLE_QUALITY_STATES:
                last_usable = current_date

            if last_usable is not None:
                result.at[index, "last_usable_observation_date"] = last_usable

    result["days_since_previous_usable_observation"] = (
        pd.to_datetime(result["acquisition_date"])
        - pd.to_datetime(result["last_usable_observation_date"])
    ).dt.days
    result["satellite_gap_state"] = result[
        "days_since_previous_usable_observation"
    ].map(lambda days: satellite_gap_state(days, config))
    result["internet_required"] = result["recommended_information_action"].isin(
        ["review_raster", "priority_field_check"]
    )
    result["degraded_mode_recommendation"] = result.apply(
        lambda row: degraded_mode_recommendation(row, config),
        axis=1,
    )
    result["offline_decision_possible"] = result[
        "degraded_mode_recommendation"
    ].isin(
        [
            "do_not_conclude_absence_of_risk_apply_low_tech_field_rule",
            "satellite_information_too_old_use_local_observation",
            "no_satellite_conclusion_wait_or_use_ground_observation",
            "climate_exposure_without_visible_response_monitor_locally",
        ]
    )
    result["last_usable_observation_date"] = pd.to_datetime(
        result["last_usable_observation_date"]
    ).dt.date
    return result.sort_index()
