from __future__ import annotations

from typing import Any

import pandas as pd


def choose_primary_type(flags: list[str], config: dict[str, Any]) -> str:
    priority = config["typology"]["priority"]
    for candidate in priority:
        if candidate in flags:
            return candidate
    return flags[0] if flags else "weak_or_unclear_signal"


def episode_type_confidence(primary: str, row: pd.Series) -> float:
    quality = float(row.get("episode_quality_risk_score_max", 0.0) or 0.0)
    consistency = row.get("episode_consistency_score_max")
    vegetation = row.get("episode_vegetation_anomaly_score_max")
    climate = row.get("episode_climatic_pressure_score_max")

    consistency_value = 0.0 if pd.isna(consistency) else float(consistency)
    vegetation_value = 0.0 if pd.isna(vegetation) else float(vegetation)
    climate_value = 0.0 if pd.isna(climate) else float(climate)

    if primary == "quality_or_artifact_suspect":
        return 0.75
    if primary == "hydroclimatic_candidate":
        return max(0.55, min(0.90, consistency_value))
    if primary == "atmospheric_demand_candidate":
        return max(0.50, min(0.80, climate_value))
    if primary == "unexplained_vegetation_anomaly":
        return max(0.55, min(0.80, vegetation_value - climate_value + 0.35))
    if primary == "phenology_transition":
        return 0.60 if quality < 0.70 else 0.45
    if primary == "management_effect_possible":
        return 0.45
    return 0.35


def classify_episode(row: pd.Series, config: dict[str, Any]) -> tuple[str, str, float]:
    typology = config["typology"]
    flags: list[str] = []
    quality = float(row.get("episode_quality_risk_score_max", 0.0) or 0.0)
    vegetation = row.get("episode_vegetation_anomaly_score_max")
    climate = row.get("episode_climatic_pressure_score_max")
    vegetation_value = 0.0 if pd.isna(vegetation) else float(vegetation)
    climate_value = 0.0 if pd.isna(climate) else float(climate)
    stage = str(row.get("episode_phenology_stage_main", "unknown"))
    vpd_state = str(row.get("highest_vpd_state", "unknown"))

    if quality >= float(typology["quality_risk_gte"]):
        flags.append("quality_or_artifact_suspect")

    if (
        stage in set(typology["transition_stages"])
        and vegetation_value >= float(typology["vegetation_anomaly_gte"])
    ):
        flags.append("phenology_transition")

    if (
        vegetation_value >= float(typology["vegetation_anomaly_gte"])
        and climate_value >= float(typology["climatic_pressure_gte"])
    ):
        flags.append("hydroclimatic_candidate")

    if (
        vpd_state in set(typology["vpd_pressure_states"])
        and stage == "summer_stress_window"
        and climate_value >= float(typology["climatic_pressure_gte"])
    ):
        flags.append("atmospheric_demand_candidate")

    if (
        vegetation_value >= float(typology["strong_vegetation_anomaly_gte"])
        and climate_value < float(typology["weak_climatic_pressure_lt"])
    ):
        flags.append("unexplained_vegetation_anomaly")
        flags.append("management_effect_possible")

    if not flags:
        flags.append("weak_or_unclear_signal")

    deduplicated = list(dict.fromkeys(flags))
    primary = choose_primary_type(deduplicated, config)
    confidence = episode_type_confidence(primary, row)
    return primary, "|".join(deduplicated), round(float(confidence), 3)
