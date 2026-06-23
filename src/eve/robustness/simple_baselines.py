from __future__ import annotations

from typing import Any

import pandas as pd


def simple_weather_alert(row: pd.Series, config: dict[str, Any]) -> bool:
    rule = config["simple_weather_rule"]
    return (
        str(row.get("hydro_context")) in set(rule["hydro_context_in"])
        and str(row.get("vpd_state")) in set(rule["vpd_state_in"])
        and str(row.get("phenology_stage")) == rule["phenology_stage"]
    )


def simple_vegetation_alert(row: pd.Series, config: dict[str, Any]) -> bool:
    value = row.get("ndmi_percentile")
    if pd.isna(value):
        return False
    return float(value) <= float(
        config["simple_vegetation_rule"]["ndmi_percentile_lte"]
    )


def simple_combined_alert(row: pd.Series, config: dict[str, Any]) -> bool:
    rule = config["simple_combined_rule"]
    weather_ok = (
        simple_weather_alert(row, config)
        if rule.get("requires_simple_weather_alert", True)
        else True
    )
    value = row.get("ndmi_percentile")
    if pd.isna(value):
        return False
    return weather_ok and float(value) <= float(rule["ndmi_percentile_lte"])


def relation_to_simple_baseline(
    eve_alert: bool,
    simple_alert: bool,
) -> str:
    if eve_alert and simple_alert:
        return "both_alert"
    if eve_alert and not simple_alert:
        return "eve_only"
    if simple_alert and not eve_alert:
        return "simple_only"
    return "neither_alert"


def add_simple_baseline_columns(
    decision_points: pd.DataFrame,
    config: dict[str, Any],
) -> pd.DataFrame:
    result = decision_points.copy()
    result["simple_weather_alert"] = result.apply(
        lambda row: simple_weather_alert(row, config),
        axis=1,
    )
    result["simple_vegetation_alert"] = result.apply(
        lambda row: simple_vegetation_alert(row, config),
        axis=1,
    )
    result["simple_combined_alert"] = result.apply(
        lambda row: simple_combined_alert(row, config),
        axis=1,
    )
    result["eve_alert"] = result["is_candidate"].fillna(False).astype(bool)
    result["eve_vs_simple_baseline_relation"] = result.apply(
        lambda row: relation_to_simple_baseline(
            bool(row["eve_alert"]),
            bool(row["simple_combined_alert"]),
        ),
        axis=1,
    )
    result["eve_only"] = result["eve_vs_simple_baseline_relation"].eq("eve_only")
    result["simple_only"] = result["eve_vs_simple_baseline_relation"].eq("simple_only")
    result["both_alert"] = result["eve_vs_simple_baseline_relation"].eq("both_alert")
    result["neither_alert"] = result["eve_vs_simple_baseline_relation"].eq("neither_alert")
    return result
