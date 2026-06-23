from __future__ import annotations

from typing import Any

import pandas as pd


RISK_ORDER = {"unknown": 0, "low": 1, "medium": 2, "high": 3}


def phenology_stage(value: object, config: dict[str, Any]) -> str:
    """Retourne un stade phénologique grossier à partir du mois."""

    if pd.isna(value):
        return "unknown"
    month = int(pd.to_datetime(value).month)
    return config["phenology"]["month_to_stage"].get(str(month), "unknown")


def phenology_interpretation_risk(stage: str, config: dict[str, Any]) -> str:
    """Niveau de prudence pour interpréter un signal satellite."""

    return config["phenology"]["interpretation_risk_by_stage"].get(
        stage,
        "unknown",
    )


def strongest_risk(values: pd.Series) -> str:
    known_values = [value for value in values.dropna().astype(str) if value in RISK_ORDER]
    if not known_values:
        return "unknown"
    return max(known_values, key=lambda value: RISK_ORDER[value])


def main_stage(values: pd.Series) -> str:
    known_values = [value for value in values.dropna().astype(str) if value != "unknown"]
    if not known_values:
        return "unknown"
    counts = pd.Series(known_values).value_counts()
    return str(counts.index[0])
