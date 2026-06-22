from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd


@dataclass(frozen=True)
class Decision:
    rule_id: str
    eve_decision: str
    recommended_information_action: str
    urgency: str
    is_candidate: bool
    probable_context: str
    context_confidence: str
    context_evidence: str


def classify_satellite_quality(
    valid_ratio: float | None,
    is_usable: bool,
    config: dict[str, Any],
) -> str:
    if valid_ratio is None or pd.isna(valid_ratio) or not is_usable:
        return "poor"

    ratio = float(valid_ratio)
    if ratio < float(config["usable_valid_ratio_gte"]):
        return "poor"
    if ratio < float(config["good_valid_ratio_gte"]):
        return "usable"
    return "good"


def classify_ndmi(percentile: float | None, config: dict[str, Any]) -> str:
    if percentile is None or pd.isna(percentile):
        return "unknown"
    if percentile <= float(config["very_low_percentile_lte"]):
        return "very_low"
    if percentile <= float(config["low_percentile_lte"]):
        return "low"
    return "normal"


def classify_low_state(
    percentile: float | None,
    threshold: float,
) -> str:
    if percentile is None or pd.isna(percentile):
        return "unknown"
    return "low" if percentile <= threshold else "normal"


def classify_hydro_context(
    water_balance_state: str,
    soil_moisture_state: str,
) -> str:
    states = [water_balance_state, soil_moisture_state]
    if "unknown" in states:
        return "unknown"
    low_count = states.count("low")
    if low_count == 2:
        return "very_dry"
    if low_count == 1:
        return "dry"
    return "normal"


def classify_vpd(percentile: float | None, config: dict[str, Any]) -> str:
    if percentile is None or pd.isna(percentile):
        return "unknown"
    if percentile >= float(config["very_high_percentile_gte"]):
        return "very_high"
    if percentile >= float(config["high_percentile_gte"]):
        return "high"
    return "normal"


def classify_precipitation(
    percentile: float | None,
    config: dict[str, Any],
) -> str:
    if percentile is None or pd.isna(percentile):
        return "unknown"
    if percentile <= float(config["low_percentile_lte"]):
        return "low"
    return "normal"


def propose_decision(row: pd.Series) -> Decision:
    """Applique R0 à R6 dans un ordre explicite et testable."""

    if not bool(row["in_scope_season"]):
        return Decision(
            "R0",
            "out_of_scope",
            "none",
            "none",
            False,
            "not_assessed",
            "none",
            "Date hors de la saison V0.1.",
        )

    if row["satellite_quality"] == "poor":
        return Decision(
            "R1",
            "abstain_insufficient_observation",
            "wait_next_sentinel",
            "none",
            False,
            "insufficient_observability",
            "none",
            "Observation satellite insuffisante pour conclure.",
        )

    if row["ndmi_state"] == "unknown":
        return Decision(
            "R1B",
            "abstain_insufficient_baseline",
            "none",
            "none",
            False,
            "insufficient_reference",
            "none",
            "Référence historique insuffisante pour classer le NDMI.",
        )

    if row["ndmi_state"] == "normal":
        return Decision(
            "R2",
            "wait",
            "none",
            "none",
            False,
            "seasonal_range",
            "none",
            "NDMI dans sa distribution saisonnière habituelle.",
        )

    if row["satellite_quality"] == "usable":
        return Decision(
            "R3",
            "additional_information_candidate",
            "review_raster",
            "low",
            True,
            "unexplained_low_confidence",
            "none",
            "Anomalie NDMI possible, mais qualité seulement utilisable.",
        )

    priority_by_intensity = (
        row["ndmi_state"] == "very_low"
        and row["hydro_context"] == "very_dry"
        and row["vpd_state"] in {"high", "very_high"}
    )
    priority_by_persistence = (
        row["ndmi_state"] in {"low", "very_low"}
        and row["hydro_context"] in {"dry", "very_dry"}
        and row["persistence_state"]
        in {"persistent", "strong_persistent"}
    )

    if priority_by_intensity or priority_by_persistence:
        return Decision(
            "R6",
            "additional_information_candidate",
            "priority_field_check",
            "high",
            True,
            "hydroclimatic_context_possible",
            "low",
            "Signal fort ou persistant, cohérent avec un contexte sec, sans vérité terrain.",
        )

    if row["hydro_context"] in {"dry", "very_dry"}:
        return Decision(
            "R5",
            "additional_information_candidate",
            "field_check",
            "medium",
            True,
            "hydroclimatic_context_possible",
            "low",
            "Anomalie NDMI cohérente avec un contexte hydrique défavorable, sans vérité terrain.",
        )

    return Decision(
        "R4",
        "additional_information_candidate",
        "review_raster",
        "medium",
        True,
        "unexplained",
        "none",
        "Anomalie NDMI non expliquée par le contexte hydrique disponible.",
    )


def build_justification_codes(row: pd.Series) -> list[str]:
    codes: list[str] = [f"rule_{str(row['rule_id']).lower()}"]

    state_codes = {
        "satellite_quality": {
            "poor": "poor_satellite_quality",
            "usable": "usable_satellite_quality",
            "good": "good_satellite_quality",
        },
        "ndmi_state": {
            "low": "ndmi_percentile_le_20",
            "very_low": "ndmi_percentile_le_10",
            "normal": "ndmi_within_seasonal_range",
            "unknown": "ndmi_state_unknown",
        },
        "hydro_context": {
            "dry": "dry_hydro_context",
            "very_dry": "very_dry_hydro_context",
            "normal": "normal_hydro_context",
            "unknown": "hydro_context_unknown",
        },
        "vpd_state": {
            "high": "vpd_percentile_ge_75",
            "very_high": "vpd_percentile_ge_90",
        },
        "persistence_state": {
            "persistent": "persistent_low_signal",
            "strong_persistent": "strong_persistent_low_signal",
        },
        "precipitation_state": {
            "low": "low_precipitation_context",
        },
    }

    for column, mapping in state_codes.items():
        value = row.get(column)
        code = mapping.get(value)
        if code:
            codes.append(code)

    return codes


def _format_number(value: Any, digits: int = 3) -> str:
    if value is None or pd.isna(value):
        return "inconnue"
    return f"{float(value):.{digits}f}"


def build_justification_text(row: pd.Series) -> str:
    if row["rule_id"] == "R0":
        return "Date hors de la saison d'analyse EVE V0.1 (avril à octobre)."

    ratio = _format_number(
        100.0 * row["valid_pixel_ratio"]
        if pd.notna(row["valid_pixel_ratio"])
        else None,
        1,
    )

    if row["rule_id"] == "R1":
        return (
            "Observation Sentinel insuffisante : "
            f"{ratio} % de pixels valides. EVE s'abstient."
        )

    if row["rule_id"] == "R1B":
        return "Référence historique insuffisante pour classer cette date."

    parts = [
        (
            f"NDMI={_format_number(row['ndmi_value'])}, "
            f"anomalie={_format_number(row['ndmi_anomaly_value'])}, "
            f"percentile mensuel={_format_number(row['ndmi_percentile'], 1)}."
        ),
        f"Qualité satellite={row['satellite_quality']} ({ratio} % valides).",
    ]

    if row["ndmi_state"] != "normal":
        parts.append(
            "Contexte hydrique="
            f"{row['hydro_context']} "
            "(bilan hydrique p="
            f"{_format_number(row['water_balance_30d_percentile'], 1)}, "
            "humidité du sol p="
            f"{_format_number(row['soil_moisture_30d_percentile'], 1)})."
        )
        parts.append(
            f"VPD={row['vpd_state']} "
            f"(p={_format_number(row['vpd_7d_percentile'], 1)})."
        )
        if row["persistence_state"] in {
            "persistent",
            "strong_persistent",
        }:
            parts.append(
                "Signal bas observé "
                f"{int(row['low_signal_count_15d'])} fois sur 15 jours."
            )

    parts.append(f"Décision V0.1 issue de {row['rule_id']}.")
    return " ".join(parts)


def cost_for_action(
    action: str,
    cost_config: dict[str, Any],
) -> dict[str, Any]:
    actions = cost_config["actions"]
    if action not in actions:
        raise KeyError(f"Aucun coût configuré pour l'action : {action}")
    return actions[action]
