from __future__ import annotations

from typing import Any

import pandas as pd


def _is_missing(value: object) -> bool:
    return value is None or pd.isna(value)


def vegetation_anomaly_score(
    ndmi_percentile: object,
    config: dict[str, Any],
) -> float | None:
    """Score de réponse végétale anormale, basé sur le percentile NDMI."""

    if _is_missing(ndmi_percentile):
        return None

    value = float(ndmi_percentile)
    for band in config["scores"]["vegetation_anomaly"]["bands"]:
        if value <= float(band["percentile_lte"]):
            return float(band["score"])
    return float(config["scores"]["vegetation_anomaly"]["default_score"])


def climatic_pressure_score(
    hydro_context: object,
    vpd_state: object,
    config: dict[str, Any],
) -> float | None:
    """Score de pression climatique, séparé du signal végétal."""

    hydro = str(hydro_context) if not _is_missing(hydro_context) else "unknown"
    vpd = str(vpd_state) if not _is_missing(vpd_state) else "unknown"
    hydro_scores = config["scores"]["climatic_pressure"]["hydro_scores"]
    if hydro not in hydro_scores:
        return None

    base = float(hydro_scores[hydro])
    bonus = float(
        config["scores"]["climatic_pressure"]["vpd_bonus"].get(vpd, 0.0)
    )
    return min(1.0, base + bonus)


def consistency_score(
    vegetation_score: float | None,
    climate_score: float | None,
    config: dict[str, Any],
) -> float | None:
    """Cohérence conservatrice : limitée par le signal le plus faible."""

    if vegetation_score is None or climate_score is None:
        return None
    method = config["scores"].get("consistency_method", "min")
    if method != "min":
        raise ValueError(f"Méthode de cohérence inconnue : {method}")
    return min(float(vegetation_score), float(climate_score))


def quality_risk_score(
    satellite_quality: object,
    valid_pixel_ratio: object,
    config: dict[str, Any],
) -> float:
    """Risque d'interprétation lié à la qualité d'observation."""

    quality = str(satellite_quality) if not _is_missing(satellite_quality) else "unknown"
    scores = config["scores"]["quality_risk"]["satellite_quality_scores"]
    base = float(scores.get(quality, scores.get("unknown", 0.8)))
    if _is_missing(valid_pixel_ratio):
        return max(base, 0.8)
    ratio_risk = max(0.0, min(1.0, 1.0 - float(valid_pixel_ratio)))
    return max(base, ratio_risk if quality != "good" else base)
