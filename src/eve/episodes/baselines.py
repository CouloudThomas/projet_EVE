from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ReferenceDistribution:
    """Distribution de référence mensuelle utilisée pour une variable."""

    values: np.ndarray
    median: float | None
    quantiles: dict[str, float | None]

    @property
    def count(self) -> int:
        return int(len(self.values))


def load_structured_config(path: Path) -> dict[str, Any]:
    """Charge un fichier YAML écrit dans le sous-ensemble JSON de YAML 1.2."""

    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)

    if not isinstance(data, dict):
        raise ValueError(f"Configuration invalide : {path}")

    return data


def parse_boolean_series(series: pd.Series) -> pd.Series:
    """Convertit explicitement les booléens CSV sans accepter de valeur ambiguë."""

    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False).astype(bool)

    normalized = series.astype("string").str.strip().str.lower()
    mapping = {
        "true": True,
        "1": True,
        "yes": True,
        "false": False,
        "0": False,
        "no": False,
        "<na>": False,
    }
    unknown = normalized[~normalized.isin(mapping)].dropna().unique()

    if len(unknown):
        raise ValueError(
            "Valeurs booléennes non reconnues : "
            + ", ".join(map(str, unknown))
        )

    return normalized.map(mapping).fillna(False).astype(bool)


def reference_years_for_target(
    target_year: int,
    reference_years: Iterable[int],
    evaluation_mode: str,
) -> list[int]:
    """Détermine les années autorisées pour une année cible."""

    years = sorted({int(year) for year in reference_years})

    if evaluation_mode == "retrospective_loyo":
        return [year for year in years if year != target_year]

    if evaluation_mode == "prospective":
        return [year for year in years if year < target_year]

    raise ValueError(f"Mode d'évaluation inconnu : {evaluation_mode}")


def percentile_rank(
    value: float | int | None,
    reference_values: np.ndarray,
) -> float | None:
    """Retourne le percentile empirique faible (part des références <= valeur)."""

    if value is None or pd.isna(value) or len(reference_values) == 0:
        return None

    numeric_value = float(value)
    return float(100.0 * np.mean(reference_values <= numeric_value))


def _distribution(
    values: pd.Series,
    quantiles: dict[str, float],
) -> ReferenceDistribution:
    numeric = pd.to_numeric(values, errors="coerce").dropna().to_numpy(float)

    if len(numeric) == 0:
        return ReferenceDistribution(
            values=numeric,
            median=None,
            quantiles={name: None for name in quantiles},
        )

    return ReferenceDistribution(
        values=numeric,
        median=float(np.median(numeric)),
        quantiles={
            name: float(np.quantile(numeric, quantile))
            for name, quantile in quantiles.items()
        },
    )


def build_reference_cache(
    vegetation: pd.DataFrame,
    weather_daily: pd.DataFrame,
    target_years: Iterable[int],
    config: dict[str, Any],
    evaluation_mode: str,
) -> dict[int, dict[int, dict[str, Any]]]:
    """Construit les distributions LOYO/prospectives par année cible et mois."""

    columns = config["columns"]
    reference_years = config["baseline"]["reference_years"]
    season_months = config["season_months"]
    minimum_count = int(config["baseline"]["minimum_reference_count"])

    vegetation = vegetation.copy()
    weather_daily = weather_daily.copy()
    vegetation["_year"] = vegetation["acquisition_date"].dt.year
    vegetation["_month"] = vegetation["acquisition_date"].dt.month
    weather_daily["_year"] = weather_daily["date"].dt.year
    weather_daily["_month"] = weather_daily["date"].dt.month

    cache: dict[int, dict[int, dict[str, Any]]] = {}

    for target_year in sorted({int(year) for year in target_years}):
        allowed_years = reference_years_for_target(
            target_year=target_year,
            reference_years=reference_years,
            evaluation_mode=evaluation_mode,
        )
        cache[target_year] = {}

        for month in season_months:
            vegetation_mask = (
                vegetation["_year"].isin(allowed_years)
                & vegetation["_month"].eq(month)
                & vegetation[columns["is_usable"]]
                & vegetation[columns["ndmi"]].notna()
            )
            weather_mask = (
                weather_daily["_year"].isin(allowed_years)
                & weather_daily["_month"].eq(month)
            )

            ndmi = _distribution(
                vegetation.loc[vegetation_mask, columns["ndmi"]],
                {"q10": 0.10, "q20": 0.20},
            )
            weather_distributions = {
                "water_balance": _distribution(
                    weather_daily.loc[weather_mask, columns["water_balance"]],
                    {"q25": 0.25},
                ),
                "soil_moisture": _distribution(
                    weather_daily.loc[weather_mask, columns["soil_moisture"]],
                    {"q25": 0.25},
                ),
                "vpd": _distribution(
                    weather_daily.loc[weather_mask, columns["vpd"]],
                    {"q75": 0.75, "q90": 0.90},
                ),
                "precipitation": _distribution(
                    weather_daily.loc[weather_mask, columns["precipitation"]],
                    {"q25": 0.25},
                ),
            }

            cache[target_year][int(month)] = {
                "reference_years": allowed_years,
                "minimum_reference_count": minimum_count,
                "ndmi": ndmi,
                **weather_distributions,
            }

    return cache


def distribution_is_sufficient(
    distribution: ReferenceDistribution,
    minimum_count: int,
) -> bool:
    return distribution.count >= minimum_count


def build_baseline_artifact(
    cache: dict[int, dict[int, dict[str, Any]]],
    *,
    site_id: str,
    config: dict[str, Any],
    evaluation_mode: str,
    baseline_version: str,
    generated_at: str,
) -> dict[str, Any]:
    """Transforme le cache interne en document JSON traçable."""

    folds: dict[str, Any] = {}

    for target_year, months in cache.items():
        fold_months: dict[str, Any] = {}

        for month, values in months.items():
            variables: dict[str, Any] = {}

            for variable in (
                "ndmi",
                "water_balance",
                "soil_moisture",
                "vpd",
                "precipitation",
            ):
                distribution = values[variable]
                variables[variable] = {
                    "count": distribution.count,
                    "median": distribution.median,
                    **distribution.quantiles,
                }

            fold_months[str(month)] = {
                "reference_years": values["reference_years"],
                "variables": variables,
            }

        folds[str(target_year)] = {"months": fold_months}

    return {
        "schema_version": "1.0",
        "site_id": site_id,
        "baseline_version": baseline_version,
        "evaluation_mode": evaluation_mode,
        "generated_at": generated_at,
        "reference_years": config["baseline"]["reference_years"],
        "season_months": config["season_months"],
        "minimum_reference_count": config["baseline"][
            "minimum_reference_count"
        ],
        "satellite_method": config["baseline"]["satellite_method"],
        "weather_method": config["baseline"]["weather_method"],
        "folds": folds,
    }
