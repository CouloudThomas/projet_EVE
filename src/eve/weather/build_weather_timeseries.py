from __future__ import annotations

import argparse
import json
import os
from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd
import requests
from dotenv import load_dotenv
from shapely.geometry import shape


PROJECT_ROOT = Path(__file__).resolve().parents[3]

DEFAULT_PARCELES_FILE = (
    PROJECT_ROOT
    / "data"
    / "input"
    / "parcels.geojson"
)

DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
)

OPEN_METEO_ARCHIVE_URL = (
    "https://archive-api.open-meteo.com/v1/archive"
)

DAILY_VARIABLES = [
    "temperature_2m_mean",
    "temperature_2m_min",
    "temperature_2m_max",
    "precipitation_sum",
    "relative_humidity_2m_mean",
    "vapour_pressure_deficit_max",
    "et0_fao_evapotranspiration",
    "shortwave_radiation_sum",
    "wind_speed_10m_mean",
    "soil_moisture_0_to_7cm_mean",
    "soil_moisture_7_to_28cm_mean",
]

SUPPORTED_MODELS = (
    "era5_seamless",
    "era5",
    "era5_land",
)

MIN_DAILY_COVERAGE = 0.99


def load_environment() -> None:
    env_file = PROJECT_ROOT / ".env"

    if env_file.exists():
        load_dotenv(env_file)


def parse_arguments() -> argparse.Namespace:
    load_environment()

    parser = argparse.ArgumentParser(
        description=(
            "Télécharge une série météo quotidienne "
            "Open-Meteo pour une parcelle EVE."
        )
    )

    parser.add_argument(
        "--site",
        default=os.getenv("EVE_SITE_ID", "site_004"),
        help="Identifiant du site dans parcels.geojson.",
    )

    parser.add_argument(
        "--start-date",
        default=os.getenv("EVE_WEATHER_START_DATE", "2015-07-01"),
        help="Date de début au format YYYY-MM-DD.",
    )

    parser.add_argument(
        "--end-date",
        default=os.getenv("EVE_WEATHER_END_DATE", "2026-06-18"),
        help="Date de fin au format YYYY-MM-DD.",
    )

    parser.add_argument(
        "--model",
        choices=SUPPORTED_MODELS,
        default=os.getenv(
            "EVE_WEATHER_MODEL",
            "era5_seamless",
        ),
        help=(
            "Modèle Open-Meteo. Par défaut : era5_seamless, "
            "qui combine ERA5-Land et ERA5."
        ),
    )

    parser.add_argument(
        "--timezone",
        default=os.getenv("EVE_WEATHER_TIMEZONE", "Europe/Paris"),
        help="Fuseau horaire utilisé pour agréger les journées.",
    )

    parser.add_argument(
        "--parcels-file",
        type=Path,
        default=DEFAULT_PARCELES_FILE,
        help="Chemin vers le GeoJSON des parcelles.",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Dossier de sortie des CSV.",
    )

    return parser.parse_args()


def read_site_feature(
    parcels_file: Path,
    site_id: str,
) -> dict[str, Any]:
    if not parcels_file.exists():
        raise FileNotFoundError(
            f"Fichier introuvable : {parcels_file}"
        )

    with parcels_file.open(
        "r",
        encoding="utf-8",
    ) as file:
        geojson_data = json.load(file)

    for feature in geojson_data.get("features", []):
        properties = feature.get("properties", {})

        if properties.get("site_id") == site_id:
            return feature

    raise ValueError(
        f"Site introuvable dans {parcels_file} : {site_id}"
    )


def get_site_centroid(
    feature: dict[str, Any],
) -> tuple[float, float]:
    geometry = shape(feature["geometry"])

    centroid = geometry.centroid

    longitude = float(centroid.x)
    latitude = float(centroid.y)

    return latitude, longitude


def request_open_meteo_daily_weather(
    latitude: float,
    longitude: float,
    start_date: str,
    end_date: str,
    model: str,
    timezone: str,
) -> dict[str, Any]:
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "start_date": start_date,
        "end_date": end_date,
        "daily": ",".join(DAILY_VARIABLES),
        "timezone": timezone,
        "models": model,
    }

    response = requests.get(
        OPEN_METEO_ARCHIVE_URL,
        params=params,
        timeout=120,
    )

    if not response.ok:
        raise RuntimeError(
            "Erreur Open-Meteo : "
            f"{response.status_code}\n"
            f"URL : {response.url}\n"
            f"Réponse : {response.text}"
        )

    return response.json()


def build_weather_dataframe(
    response_data: dict[str, Any],
    site_id: str,
    latitude: float,
    longitude: float,
    model: str,
    timezone: str,
) -> pd.DataFrame:
    daily_data = response_data.get("daily")

    if not daily_data:
        raise RuntimeError(
            "La réponse Open-Meteo ne contient pas de bloc 'daily'."
        )

    dataframe = pd.DataFrame(daily_data)

    if "time" not in dataframe.columns:
        raise RuntimeError(
            "La réponse Open-Meteo ne contient pas de colonne 'time'."
        )

    dataframe = dataframe.rename(
        columns={
            "time": "date",
        }
    )

    dataframe.insert(
        0,
        "site_id",
        site_id,
    )

    dataframe.insert(
        1,
        "latitude",
        latitude,
    )

    dataframe.insert(
        2,
        "longitude",
        longitude,
    )

    dataframe.insert(
        3,
        "weather_model",
        model,
    )

    dataframe.insert(
        4,
        "timezone",
        timezone,
    )

    dataframe["date"] = pd.to_datetime(
        dataframe["date"],
        errors="coerce",
    )

    for column in DAILY_VARIABLES:
        if column in dataframe.columns:
            dataframe[column] = pd.to_numeric(
                dataframe[column],
                errors="coerce",
            )

    dataframe = dataframe.sort_values("date").reset_index(
        drop=True
    )

    return dataframe


def validate_weather_dataframe(
    dataframe: pd.DataFrame,
    minimum_coverage: float = MIN_DAILY_COVERAGE,
) -> None:
    """Refuse une réponse incomplète avant de remplacer le CSV existant."""
    if dataframe.empty:
        raise RuntimeError("La série météo reçue est vide.")

    missing_columns = [
        column
        for column in ["date", *DAILY_VARIABLES]
        if column not in dataframe.columns
    ]

    if missing_columns:
        raise RuntimeError(
            "Colonnes météo absentes : "
            + ", ".join(missing_columns)
        )

    invalid_dates = int(dataframe["date"].isna().sum())

    if invalid_dates:
        raise RuntimeError(
            f"Dates météo invalides : {invalid_dates}."
        )

    duplicate_dates = int(dataframe["date"].duplicated().sum())

    if duplicate_dates:
        raise RuntimeError(
            f"Dates météo dupliquées : {duplicate_dates}."
        )

    coverage = dataframe[DAILY_VARIABLES].notna().mean()
    insufficient_coverage = coverage[
        coverage < minimum_coverage
    ]

    if not insufficient_coverage.empty:
        details = ", ".join(
            f"{column}={ratio:.1%}"
            for column, ratio in insufficient_coverage.items()
        )
        raise RuntimeError(
            "Couverture météo insuffisante "
            f"(minimum {minimum_coverage:.0%}) : {details}"
        )


def add_rolling_weather_features(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """Calcule uniquement des fenêtres complètes, sans inventer de zéros."""
    dataframe = dataframe.copy()

    dataframe = dataframe.sort_values("date").reset_index(
        drop=True
    )

    windows = [7, 14, 30]

    for window in windows:
        precipitation_column = f"precipitation_{window}d"
        et0_column = f"et0_{window}d"
        water_balance_column = f"water_balance_{window}d"

        dataframe[precipitation_column] = (
            dataframe["precipitation_sum"]
            .rolling(window=window, min_periods=window)
            .sum()
        )

        dataframe[et0_column] = (
            dataframe["et0_fao_evapotranspiration"]
            .rolling(window=window, min_periods=window)
            .sum()
        )

        dataframe[water_balance_column] = (
            dataframe[precipitation_column]
            - dataframe[et0_column]
        )

        dataframe[f"temperature_mean_{window}d"] = (
            dataframe["temperature_2m_mean"]
            .rolling(window=window, min_periods=window)
            .mean()
        )

        dataframe[f"temperature_max_{window}d"] = (
            dataframe["temperature_2m_max"]
            .rolling(window=window, min_periods=window)
            .max()
        )

        dataframe[f"relative_humidity_mean_{window}d"] = (
            dataframe["relative_humidity_2m_mean"]
            .rolling(window=window, min_periods=window)
            .mean()
        )

        dataframe[f"vpd_max_{window}d"] = (
            dataframe["vapour_pressure_deficit_max"]
            .rolling(window=window, min_periods=window)
            .max()
        )

        dataframe[f"radiation_sum_{window}d"] = (
            dataframe["shortwave_radiation_sum"]
            .rolling(window=window, min_periods=window)
            .sum()
        )

        dataframe[f"wind_speed_mean_{window}d"] = (
            dataframe["wind_speed_10m_mean"]
            .rolling(window=window, min_periods=window)
            .mean()
        )

        dataframe[f"soil_moisture_0_7cm_mean_{window}d"] = (
            dataframe["soil_moisture_0_to_7cm_mean"]
            .rolling(window=window, min_periods=window)
            .mean()
        )

        dataframe[f"soil_moisture_7_28cm_mean_{window}d"] = (
            dataframe["soil_moisture_7_to_28cm_mean"]
            .rolling(window=window, min_periods=window)
            .mean()
        )

        rainy_day = (
            dataframe["precipitation_sum"]
            .gt(1.0)
            .astype(float)
            .where(dataframe["precipitation_sum"].notna())
        )

        dataframe[f"rainy_days_{window}d"] = (
            rainy_day
            .rolling(window=window, min_periods=window)
            .sum()
        )

    dataframe["dryness_index_30d"] = (
        dataframe["et0_30d"]
        - dataframe["precipitation_30d"]
    )

    return dataframe


def main() -> None:
    args = parse_arguments()

    site_id = args.site
    start_date = date.fromisoformat(args.start_date)
    end_date = date.fromisoformat(args.end_date)

    if end_date < start_date:
        raise ValueError(
            "La date de fin doit être postérieure "
            "ou égale à la date de début."
        )

    feature = read_site_feature(
        parcels_file=args.parcels_file,
        site_id=site_id,
    )

    latitude, longitude = get_site_centroid(feature)

    print(f"Site : {site_id}")
    print(f"Latitude : {latitude}")
    print(f"Longitude : {longitude}")
    print(f"Période : {start_date} -> {end_date}")
    print(f"Modèle météo : {args.model}")
    print(f"Timezone : {args.timezone}")

    response_data = request_open_meteo_daily_weather(
        latitude=latitude,
        longitude=longitude,
        start_date=start_date.isoformat(),
        end_date=end_date.isoformat(),
        model=args.model,
        timezone=args.timezone,
    )

    weather_dataframe = build_weather_dataframe(
        response_data=response_data,
        site_id=site_id,
        latitude=latitude,
        longitude=longitude,
        model=args.model,
        timezone=args.timezone,
    )

    validate_weather_dataframe(weather_dataframe)

    weather_dataframe = add_rolling_weather_features(
        weather_dataframe
    )

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_file = (
        args.output_dir
        / f"weather_daily_{site_id}.csv"
    )

    temporary_output_file = output_file.with_suffix(
        f"{output_file.suffix}.tmp"
    )

    weather_dataframe.to_csv(
        temporary_output_file,
        index=False,
        encoding="utf-8",
    )

    temporary_output_file.replace(output_file)

    print()
    print(f"CSV météo créé : {output_file}")
    print(f"Lignes : {len(weather_dataframe)}")
    print(
        "Période CSV : "
        f"{weather_dataframe['date'].min().date()} -> "
        f"{weather_dataframe['date'].max().date()}"
    )


if __name__ == "__main__":
    main()
