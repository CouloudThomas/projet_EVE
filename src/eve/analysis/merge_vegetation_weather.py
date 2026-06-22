from __future__ import annotations

import argparse
import os
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parents[3]

DEFAULT_PROCESSED_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
)


def load_environment() -> None:
    env_file = PROJECT_ROOT / ".env"

    if env_file.exists():
        load_dotenv(env_file)


def parse_arguments() -> argparse.Namespace:
    load_environment()

    parser = argparse.ArgumentParser(
        description=(
            "Fusionne les indices Sentinel avec les données "
            "météo quotidiennes et leurs variables dérivées."
        )
    )

    parser.add_argument(
        "--site",
        default=os.getenv("EVE_SITE_ID", "site_004"),
        help="Identifiant du site.",
    )

    parser.add_argument(
        "--processed-dir",
        type=Path,
        default=DEFAULT_PROCESSED_DIR,
        help="Dossier contenant les CSV traités.",
    )

    parser.add_argument(
        "--use-border-ndmi",
        action="store_true",
        help=(
            "Ajoute la série NDMI calculée avec buffer "
            "depuis border_ndmi_timeseries_SITE.csv."
        ),
    )

    parser.add_argument(
        "--buffer-m",
        type=int,
        default=10,
        help="Buffer NDMI à utiliser si --use-border-ndmi est activé.",
    )

    parser.add_argument(
        "--replace-ndmi",
        action="store_true",
        help=(
            "Remplace ndmi_mean d'analyse par la série NDMI bufferisée. "
            "La colonne originale ndmi_mean est conservée."
        ),
    )

    return parser.parse_args()


def read_vegetation_timeseries(
    processed_dir: Path,
    site_id: str,
) -> pd.DataFrame:
    vegetation_file = (
        processed_dir
        / f"vegetation_timeseries_{site_id}.csv"
    )

    if not vegetation_file.exists():
        raise FileNotFoundError(
            f"CSV végétation introuvable : {vegetation_file}"
        )

    vegetation = pd.read_csv(
        vegetation_file,
        parse_dates=["acquisition_date"],
    )

    vegetation["date"] = vegetation[
        "acquisition_date"
    ].dt.normalize()

    return vegetation


def read_weather_timeseries(
    processed_dir: Path,
    site_id: str,
) -> pd.DataFrame:
    weather_file = (
        processed_dir
        / f"weather_daily_{site_id}.csv"
    )

    if not weather_file.exists():
        raise FileNotFoundError(
            f"CSV météo introuvable : {weather_file}"
        )

    weather = pd.read_csv(
        weather_file,
        parse_dates=["date"],
    )

    weather["date"] = weather["date"].dt.normalize()

    return weather


def add_border_ndmi(
    merged: pd.DataFrame,
    processed_dir: Path,
    site_id: str,
    buffer_m: int,
    replace_ndmi: bool,
) -> pd.DataFrame:
    border_file = (
        processed_dir
        / f"border_ndmi_timeseries_{site_id}.csv"
    )

    if not border_file.exists():
        raise FileNotFoundError(
            f"CSV NDMI buffer introuvable : {border_file}"
        )

    border_ndmi = pd.read_csv(
        border_file,
        parse_dates=["acquisition_date"],
    )

    numeric_columns = [
        "buffer_m",
        "mean",
        "median",
        "std",
        "minimum",
        "maximum",
        "p10",
        "p25",
        "p75",
        "p90",
        "valid_pixel_count",
        "valid_pixel_ratio",
        "is_usable",
    ]

    for column in numeric_columns:
        if column in border_ndmi.columns:
            border_ndmi[column] = pd.to_numeric(
                border_ndmi[column],
                errors="coerce",
            )

    border_ndmi = border_ndmi[
        border_ndmi["buffer_m"] == buffer_m
    ].copy()

    border_ndmi["date"] = border_ndmi[
        "acquisition_date"
    ].dt.normalize()

    rename_mapping = {
        "mean": f"ndmi_buffer_{buffer_m}m_mean",
        "median": f"ndmi_buffer_{buffer_m}m_median",
        "std": f"ndmi_buffer_{buffer_m}m_std",
        "minimum": f"ndmi_buffer_{buffer_m}m_minimum",
        "maximum": f"ndmi_buffer_{buffer_m}m_maximum",
        "p10": f"ndmi_buffer_{buffer_m}m_p10",
        "p25": f"ndmi_buffer_{buffer_m}m_p25",
        "p75": f"ndmi_buffer_{buffer_m}m_p75",
        "p90": f"ndmi_buffer_{buffer_m}m_p90",
        "valid_pixel_count": (
            f"ndmi_buffer_{buffer_m}m_valid_pixel_count"
        ),
        "valid_pixel_ratio": (
            f"ndmi_buffer_{buffer_m}m_valid_pixel_ratio"
        ),
        "is_usable": f"ndmi_buffer_{buffer_m}m_is_usable",
    }

    selected_columns = [
        "date",
        *rename_mapping.keys(),
    ]

    border_ndmi = border_ndmi[
        selected_columns
    ].rename(
        columns=rename_mapping
    )

    merged = merged.merge(
        border_ndmi,
        on="date",
        how="left",
    )

    analysis_column = f"ndmi_buffer_{buffer_m}m_mean"

    if replace_ndmi:
        merged["ndmi_analysis_mean"] = merged[
            analysis_column
        ].combine_first(
            merged["ndmi_mean"]
        )
    else:
        merged["ndmi_analysis_mean"] = merged[
            "ndmi_mean"
        ]

    return merged


def main() -> None:
    args = parse_arguments()

    site_id = args.site
    processed_dir = args.processed_dir

    vegetation = read_vegetation_timeseries(
        processed_dir=processed_dir,
        site_id=site_id,
    )

    weather = read_weather_timeseries(
        processed_dir=processed_dir,
        site_id=site_id,
    )

    merged = vegetation.merge(
        weather,
        on="date",
        how="left",
        suffixes=("", "_weather"),
    )

    if args.use_border_ndmi:
        merged = add_border_ndmi(
            merged=merged,
            processed_dir=processed_dir,
            site_id=site_id,
            buffer_m=args.buffer_m,
            replace_ndmi=args.replace_ndmi,
        )
    else:
        merged["ndmi_analysis_mean"] = merged[
            "ndmi_mean"
        ]

    output_file = (
        processed_dir
        / f"vegetation_weather_{site_id}.csv"
    )

    merged.to_csv(
        output_file,
        index=False,
        encoding="utf-8",
    )

    print(f"CSV fusionné créé : {output_file}")
    print(f"Lignes : {len(merged)}")
    print(f"Colonnes : {len(merged.columns)}")
    print()
    print("Colonnes météo ajoutées :")

    weather_columns = [
        column
        for column in weather.columns
        if column != "date"
    ]

    for column in weather_columns:
        print(f"- {column}")

    if args.use_border_ndmi:
        print()
        print(
            f"NDMI buffer {args.buffer_m} m ajouté."
        )

        if args.replace_ndmi:
            print(
                "ndmi_analysis_mean utilise le NDMI bufferisé."
            )
        else:
            print(
                "ndmi_analysis_mean utilise le NDMI original."
            )


if __name__ == "__main__":
    main()