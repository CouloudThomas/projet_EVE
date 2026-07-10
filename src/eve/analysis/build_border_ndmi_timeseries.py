from __future__ import annotations

import argparse
import os
from datetime import date
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from shapely.geometry import mapping, shape

from eve.sentinel.build_ndmi_timeseries import (
    get_access_token,
    load_parcel,
    parse_statistics,
    project_geometry_to_local_utm,
    request_ndmi_statistics,
)


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
ENV_FILE = PROJECT_ROOT / ".env"


def parse_date(value: str) -> date:
    try:
        return date.fromisoformat(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            f"Date invalide {value!r}. Format attendu : YYYY-MM-DD."
        ) from error


def parse_arguments() -> argparse.Namespace:
    load_dotenv(ENV_FILE)

    parser = argparse.ArgumentParser(
        description=(
            "Calcule une série NDMI avec retraits de bordure pour une parcelle. "
            "La sortie sert à alimenter la V0.2 officielle d'EVE."
        )
    )

    parser.add_argument(
        "--site",
        default=os.getenv("EVE_SITE_ID", "site_004"),
        help="Identifiant du site à traiter.",
    )
    parser.add_argument(
        "--start-date",
        type=parse_date,
        default=parse_date(os.getenv("EVE_START_DATE", "2025-01-01")),
        help="Date de début incluse, au format YYYY-MM-DD.",
    )
    parser.add_argument(
        "--end-date",
        type=parse_date,
        default=parse_date(os.getenv("EVE_END_DATE", date.today().isoformat())),
        help="Date de fin incluse, au format YYYY-MM-DD.",
    )
    parser.add_argument(
        "--buffers",
        type=int,
        nargs="+",
        default=[0, 10, 20],
        help=(
            "Retraits de bordure à calculer en mètres. "
            "0 = parcelle entière, 10 = parcelle sans les 10 premiers mètres."
        ),
    )
    parser.add_argument(
        "--min-valid-pixel-ratio",
        type=float,
        default=float(os.getenv("EVE_MIN_VALID_PIXEL_RATIO", "0.70")),
        help="Seuil minimal de pixels valides pour considérer une date exploitable.",
    )
    parser.add_argument(
        "--processed-dir",
        type=Path,
        default=DEFAULT_PROCESSED_DIR,
        help="Dossier de sortie des CSV traités.",
    )

    args = parser.parse_args()

    if args.start_date > args.end_date:
        parser.error("--start-date doit être antérieure ou égale à --end-date.")

    if any(buffer_m < 0 for buffer_m in args.buffers):
        parser.error("--buffers doit contenir uniquement des valeurs positives ou nulles.")

    return args


def standardize_statistics(
    dataframe: pd.DataFrame,
    site_id: str,
    buffer_m: int,
    area_m2: float,
    area_retained_ratio: float,
    min_valid_pixel_ratio: float,
) -> pd.DataFrame:
    if dataframe.empty:
        return pd.DataFrame()

    result = dataframe.copy()
    result["site_id"] = site_id
    result["index"] = "ndmi"
    result["buffer_m"] = buffer_m
    result["area_m2"] = area_m2
    result["area_retained_ratio"] = area_retained_ratio

    rename_mapping = {
        "ndmi_mean": "mean",
        "ndmi_median": "median",
        "ndmi_std": "std",
        "ndmi_min": "minimum",
        "ndmi_max": "maximum",
        "ndmi_p10": "p10",
        "ndmi_p25": "p25",
        "ndmi_p75": "p75",
        "ndmi_p90": "p90",
    }

    result = result.rename(columns=rename_mapping)

    result["valid_pixel_ratio"] = pd.to_numeric(
        result.get("valid_pixel_ratio"),
        errors="coerce",
    )
    result["mean"] = pd.to_numeric(
        result.get("mean"),
        errors="coerce",
    )
    result["is_usable"] = (
        result["valid_pixel_ratio"].ge(min_valid_pixel_ratio)
        & result["mean"].notna()
    )

    ordered_columns = [
        "site_id",
        "acquisition_date",
        "index",
        "buffer_m",
        "area_m2",
        "area_retained_ratio",
        "geometry_pixel_count",
        "sample_count",
        "no_data_count",
        "valid_pixel_count",
        "valid_pixel_ratio",
        "mean",
        "median",
        "std",
        "minimum",
        "maximum",
        "p10",
        "p25",
        "p75",
        "p90",
        "is_usable",
    ]

    for column in ordered_columns:
        if column not in result.columns:
            result[column] = pd.NA

    return result[ordered_columns]


def main() -> None:
    args = parse_arguments()

    geometry, properties = load_parcel(args.site)
    parcel_name = properties.get("name", args.site)

    projected_geometry_mapping, crs_url, epsg_code = project_geometry_to_local_utm(
        geometry
    )
    projected_geometry = shape(projected_geometry_mapping)
    original_area = projected_geometry.area

    print(f"Parcelle : {args.site} - {parcel_name}")
    print(f"Projection locale : EPSG:{epsg_code}")
    print(f"Période : {args.start_date} -> {args.end_date}")
    print(f"Retraits testés : {args.buffers} m")

    token = get_access_token()
    frames: list[pd.DataFrame] = []

    for buffer_m in args.buffers:
        if buffer_m == 0:
            analysis_geometry = projected_geometry
        else:
            analysis_geometry = projected_geometry.buffer(-buffer_m)

        if analysis_geometry.is_empty:
            print(
                f"Buffer {buffer_m} m ignoré : la géométrie restante est vide."
            )
            continue

        area_m2 = float(analysis_geometry.area)
        area_retained_ratio = (
            area_m2 / original_area
            if original_area
            else float("nan")
        )

        print(
            f"Calcul NDMI buffer {buffer_m} m "
            f"(surface conservée : {area_retained_ratio:.3f})..."
        )

        statistics_response = request_ndmi_statistics(
            token=token,
            geometry=mapping(analysis_geometry),
            geometry_crs_url=crs_url,
            start_date=args.start_date,
            end_date=args.end_date,
        )
        statistics = parse_statistics(statistics_response)
        standardized = standardize_statistics(
            dataframe=statistics,
            site_id=args.site,
            buffer_m=buffer_m,
            area_m2=area_m2,
            area_retained_ratio=area_retained_ratio,
            min_valid_pixel_ratio=args.min_valid_pixel_ratio,
        )
        frames.append(standardized)

    if not frames:
        raise RuntimeError("Aucune statistique NDMI de bordure n'a été produite.")

    output = (
        pd.concat(frames, ignore_index=True)
        .sort_values(["acquisition_date", "buffer_m"])
        .reset_index(drop=True)
    )

    args.processed_dir.mkdir(parents=True, exist_ok=True)
    output_file = args.processed_dir / f"border_ndmi_timeseries_{args.site}.csv"
    output.to_csv(output_file, index=False, encoding="utf-8")

    print()
    print("Traitement terminé.")
    print(f"Lignes : {len(output)}")
    print(f"Fichier : {output_file}")
    print(
        output[
            [
                "acquisition_date",
                "buffer_m",
                "mean",
                "valid_pixel_ratio",
                "is_usable",
            ]
        ]
        .tail(15)
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()
