from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import requests
from rasterio.io import MemoryFile

from download_ndvi import (
    EVALSCRIPT,
    PROCESS_URL,
    get_access_token,
)


PROJECT_ROOT = Path(__file__).resolve().parents[3]

LOCATIONS_FILE = PROJECT_ROOT / "data" / "input" / "locations.csv"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
RASTER_DIR = PROJECT_ROOT / "data" / "output" / "ndvi"

OUTPUT_CSV = PROCESSED_DIR / "ndvi_summary.csv"


def format_datetime(value: datetime) -> str:
    return value.strftime("%Y-%m-%dT%H:%M:%SZ")


def build_bbox(row: pd.Series) -> list[float]:
    """Construit une BBOX Sentinel depuis une ligne du CSV."""

    return [
        float(row["min_longitude"]),
        float(row["min_latitude"]),
        float(row["max_longitude"]),
        float(row["max_latitude"]),
    ]


def request_ndvi(
    token: str,
    bbox: list[float],
    start_date: datetime,
    end_date: datetime,
) -> bytes:
    """Récupère le raster NDVI correspondant à une zone."""

    request_body = {
        "input": {
            "bounds": {
                "bbox": bbox,
                "properties": {
                    "crs": (
                        "http://www.opengis.net/"
                        "def/crs/OGC/1.3/CRS84"
                    )
                },
            },
            "data": [
                {
                    "type": "sentinel-2-l2a",
                    "dataFilter": {
                        "timeRange": {
                            "from": format_datetime(start_date),
                            "to": format_datetime(end_date),
                        },
                        "mosaickingOrder": "leastCC",
                        "maxCloudCoverage": 60,
                    },
                }
            ],
        },
        "output": {
            "width": 512,
            "height": 512,
            "responses": [
                {
                    "identifier": "default",
                    "format": {
                        "type": "image/tiff"
                    },
                }
            ],
        },
        "evalscript": EVALSCRIPT,
    }

    response = requests.post(
        PROCESS_URL,
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
            "Accept": "image/tiff",
        },
        json=request_body,
        timeout=120,
    )

    if not response.ok:
        raise RuntimeError(
            f"Erreur Sentinel : "
            f"{response.status_code} - {response.text}"
        )

    return response.content


def calculate_ndvi_statistics(
    raster_bytes: bytes,
) -> dict[str, Any]:
    """Calcule les statistiques du raster NDVI."""

    with MemoryFile(raster_bytes) as memory_file:
        with memory_file.open() as dataset:
            ndvi = dataset.read(1).astype(np.float32)

            total_pixel_count = ndvi.size

            valid_mask = (
                np.isfinite(ndvi)
                & (ndvi > -9990)
                & (ndvi >= -1)
                & (ndvi <= 1)
            )

            valid_values = ndvi[valid_mask]

            valid_pixel_count = valid_values.size

    if valid_pixel_count == 0:
        return {
            "pixel_count": total_pixel_count,
            "valid_pixel_count": 0,
            "valid_pixel_ratio": 0.0,
            "mean_ndvi": np.nan,
            "median_ndvi": np.nan,
            "std_ndvi": np.nan,
            "min_ndvi": np.nan,
            "max_ndvi": np.nan,
            "q25_ndvi": np.nan,
            "q75_ndvi": np.nan,
            "ratio_ndvi_above_0_3": np.nan,
            "ratio_ndvi_above_0_5": np.nan,
        }

    return {
        "pixel_count": total_pixel_count,
        "valid_pixel_count": valid_pixel_count,
        "valid_pixel_ratio": (
            valid_pixel_count / total_pixel_count
        ),
        "mean_ndvi": float(np.mean(valid_values)),
        "median_ndvi": float(np.median(valid_values)),
        "std_ndvi": float(np.std(valid_values)),
        "min_ndvi": float(np.min(valid_values)),
        "max_ndvi": float(np.max(valid_values)),
        "q25_ndvi": float(np.percentile(valid_values, 25)),
        "q75_ndvi": float(np.percentile(valid_values, 75)),
        "ratio_ndvi_above_0_3": float(
            np.mean(valid_values > 0.3)
        ),
        "ratio_ndvi_above_0_5": float(
            np.mean(valid_values > 0.5)
        ),
    }


def main() -> None:
    if not LOCATIONS_FILE.exists():
        raise FileNotFoundError(
            f"Fichier introuvable : {LOCATIONS_FILE}"
        )

    locations = pd.read_csv(LOCATIONS_FILE)

    required_columns = {
        "site_id",
        "name",
        "min_longitude",
        "min_latitude",
        "max_longitude",
        "max_latitude",
    }

    missing_columns = required_columns - set(locations.columns)

    if missing_columns:
        raise ValueError(
            f"Colonnes manquantes : {sorted(missing_columns)}"
        )

    end_date = datetime.now(timezone.utc)
    start_date = end_date - timedelta(days=90)

    token = get_access_token()

    results: list[dict[str, Any]] = []

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    RASTER_DIR.mkdir(parents=True, exist_ok=True)

    for _, location in locations.iterrows():
        site_id = str(location["site_id"])
        site_name = str(location["name"])

        print(f"Téléchargement : {site_id} - {site_name}")

        bbox = build_bbox(location)

        try:
            raster_bytes = request_ndvi(
                token=token,
                bbox=bbox,
                start_date=start_date,
                end_date=end_date,
            )

            raster_path = RASTER_DIR / f"{site_id}_ndvi.tif"
            raster_path.write_bytes(raster_bytes)

            statistics = calculate_ndvi_statistics(
                raster_bytes
            )

            result = {
                "site_id": site_id,
                "name": site_name,
                "crop_type": location.get(
                    "crop_type",
                    None,
                ),
                "date_from": start_date.date().isoformat(),
                "date_to": end_date.date().isoformat(),
                "min_longitude": bbox[0],
                "min_latitude": bbox[1],
                "max_longitude": bbox[2],
                "max_latitude": bbox[3],
                "status": "success",
                **statistics,
            }

        except Exception as error:
            print(f"Erreur pour {site_id} : {error}")

            result = {
                "site_id": site_id,
                "name": site_name,
                "crop_type": location.get(
                    "crop_type",
                    None,
                ),
                "date_from": start_date.date().isoformat(),
                "date_to": end_date.date().isoformat(),
                "min_longitude": bbox[0],
                "min_latitude": bbox[1],
                "max_longitude": bbox[2],
                "max_latitude": bbox[3],
                "status": "error",
                "error": str(error),
            }

        results.append(result)

    ndvi_dataframe = pd.DataFrame(results)

    ndvi_dataframe.to_csv(
        OUTPUT_CSV,
        index=False,
        encoding="utf-8",
    )

    print()
    print("DataFrame NDVI :")

    columns_to_display = [
        "site_id",
        "name",
        "mean_ndvi",
        "median_ndvi",
        "std_ndvi",
        "valid_pixel_ratio",
        "status",
    ]

    existing_columns = [
        column
        for column in columns_to_display
        if column in ndvi_dataframe.columns
    ]

    print(
        ndvi_dataframe[
            existing_columns
        ].to_string(index=False)
    )

    print()
    print(f"CSV créé : {OUTPUT_CSV}")


if __name__ == "__main__":
    main()