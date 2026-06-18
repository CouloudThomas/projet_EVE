from __future__ import annotations

import json
import os
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import requests
from dotenv import load_dotenv
from pyproj import Transformer
from shapely.geometry import mapping, shape
from shapely.ops import transform as shapely_transform


TOKEN_URL = (
    "https://identity.dataspace.copernicus.eu/"
    "auth/realms/CDSE/protocol/openid-connect/token"
)
PROCESS_URL = "https://sh.dataspace.copernicus.eu/process/v1"

PROJECT_ROOT = Path(__file__).resolve().parents[3]
ENV_FILE = PROJECT_ROOT / ".env"
PARCELS_FILE = PROJECT_ROOT / "data" / "input" / "parcels.geojson"
RASTERS_OUTPUT_DIR = PROJECT_ROOT / "data" / "output" / "rasters"

load_dotenv(ENV_FILE)


def get_access_token() -> str:
    client_id = os.getenv("COPERNICUS_CLIENT_ID")
    client_secret = os.getenv("COPERNICUS_CLIENT_SECRET")

    if not client_id or not client_secret:
        raise RuntimeError(
            "COPERNICUS_CLIENT_ID ou COPERNICUS_CLIENT_SECRET "
            "est absent du fichier .env."
        )

    response = requests.post(
        TOKEN_URL,
        data={
            "grant_type": "client_credentials",
            "client_id": client_id,
            "client_secret": client_secret,
        },
        timeout=30,
    )

    if not response.ok:
        raise RuntimeError(
            "Échec de l'authentification Copernicus : "
            f"{response.status_code} - {response.text}"
        )

    token = response.json().get("access_token")

    if not token:
        raise RuntimeError(
            "Copernicus n'a pas renvoyé de token d'accès."
        )

    return token


def load_parcel(
    site_id: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if not PARCELS_FILE.exists():
        raise FileNotFoundError(
            f"Fichier GeoJSON introuvable : {PARCELS_FILE}"
        )

    with PARCELS_FILE.open("r", encoding="utf-8") as file:
        geojson = json.load(file)

    if geojson.get("type") != "FeatureCollection":
        raise ValueError(
            "parcels.geojson doit être une FeatureCollection."
        )

    for feature in geojson.get("features", []):
        properties = feature.get("properties", {})

        if properties.get("site_id") != site_id:
            continue

        geometry = feature.get("geometry")

        if not geometry:
            raise ValueError(
                f"La parcelle {site_id!r} n'a pas de géométrie."
            )

        if geometry.get("type") not in {"Polygon", "MultiPolygon"}:
            raise ValueError(
                "La géométrie doit être un Polygon ou un MultiPolygon."
            )

        if not geometry.get("coordinates"):
            raise ValueError(
                f"La parcelle {site_id!r} ne contient aucune coordonnée."
            )

        return geometry, properties

    raise ValueError(
        f"Aucune parcelle avec site_id={site_id!r} "
        "dans parcels.geojson."
    )


def project_geometry_to_local_utm(
    geometry: dict[str, Any],
) -> tuple[dict[str, Any], str, int]:
    wgs84_geometry = shape(geometry)
    centroid = wgs84_geometry.centroid

    longitude = centroid.x
    latitude = centroid.y
    utm_zone = int((longitude + 180) // 6) + 1

    epsg_code = (
        32600 + utm_zone
        if latitude >= 0
        else 32700 + utm_zone
    )

    transformer = Transformer.from_crs(
        "EPSG:4326",
        f"EPSG:{epsg_code}",
        always_xy=True,
    )

    projected_geometry = shapely_transform(
        transformer.transform,
        wgs84_geometry,
    )

    crs_url = (
        "http://www.opengis.net/"
        f"def/crs/EPSG/0/{epsg_code}"
    )

    return mapping(projected_geometry), crs_url, epsg_code


def parse_acquisition_date(value: str) -> date:
    try:
        return date.fromisoformat(value)
    except ValueError as error:
        raise ValueError(
            "La date doit être au format YYYY-MM-DD."
        ) from error


def date_to_utc_string(value: date) -> str:
    value_as_datetime = datetime(
        value.year,
        value.month,
        value.day,
        tzinfo=timezone.utc,
    )
    return value_as_datetime.strftime("%Y-%m-%dT%H:%M:%SZ")


def download_index_for_date(
    *,
    site_id: str,
    acquisition_date: str,
    index_name: str,
    resolution_m: int,
    evalscript: str,
    max_cloud_coverage: float = 100,
) -> Path:
    selected_date = parse_acquisition_date(acquisition_date)
    end_exclusive = selected_date + timedelta(days=1)

    geometry, properties = load_parcel(site_id)
    projected_geometry, crs_url, epsg_code = (
        project_geometry_to_local_utm(geometry)
    )

    parcel_name = properties.get("name", site_id)
    normalized_index = index_name.lower()

    print(f"Indice : {index_name.upper()}")
    print(f"Parcelle : {site_id} - {parcel_name}")
    print(f"Date : {selected_date}")
    print(f"Résolution : {resolution_m} m")
    print(f"Projection : EPSG:{epsg_code}")

    token = get_access_token()

    request_body = {
        "input": {
            "bounds": {
                "geometry": projected_geometry,
                "properties": {
                    "crs": crs_url
                },
            },
            "data": [
                {
                    "type": "sentinel-2-l2a",
                    "dataFilter": {
                        "timeRange": {
                            "from": date_to_utc_string(selected_date),
                            "to": date_to_utc_string(end_exclusive),
                        },
                        "mosaickingOrder": "leastCC",
                        "maxCloudCoverage": max_cloud_coverage,
                    },
                }
            ],
        },
        "output": {
            "resx": resolution_m,
            "resy": resolution_m,
            "responses": [
                {
                    "identifier": "default",
                    "format": {
                        "type": "image/tiff"
                    },
                }
            ],
        },
        "evalscript": evalscript,
    }

    response = requests.post(
        PROCESS_URL,
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
            "Accept": "image/tiff",
        },
        json=request_body,
        timeout=180,
    )

    if not response.ok:
        raise RuntimeError(
            f"Échec de la Process API pour {index_name.upper()} : "
            f"{response.status_code} - {response.text}"
        )

    output_dir = RASTERS_OUTPUT_DIR / site_id / normalized_index
    output_dir.mkdir(parents=True, exist_ok=True)

    output_file = (
        output_dir
        / f"{acquisition_date}_{normalized_index}.tif"
    )
    output_file.write_bytes(response.content)

    print(f"GeoTIFF créé : {output_file}")

    return output_file
