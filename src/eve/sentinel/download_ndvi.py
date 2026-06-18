from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests
from dotenv import load_dotenv


# ------------------------------------------------------------------
# CONFIGURATION
# ------------------------------------------------------------------

TOKEN_URL = (
    "https://identity.dataspace.copernicus.eu/"
    "auth/realms/CDSE/protocol/openid-connect/token"
)

PROCESS_URL = "https://sh.dataspace.copernicus.eu/process/v1"

# download_ndvi.py -> sentinel -> eve -> src -> racine EVE
PROJECT_ROOT = Path(__file__).resolve().parents[3]

ENV_FILE = PROJECT_ROOT / ".env"
PARCELS_FILE = PROJECT_ROOT / "data" / "input" / "parcels.geojson"
OUTPUT_DIR = PROJECT_ROOT / "data" / "output"

# Parcelle à récupérer dans parcels.geojson
SELECTED_SITE_ID = "site_001"

START_DATE = datetime(
    2026,
    5,
    1,
    tzinfo=timezone.utc,
)

END_DATE = datetime(
    2026,
    6,
    1,
    tzinfo=timezone.utc,
)

load_dotenv(ENV_FILE)


# ------------------------------------------------------------------
# SCRIPT SENTINEL : CALCUL DU NDVI
# ------------------------------------------------------------------

EVALSCRIPT = """
//VERSION=3

function setup() {
    return {
        input: ["B04", "B08", "SCL", "dataMask"],
        output: {
            bands: 1,
            sampleType: "FLOAT32"
        }
    };
}

function evaluatePixel(sample) {
    const invalidClasses = [0, 1, 3, 8, 9, 10, 11];

    if (
        sample.dataMask === 0 ||
        invalidClasses.includes(sample.SCL)
    ) {
        return [-9999];
    }

    const denominator = sample.B08 + sample.B04;

    if (denominator === 0) {
        return [-9999];
    }

    const ndvi = (sample.B08 - sample.B04) / denominator;

    return [ndvi];
}
"""


# ------------------------------------------------------------------
# AUTHENTIFICATION
# ------------------------------------------------------------------

def get_access_token() -> str:
    """Récupère un token Copernicus avec les identifiants du fichier .env."""

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

    response_data = response.json()
    access_token = response_data.get("access_token")

    if not access_token:
        raise RuntimeError(
            "Copernicus n'a pas renvoyé de token d'accès."
        )

    return access_token


# ------------------------------------------------------------------
# LECTURE DU GEOJSON
# ------------------------------------------------------------------

def load_parcel(
    site_id: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """
    Cherche une parcelle dans parcels.geojson.

    Retourne :
    - sa géométrie ;
    - ses propriétés.
    """

    if not PARCELS_FILE.exists():
        raise FileNotFoundError(
            f"Fichier GeoJSON introuvable : {PARCELS_FILE}"
        )

    with PARCELS_FILE.open(
        mode="r",
        encoding="utf-8",
    ) as file:
        geojson_data = json.load(file)

    if geojson_data.get("type") != "FeatureCollection":
        raise ValueError(
            "parcels.geojson doit être une FeatureCollection."
        )

    features = geojson_data.get("features", [])

    if not features:
        raise ValueError(
            "parcels.geojson ne contient aucune parcelle."
        )

    for feature in features:
        properties = feature.get("properties", {})

        if properties.get("site_id") != site_id:
            continue

        geometry = feature.get("geometry")

        if geometry is None:
            raise ValueError(
                f"La parcelle {site_id!r} n'a pas de géométrie."
            )

        geometry_type = geometry.get("type")

        if geometry_type not in {"Polygon", "MultiPolygon"}:
            raise ValueError(
                f"La parcelle {site_id!r} doit être un Polygon "
                f"ou MultiPolygon, pas {geometry_type!r}."
            )

        if not geometry.get("coordinates"):
            raise ValueError(
                f"La parcelle {site_id!r} ne contient "
                "aucune coordonnée."
            )

        return geometry, properties

    raise ValueError(
        f"Aucune parcelle avec site_id={site_id!r} "
        "dans parcels.geojson."
    )


# ------------------------------------------------------------------
# OUTILS
# ------------------------------------------------------------------

def format_datetime(value: datetime) -> str:
    """Convertit une date au format attendu par Sentinel Hub."""

    return value.strftime("%Y-%m-%dT%H:%M:%SZ")


# ------------------------------------------------------------------
# TÉLÉCHARGEMENT DU NDVI
# ------------------------------------------------------------------

def download_ndvi() -> Path:
    """Télécharge le NDVI de la parcelle sélectionnée."""

    geometry, properties = load_parcel(
        SELECTED_SITE_ID
    )

    parcel_name = properties.get(
        "name",
        SELECTED_SITE_ID,
    )

    start_date = START_DATE
    end_date = END_DATE

    print(
        f"Parcelle sélectionnée : "
        f"{SELECTED_SITE_ID} - {parcel_name}"
    )

    print(
        f"Période : "
        f"{start_date.date()} → {end_date.date()}"
    )

    token = get_access_token()

    request_body = {
        "input": {
            "bounds": {
                "geometry": geometry,
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
                        "maxCloudCoverage": 80,
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
            "Échec de la requête Sentinel : "
            f"{response.status_code} - {response.text}"
        )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_file = (
        OUTPUT_DIR
        / f"{SELECTED_SITE_ID}_ndvi.tif"
    )

    output_file.write_bytes(response.content)

    return output_file


# ------------------------------------------------------------------
# EXÉCUTION
# ------------------------------------------------------------------

if __name__ == "__main__":
    try:
        result = download_ndvi()

        print("Téléchargement terminé.")
        print(f"Fichier créé : {result}")

    except Exception as error:
        print(f"Erreur : {error}")
        raise