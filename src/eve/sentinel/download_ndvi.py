from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone
from pathlib import Path

import requests
from dotenv import load_dotenv


TOKEN_URL = (
    "https://identity.dataspace.copernicus.eu/"
    "auth/realms/CDSE/protocol/openid-connect/token"
)

PROCESS_URL = "https://sh.dataspace.copernicus.eu/process/v1"

# download_ndvi.py -> sentinel -> eve -> src -> racine EVE
PROJECT_ROOT = Path(__file__).resolve().parents[3]

load_dotenv(PROJECT_ROOT / ".env")

OUTPUT_DIR = PROJECT_ROOT / "data" / "output"
OUTPUT_FILE = OUTPUT_DIR / "ndvi_test.tif"


# Zone de TEST issue de l'exemple Copernicus.
# Ordre : longitude min, latitude min, longitude max, latitude max.
# On la remplacera ensuite par ta propre parcelle.
BBOX = [
    13.822174072265625,
    45.85080395917834,
    14.55963134765625,
    46.29191774991382,
]


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


def get_access_token() -> str:
    """Récupère un token OAuth2 à partir du fichier .env."""

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
            f"Échec de l'authentification : "
            f"{response.status_code} - {response.text}"
        )

    response_data = response.json()

    if "access_token" not in response_data:
        raise RuntimeError("Le serveur n'a pas renvoyé de token.")

    return response_data["access_token"]


def format_datetime(value: datetime) -> str:
    """Convertit une date au format attendu par l'API."""

    return value.strftime("%Y-%m-%dT%H:%M:%SZ")


def download_ndvi() -> Path:
    """Télécharge un raster NDVI Sentinel-2 L2A."""

    end_date = datetime.now(timezone.utc)
    start_date = end_date - timedelta(days=90)

    token = get_access_token()

    request_body = {
        "input": {
            "bounds": {
                "bbox": BBOX,
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
            f"Échec de la requête Sentinel : "
            f"{response.status_code} - {response.text}"
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_FILE.write_bytes(response.content)

    return OUTPUT_FILE


if __name__ == "__main__":
    try:
        result = download_ndvi()
        print("Téléchargement terminé.")
        print(f"Fichier créé : {result}")
    except Exception as error:
        print(f"Erreur : {error}")
        raise