from __future__ import annotations

from pyproj import Transformer
from shapely.geometry import mapping, shape
from shapely.ops import transform as shapely_transform

import json
import os
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd
import requests
from dotenv import load_dotenv


# ==============================================================
# CONFIGURATION : MODIFIE PRINCIPALEMENT CES TROIS VALEURS
# ==============================================================

SELECTED_SITE_ID = "site_001"

# Dates incluses dans l'analyse.
START_DATE = "2025-01-01"
END_DATE = date.today().isoformat()

# Résolution Sentinel-2 en mètres.
RESOLUTION_METERS = 20

# Seuil minimal de pixels valides pour considérer une date exploitable.
# C'est un seuil pratique pour la V0, pas une vérité agronomique.
MIN_VALID_PIXEL_RATIO = 0.30

SAVE_RAW_RESPONSES = False


# ==============================================================
# CHEMINS ET URL
# ==============================================================

TOKEN_URL = (
    "https://identity.dataspace.copernicus.eu/"
    "auth/realms/CDSE/protocol/openid-connect/token"
)

CATALOG_URL = (
    "https://sh.dataspace.copernicus.eu/"
    "catalog/v1/search"
)

STATISTICS_URL = (
    "https://sh.dataspace.copernicus.eu/"
    "statistics/v1"
)

# build_ndre_timeseries.py -> sentinel -> eve -> src -> EVE
PROJECT_ROOT = Path(__file__).resolve().parents[3]

ENV_FILE = PROJECT_ROOT / ".env"

PARCELS_FILE = (
    PROJECT_ROOT
    / "data"
    / "input"
    / "parcels.geojson"
)

PROCESSED_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "output"
)

load_dotenv(ENV_FILE)


# ==============================================================
# EVALSCRIPT NDRE
# ==============================================================

EVALSCRIPT = """
//VERSION=3

function setup() {
    return {
        input: [{
            bands: [
                "B05",
                "B8A",
                "SCL",
                "dataMask"
            ]
        }],
        output: [
            {
                id: "ndre",
                bands: 1,
                sampleType: "FLOAT32"
            },
            {
                id: "dataMask",
                bands: 1
            }
        ]
    };
}

function evaluatePixel(sample) {
    const invalidClasses = [0, 1, 3, 8, 9, 10, 11];
    const denominator = sample.B8A + sample.B05;

    const validPixel =
        sample.dataMask === 1
        && denominator !== 0
        && !invalidClasses.includes(sample.SCL);

    const ndre = validPixel
        ? (
            sample.B8A - sample.B05
        ) / denominator
        : 0;

    return {
        ndre: [ndre],
        dataMask: [validPixel ? 1 : 0]
    };
}
"""


# ==============================================================
# AUTHENTIFICATION
# ==============================================================

def get_access_token() -> str:
    """Récupère un token OAuth2 Copernicus."""

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
            "Copernicus n'a pas renvoyé de token."
        )

    return token


def get_headers(
    token: str,
    accept: str = "application/json",
) -> dict[str, str]:
    """Construit les en-têtes HTTP authentifiés."""

    return {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
        "Accept": accept,
    }

# ==============================================================
# LECTURE DE LA PARCELLE
# ==============================================================

def load_parcel(
    site_id: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Charge une parcelle depuis parcels.geojson."""

    if not PARCELS_FILE.exists():
        raise FileNotFoundError(
            f"Fichier introuvable : {PARCELS_FILE}"
        )

    with PARCELS_FILE.open(
        mode="r",
        encoding="utf-8",
    ) as file:
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

        if geometry.get("type") not in {
            "Polygon",
            "MultiPolygon",
        }:
            raise ValueError(
                "La géométrie doit être un Polygon "
                "ou un MultiPolygon."
            )

        return geometry, properties

    raise ValueError(
        f"Parcelle inconnue : {site_id!r}"
    )


# ==============================================================
# GESTION DES DATES
# ==============================================================

def parse_dates() -> tuple[date, date]:
    """Valide les dates configurées."""

    try:
        start_date = date.fromisoformat(START_DATE)
        end_date = date.fromisoformat(END_DATE)
    except ValueError as error:
        raise ValueError(
            "Les dates doivent utiliser le format YYYY-MM-DD."
        ) from error

    if start_date > end_date:
        raise ValueError(
            "START_DATE doit être antérieure à END_DATE."
        )

    return start_date, end_date


def date_to_utc_string(value: date) -> str:
    """Transforme une date en date-heure UTC."""

    date_time = datetime(
        value.year,
        value.month,
        value.day,
        tzinfo=timezone.utc,
    )

    return date_time.strftime("%Y-%m-%dT%H:%M:%SZ")


# ==============================================================
# CATALOG API : RECHERCHE DES ACQUISITIONS
# ==============================================================

def search_catalog(
    token: str,
    geometry: dict[str, Any],
    start_date: date,
    end_date: date,
) -> list[dict[str, Any]]:
    """Recherche toutes les acquisitions Sentinel-2 disponibles."""

    # La borne "to" est exclusive :
    # on ajoute un jour pour inclure entièrement END_DATE.
    end_exclusive = end_date + timedelta(days=1)

    request_body: dict[str, Any] = {
        "datetime": (
            f"{date_to_utc_string(start_date)}/"
            f"{date_to_utc_string(end_exclusive)}"
        ),
        "collections": ["sentinel-2-l2a"],
        "intersects": geometry,
        "limit": 100,
    }

    features: list[dict[str, Any]] = []

    while True:
        response = requests.post(
    CATALOG_URL,
    headers=get_headers(
        token,
        accept="application/geo+json",
    ),
    json=request_body,
    timeout=60,
)

        if not response.ok:
            raise RuntimeError(
                "Échec de la recherche dans le catalogue : "
                f"{response.status_code} - {response.text}"
            )

        response_data = response.json()

        features.extend(
            response_data.get("features", [])
        )

        next_token = (
            response_data
            .get("context", {})
            .get("next")
        )

        if next_token is None:
            break

        request_body["next"] = next_token

    return features


def build_catalog_dataframe(
    features: list[dict[str, Any]],
) -> pd.DataFrame:
    """Transforme les acquisitions du catalogue en DataFrame."""

    rows: list[dict[str, Any]] = []

    for feature in features:
        properties = feature.get("properties", {})
        acquisition_datetime = properties.get("datetime")

        if not acquisition_datetime:
            continue

        rows.append(
            {
                "product_id": feature.get("id"),
                "acquisition_datetime": acquisition_datetime,
                "tile_cloud_cover": properties.get(
                    "eo:cloud_cover"
                ),
            }
        )

    if not rows:
        return pd.DataFrame()

    catalog = pd.DataFrame(rows)

    catalog["acquisition_datetime"] = pd.to_datetime(
        catalog["acquisition_datetime"],
        utc=True,
        errors="coerce",
    )

    catalog = catalog.dropna(
        subset=["acquisition_datetime"]
    )

    catalog["acquisition_date"] = (
        catalog["acquisition_datetime"]
        .dt.strftime("%Y-%m-%d")
    )

    catalog["tile_cloud_cover"] = pd.to_numeric(
        catalog["tile_cloud_cover"],
        errors="coerce",
    )

    def join_unique(values: pd.Series) -> str:
        valid_values = {
            str(value)
            for value in values
            if pd.notna(value)
        }

        return " | ".join(sorted(valid_values))

    def join_datetimes(values: pd.Series) -> str:
        formatted = {
            value.isoformat()
            for value in values
            if pd.notna(value)
        }

        return " | ".join(sorted(formatted))

    daily_catalog = (
        catalog
        .groupby(
            "acquisition_date",
            as_index=False,
        )
        .agg(
            acquisition_count=(
                "product_id",
                "nunique",
            ),
            product_ids=(
                "product_id",
                join_unique,
            ),
            acquisition_datetimes=(
                "acquisition_datetime",
                join_datetimes,
            ),
            tile_cloud_cover_min=(
                "tile_cloud_cover",
                "min",
            ),
            tile_cloud_cover_mean=(
                "tile_cloud_cover",
                "mean",
            ),
            tile_cloud_cover_max=(
                "tile_cloud_cover",
                "max",
            ),
        )
    )

    return daily_catalog


# ==============================================================
# STATISTICAL API : STATISTIQUES NDRE JOURNALIÈRES
# ==============================================================

def project_geometry_to_local_utm(
    geometry: dict[str, Any],
) -> tuple[dict[str, Any], str, int]:
    """
    Projette automatiquement une géométrie WGS84
    dans la zone UTM locale correspondante.
    """

    wgs84_geometry = shape(geometry)

    centroid = wgs84_geometry.centroid
    longitude = centroid.x
    latitude = centroid.y

    utm_zone = int((longitude + 180) // 6) + 1

    if latitude >= 0:
        epsg_code = 32600 + utm_zone
    else:
        epsg_code = 32700 + utm_zone

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

    return (
        mapping(projected_geometry),
        crs_url,
        epsg_code,
    )

def request_ndre_statistics(
    token: str,
    geometry: dict[str, Any],
    geometry_crs_url: str,
    start_date: date,
    end_date: date,
) -> dict[str, Any]:
    """Récupère les statistiques NDRE journalières."""

    end_exclusive = end_date + timedelta(days=1)

    request_body = {
        "input": {
            "bounds": {
                "geometry": geometry,
                "properties": {
                    "crs": geometry_crs_url
                },
            },
            "data": [
                {
                    "type": "sentinel-2-l2a",
                    "dataFilter": {
                        "mosaickingOrder": "leastCC",

                        # On ne supprime pas des acquisitions
                        # uniquement à partir du nuage de la tuile.
                        # Le masque SCL travaille sur la parcelle.
                        "maxCloudCoverage": 100,
                    },
                }
            ],
        },
        "aggregation": {
            "timeRange": {
                "from": date_to_utc_string(start_date),
                "to": date_to_utc_string(end_exclusive),
            },
            "aggregationInterval": {
                "of": "P1D"
            },
            "evalscript": EVALSCRIPT,
            "resx": RESOLUTION_METERS,
            "resy": RESOLUTION_METERS,
        },
        "calculations": {
            "ndre": {
                "statistics": {
                    "default": {
                        "percentiles": {
                            "k": [
                                10,
                                25,
                                50,
                                75,
                                90,
                            ]
                        }
                    }
                },
                "histograms": {
                    "default": {
                        "bins": [
                            -1.0,
                            0.0,
                            0.2,
                            0.4,
                            0.6,
                            0.8,
                            1.0,
                        ]
                    }
                },
            }
        },
    }

    response = requests.post(
        STATISTICS_URL,
        headers=get_headers(token),
        json=request_body,
        timeout=180,
    )

    if not response.ok:
        raise RuntimeError(
            "Échec de la Statistical API : "
            f"{response.status_code} - {response.text}"
        )

    return response.json()


# ==============================================================
# TRANSFORMATION DE LA RÉPONSE EN DATAFRAME
# ==============================================================

def get_percentile(
    percentiles: dict[str, Any],
    percentile: int,
) -> float | None:
    """Lit un percentile quelle que soit la forme de sa clé."""

    possible_keys = [
        str(percentile),
        f"{percentile}.0",
    ]

    for key in possible_keys:
        if key in percentiles:
            return float(percentiles[key])

    return None


def get_bin_count(
    bins: list[dict[str, Any]],
    expected_low: float,
    expected_high: float,
) -> int:
    """Récupère le nombre de pixels d'une classe NDRE."""

    tolerance = 1e-6

    for histogram_bin in bins:
        low_edge = float(
            histogram_bin.get("lowEdge", 0)
        )
        high_edge = float(
            histogram_bin.get("highEdge", 0)
        )

        if (
            abs(low_edge - expected_low) < tolerance
            and abs(high_edge - expected_high) < tolerance
        ):
            return int(
                histogram_bin.get("count", 0)
            )

    return 0


def parse_statistics(
    response_data: dict[str, Any],
) -> pd.DataFrame:
    """Transforme la réponse Statistical API en DataFrame."""

    rows: list[dict[str, Any]] = []

    geometry_pixel_count = response_data.get(
        "geometryPixelCount"
    )

    for interval_data in response_data.get("data", []):
        interval = interval_data.get("interval", {})

        interval_start = interval.get("from")

        if not interval_start:
            continue

        acquisition_date = interval_start[:10]

        band_data = (
            interval_data
            .get("outputs", {})
            .get("ndre", {})
            .get("bands", {})
            .get("B0", {})
        )

        statistics = band_data.get("stats", {})
        histogram = band_data.get("histogram", {})

        if not statistics:
            rows.append(
                {
                    "acquisition_date": acquisition_date,
                    "geometry_pixel_count": (
                        geometry_pixel_count
                    ),
                }
            )
            continue

        sample_count = int(
            statistics.get("sampleCount", 0) or 0
        )

        no_data_count = int(
            statistics.get("noDataCount", 0) or 0
        )

        valid_pixel_count = max(
            sample_count - no_data_count,
            0,
        )

        denominator = (
            geometry_pixel_count
            if geometry_pixel_count
            else sample_count
        )

        if denominator:
            valid_pixel_ratio = min(
                valid_pixel_count / denominator,
                1.0,
            )
        else:
            valid_pixel_ratio = None

        percentiles = statistics.get(
            "percentiles",
            {},
        )

        bins = histogram.get("bins", [])

        threshold_counts = {
            "ratio_ndre_below_0": get_bin_count(
                bins,
                -1.0,
                0.0,
            ),
            "ratio_ndre_0_0_2": get_bin_count(
                bins,
                0.0,
                0.2,
            ),
            "ratio_ndre_0_2_0_4": get_bin_count(
                bins,
                0.2,
                0.4,
            ),
            "ratio_ndre_0_4_0_6": get_bin_count(
                bins,
                0.4,
                0.6,
            ),
            "ratio_ndre_0_6_0_8": get_bin_count(
                bins,
                0.6,
                0.8,
            ),
            "ratio_ndre_above_0_8": get_bin_count(
                bins,
                0.8,
                1.0,
            ),
        }

        histogram_pixel_count = sum(
            threshold_counts.values()
        )

        if histogram_pixel_count > 0:
            threshold_ratios = {
                column: count / histogram_pixel_count
                for column, count
                in threshold_counts.items()
            }
        else:
            threshold_ratios = {
                column: None
                for column in threshold_counts
            }

        rows.append(
            {
                "acquisition_date": acquisition_date,
                "interval_from": interval.get("from"),
                "interval_to": interval.get("to"),

                "geometry_pixel_count": (
                    geometry_pixel_count
                ),
                "sample_count": sample_count,
                "no_data_count": no_data_count,
                "valid_pixel_count": valid_pixel_count,
                "valid_pixel_ratio": valid_pixel_ratio,

                "ndre_min": statistics.get("min"),
                "ndre_max": statistics.get("max"),
                "ndre_mean": statistics.get("mean"),
                "ndre_std": statistics.get("stDev"),

                "ndre_p10": get_percentile(
                    percentiles,
                    10,
                ),
                "ndre_p25": get_percentile(
                    percentiles,
                    25,
                ),
                "ndre_median": get_percentile(
                    percentiles,
                    50,
                ),
                "ndre_p75": get_percentile(
                    percentiles,
                    75,
                ),
                "ndre_p90": get_percentile(
                    percentiles,
                    90,
                ),

                **threshold_ratios,
            }
        )

    return pd.DataFrame(rows)


# ==============================================================
# TENDANCES
# ==============================================================

def add_trend_columns(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """Ajoute des indicateurs simples d'évolution."""

    dataframe = dataframe.copy()

    dataframe["acquisition_date"] = pd.to_datetime(
        dataframe["acquisition_date"],
        errors="coerce",
    )

    dataframe = dataframe.sort_values(
        "acquisition_date"
    ).reset_index(drop=True)

    dataframe["valid_pixel_ratio"] = pd.to_numeric(
        dataframe.get("valid_pixel_ratio"),
        errors="coerce",
    )

    dataframe["ndre_mean"] = pd.to_numeric(
        dataframe.get("ndre_mean"),
        errors="coerce",
    )

    dataframe["is_usable"] = (
        dataframe["valid_pixel_ratio"]
        >= MIN_VALID_PIXEL_RATIO
    ) & dataframe["ndre_mean"].notna()

    dataframe[
        "ndre_change_from_previous_usable"
    ] = pd.NA

    dataframe["ndre_rolling_mean_3"] = pd.NA

    usable_indexes = dataframe.index[
        dataframe["is_usable"]
    ]

    usable_values = dataframe.loc[
        usable_indexes,
        "ndre_mean",
    ]

    dataframe.loc[
        usable_indexes,
        "ndre_change_from_previous_usable",
    ] = usable_values.diff().values

    dataframe.loc[
        usable_indexes,
        "ndre_rolling_mean_3",
    ] = (
        usable_values
        .rolling(
            window=3,
            min_periods=1,
        )
        .mean()
        .values
    )

    return dataframe


# ==============================================================
# GRAPHIQUE
# ==============================================================

def create_plot(
    dataframe: pd.DataFrame,
    parcel_name: str,
    output_file: Path,
) -> None:
    """Crée une courbe temporelle du NDRE."""

    plot_data = dataframe[
        dataframe["is_usable"]
        & dataframe["ndre_mean"].notna()
    ].copy()

    if plot_data.empty:
        print(
            "Aucun graphique créé : "
            "aucune acquisition exploitable."
        )
        return

    figure, axis = plt.subplots(
        figsize=(12, 6)
    )

    axis.plot(
        plot_data["acquisition_date"],
        plot_data["ndre_mean"],
        marker="o",
        label="NDRE moyen",
    )

    axis.plot(
        plot_data["acquisition_date"],
        plot_data["ndre_median"],
        marker=".",
        label="NDRE médian",
    )

    axis.plot(
        plot_data["acquisition_date"],
        pd.to_numeric(
            plot_data["ndre_rolling_mean_3"],
            errors="coerce",
        ),
        label="Moyenne mobile sur 3 acquisitions",
    )

    axis.set_title(
        f"Évolution du NDRE — {parcel_name}"
    )

    axis.set_xlabel("Date d'acquisition")
    axis.set_ylabel("NDRE")
    axis.set_ylim(-1, 1)
    axis.grid(True)
    axis.legend()

    figure.autofmt_xdate()
    figure.tight_layout()

    figure.savefig(
        output_file,
        dpi=150,
        bbox_inches="tight",
    )

    plt.close(figure)


# ==============================================================
# PROGRAMME PRINCIPAL
# ==============================================================

def main() -> None:
    start_date, end_date = parse_dates()

    geometry, properties = load_parcel(
        SELECTED_SITE_ID
    )

    statistics_geometry, statistics_crs_url, statistics_epsg = (
        project_geometry_to_local_utm(geometry)
    )

    print(
        f"Projection utilisée pour les statistiques : "
        f"EPSG:{statistics_epsg}"
    )

    parcel_name = properties.get(
        "name",
        SELECTED_SITE_ID,
    )

    crop_type = properties.get(
        "crop_type"
    )

    print(
        f"Parcelle : "
        f"{SELECTED_SITE_ID} - {parcel_name}"
    )

    print(
        f"Période : "
        f"{start_date} → {end_date}"
    )

    token = get_access_token()

    print("Recherche des acquisitions Sentinel-2...")

    catalog_features = search_catalog(
        token=token,
        geometry=geometry,
        start_date=start_date,
        end_date=end_date,
    )

    catalog_dataframe = build_catalog_dataframe(
        catalog_features
    )

    print(
        f"Produits trouvés : "
        f"{len(catalog_features)}"
    )

    print("Calcul des statistiques NDRE...")

    statistics_response = request_ndre_statistics(
        token=token,
        geometry=statistics_geometry,
        geometry_crs_url=statistics_crs_url,
        start_date=start_date,
        end_date=end_date,
    )

    statistics_dataframe = parse_statistics(
        statistics_response
    )

    if (
        catalog_dataframe.empty
        and statistics_dataframe.empty
    ):
        raise RuntimeError(
            "Aucune acquisition ou statistique "
            "n'a été trouvée sur cette période."
        )

    if catalog_dataframe.empty:
        dataframe = statistics_dataframe

    elif statistics_dataframe.empty:
        dataframe = catalog_dataframe

    else:
        dataframe = catalog_dataframe.merge(
            statistics_dataframe,
            on="acquisition_date",
            how="outer",
        )

    dataframe["site_id"] = SELECTED_SITE_ID
    dataframe["name"] = parcel_name
    dataframe["crop_type"] = crop_type

    dataframe = add_trend_columns(
        dataframe
    )

    PROCESSED_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    csv_file = (
        PROCESSED_DIR
        / f"ndre_timeseries_{SELECTED_SITE_ID}.csv"
    )

    catalog_json_file = (
        PROCESSED_DIR
        / f"catalog_{SELECTED_SITE_ID}_raw.json"
    )

    statistics_json_file = (
        PROCESSED_DIR
        / f"ndre_statistics_{SELECTED_SITE_ID}_raw.json"
    )

    plot_file = (
        OUTPUT_DIR
        / f"ndre_timeseries_{SELECTED_SITE_ID}.png"
    )

    dataframe.to_csv(
        csv_file,
        index=False,
        encoding="utf-8",
    )

    if SAVE_RAW_RESPONSES:
        with catalog_json_file.open(
            mode="w",
            encoding="utf-8",
        ) as file:
            json.dump(
                {
                    "type": "FeatureCollection",
                    "features": catalog_features,
                },
                file,
                indent=2,
                ensure_ascii=False,
            )

        with statistics_json_file.open(
            mode="w",
            encoding="utf-8",
        ) as file:
            json.dump(
                statistics_response,
                file,
                indent=2,
                ensure_ascii=False,
            )

    create_plot(
        dataframe=dataframe,
        parcel_name=parcel_name,
        output_file=plot_file,
    )

    usable_count = int(
        dataframe["is_usable"].fillna(False).sum()
    )

    print()
    print("Traitement terminé.")
    print(
        f"Dates trouvées : {len(dataframe)}"
    )
    print(
        f"Dates exploitables : {usable_count}"
    )
    print(f"DataFrame : {csv_file}")
    print(f"Graphique : {plot_file}")

    if SAVE_RAW_RESPONSES:
        print(f"Catalogue brut : {catalog_json_file}")
        print(
            f"Statistiques brutes : "
            f"{statistics_json_file}"
        )

    display_columns = [
        "acquisition_date",
        "ndre_mean",
        "ndre_median",
        "ndre_std",
        "valid_pixel_ratio",
        "tile_cloud_cover_mean",
        "is_usable",
    ]

    existing_columns = [
        column
        for column in display_columns
        if column in dataframe.columns
    ]

    print()
    print(
        dataframe[
            existing_columns
        ].tail(15).to_string(index=False)
    )


if __name__ == "__main__":
    main()
