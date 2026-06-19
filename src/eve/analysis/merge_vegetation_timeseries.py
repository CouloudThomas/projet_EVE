from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parents[3]

load_dotenv(PROJECT_ROOT / ".env")

SELECTED_SITE_ID = os.getenv(
    "EVE_SITE_ID",
    "site_001",
)

PROCESSED_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
)


GENERIC_COLUMNS = [
    "geometry_pixel_count",
    "sample_count",
    "no_data_count",
    "valid_pixel_count",
    "valid_pixel_ratio",
    "is_usable",
]


def load_index(
    prefix: str,
    keep_metadata: bool = False,
) -> pd.DataFrame:
    csv_file = (
        PROCESSED_DIR
        / f"{prefix}_timeseries_{SELECTED_SITE_ID}.csv"
    )

    if not csv_file.exists():
        raise FileNotFoundError(
            f"Fichier introuvable : {csv_file}"
        )

    dataframe = pd.read_csv(
        csv_file,
        parse_dates=["acquisition_date"],
    )

    rename_map = {
        column: f"{prefix}_{column}"
        for column in GENERIC_COLUMNS
        if column in dataframe.columns
    }

    dataframe = dataframe.rename(
        columns=rename_map
    )

    key_columns = [
        "site_id",
        "acquisition_date",
    ]

    index_columns = [
        column
        for column in dataframe.columns
        if (
            column.startswith(f"{prefix}_")
            or column.startswith(
                f"ratio_{prefix}_"
            )
        )
    ]

    if keep_metadata:
        metadata_columns = [
            "site_id",
            "name",
            "crop_type",
            "acquisition_date",
            "acquisition_count",
            "product_ids",
            "acquisition_datetimes",
            "tile_cloud_cover_min",
            "tile_cloud_cover_mean",
            "tile_cloud_cover_max",
        ]

        columns = [
            column
            for column in metadata_columns
            if column in dataframe.columns
        ] + index_columns

    else:
        columns = key_columns + index_columns

    columns = list(dict.fromkeys(columns))

    return dataframe[columns].copy()


def main() -> None:
    ndvi = load_index(
        "ndvi",
        keep_metadata=True,
    )

    evi = load_index("evi")
    ndmi = load_index("ndmi")
    ndre = load_index("ndre")

    merge_keys = [
        "site_id",
        "acquisition_date",
    ]

    vegetation = (
        ndvi
        .merge(
            evi,
            on=merge_keys,
            how="outer",
        )
        .merge(
            ndmi,
            on=merge_keys,
            how="outer",
        )
        .merge(
            ndre,
            on=merge_keys,
            how="outer",
        )
        .sort_values("acquisition_date")
        .reset_index(drop=True)
    )

    usable_columns = [
        "ndvi_is_usable",
        "evi_is_usable",
        "ndmi_is_usable",
        "ndre_is_usable",
    ]

    for column in usable_columns:
        if column not in vegetation.columns:
            vegetation[column] = False

    vegetation["usable_all_indices"] = (
        vegetation[usable_columns]
        .fillna(False)
        .all(axis=1)
    )

    output_file = (
        PROCESSED_DIR
        / (
            "vegetation_timeseries_"
            f"{SELECTED_SITE_ID}.csv"
        )
    )

    vegetation.to_csv(
        output_file,
        index=False,
        encoding="utf-8",
    )

    print("Fusion terminée.")
    print(f"Lignes : {len(vegetation)}")
    print(f"Colonnes : {len(vegetation.columns)}")
    print(f"Fichier : {output_file}")

    columns_to_display = [
        "acquisition_date",
        "ndvi_mean",
        "evi_mean",
        "ndmi_mean",
        "ndre_mean",
        "usable_all_indices",
    ]

    existing_columns = [
        column
        for column in columns_to_display
        if column in vegetation.columns
    ]

    print()
    print(
        vegetation[
            existing_columns
        ].tail(15).to_string(index=False)
    )


if __name__ == "__main__":
    main()
