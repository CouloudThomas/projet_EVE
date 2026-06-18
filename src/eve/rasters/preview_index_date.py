from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import rasterio


PROJECT_ROOT = Path(__file__).resolve().parents[3]
RASTERS_OUTPUT_DIR = PROJECT_ROOT / "data" / "output" / "rasters"

DISPLAY_SETTINGS = {
    "ndvi": {
        "minimum": -1.0,
        "maximum": 1.0,
        "cmap": "RdYlGn",
    },
    "evi": {
        "minimum": -1.0,
        "maximum": 1.5,
        "cmap": "RdYlGn",
    },
    "ndmi": {
        "minimum": -1.0,
        "maximum": 1.0,
        "cmap": "BrBG",
    },
    "ndre": {
        "minimum": -1.0,
        "maximum": 1.0,
        "cmap": "RdYlGn",
    },
}


def create_preview(
    *,
    site_id: str,
    acquisition_date: str,
    index_name: str,
) -> Path:
    normalized_index = index_name.lower()

    if normalized_index not in DISPLAY_SETTINGS:
        raise ValueError(
            "Indice inconnu. Valeurs possibles : "
            "ndvi, evi, ndmi, ndre."
        )

    input_file = (
        RASTERS_OUTPUT_DIR
        / site_id
        / normalized_index
        / f"{acquisition_date}_{normalized_index}.tif"
    )
    output_file = input_file.with_suffix(".png")

    if not input_file.exists():
        raise FileNotFoundError(
            f"GeoTIFF introuvable : {input_file}"
        )

    with rasterio.open(input_file) as dataset:
        values = dataset.read(1).astype(np.float32)

        print(f"Format : {dataset.driver}")
        print(f"Dimensions : {dataset.width} × {dataset.height}")
        print(f"Type : {dataset.dtypes[0]}")
        print(f"CRS : {dataset.crs}")
        print(f"Limites : {dataset.bounds}")

    invalid_pixels = (
        ~np.isfinite(values)
        | (values <= -9990)
    )

    masked_values = np.ma.array(
        values,
        mask=invalid_pixels,
    )

    if masked_values.count() == 0:
        raise RuntimeError(
            f"Le fichier ne contient aucun pixel "
            f"{normalized_index.upper()} valide."
        )

    print(
        f"{normalized_index.upper()} minimum : "
        f"{float(masked_values.min()):.3f}"
    )
    print(
        f"{normalized_index.upper()} moyen : "
        f"{float(masked_values.mean()):.3f}"
    )
    print(
        f"{normalized_index.upper()} maximum : "
        f"{float(masked_values.max()):.3f}"
    )
    print(f"Pixels valides : {masked_values.count()}")

    settings = DISPLAY_SETTINGS[normalized_index]

    figure, axis = plt.subplots(figsize=(10, 8))

    image = axis.imshow(
        masked_values,
        cmap=settings["cmap"],
        vmin=settings["minimum"],
        vmax=settings["maximum"],
    )

    axis.set_title(
        f"{normalized_index.upper()} — "
        f"{site_id} — {acquisition_date}"
    )
    axis.axis("off")

    figure.colorbar(
        image,
        ax=axis,
        label=normalized_index.upper(),
        fraction=0.046,
        pad=0.04,
    )

    figure.tight_layout()
    figure.savefig(
        output_file,
        dpi=180,
        bbox_inches="tight",
    )
    plt.close(figure)

    print(f"PNG créé : {output_file}")

    return output_file


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Crée le PNG d'un GeoTIFF NDVI/EVI/NDMI/NDRE."
    )
    parser.add_argument(
        "--index",
        required=True,
        choices=["ndvi", "evi", "ndmi", "ndre"],
    )
    parser.add_argument("--site", default="site_001")
    parser.add_argument("--date", required=True)
    args = parser.parse_args()

    create_preview(
        site_id=args.site,
        acquisition_date=args.date,
        index_name=args.index,
    )


if __name__ == "__main__":
    main()
