from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import rasterio


PROJECT_ROOT = Path(__file__).resolve().parents[3]

INPUT_FILE = PROJECT_ROOT / "data" / "output" / "site_001_ndvi.tif"
OUTPUT_FILE = PROJECT_ROOT / "data" / "output" / "ndvi_test.png"


def create_preview() -> None:
    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"Fichier introuvable : {INPUT_FILE}")

    with rasterio.open(INPUT_FILE) as dataset:
        ndvi = dataset.read(1).astype(np.float32)

        print(f"Format : {dataset.driver}")
        print(f"Dimensions : {dataset.width} × {dataset.height}")
        print(f"Type : {dataset.dtypes[0]}")
        print(f"CRS : {dataset.crs}")
        print(f"Limites géographiques : {dataset.bounds}")

    # Masque les pixels invalides créés par notre evalscript.
    invalid_pixels = ~np.isfinite(ndvi) | (ndvi <= -9990)
    ndvi_masked = np.ma.array(ndvi, mask=invalid_pixels)

    if ndvi_masked.count() == 0:
        raise RuntimeError(
            "Le fichier ne contient aucun pixel NDVI valide."
        )

    print(f"NDVI minimum : {ndvi_masked.min():.3f}")
    print(f"NDVI maximum : {ndvi_masked.max():.3f}")
    print(f"NDVI moyen : {ndvi_masked.mean():.3f}")

    figure, axis = plt.subplots(figsize=(10, 8))

    image = axis.imshow(
        ndvi_masked,
        cmap="RdYlGn",
        vmin=-1,
        vmax=1,
    )

    axis.set_title("Aperçu NDVI Sentinel-2")
    axis.axis("off")

    figure.colorbar(
        image,
        ax=axis,
        label="NDVI",
        fraction=0.046,
        pad=0.04,
    )

    figure.tight_layout()
    figure.savefig(OUTPUT_FILE, dpi=150, bbox_inches="tight")
    plt.close(figure)

    print(f"Aperçu créé : {OUTPUT_FILE}")


if __name__ == "__main__":
    create_preview()