from pathlib import Path

import pandas as pd


# analyze_ndvi.py -> analysis -> eve -> src -> racine EVE
PROJECT_ROOT = Path(__file__).resolve().parents[3]

NDVI_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "ndvi_summary.csv"
)


def load_ndvi_dataframe() -> pd.DataFrame:
    """Charge les résultats NDVI générés par le pipeline Sentinel."""

    if not NDVI_FILE.exists():
        raise FileNotFoundError(
            "Le fichier NDVI n'existe pas encore.\n"
            "Lance d'abord :\n"
            "python src/eve/sentinel/build_ndvi_dataframe.py"
        )

    dataframe = pd.read_csv(NDVI_FILE)

    return dataframe


def main() -> None:
    df = load_ndvi_dataframe()

    print("Aperçu du DataFrame :")
    print(df.head())

    print("\nDimensions :")
    print(df.shape)

    print("\nColonnes :")
    print(df.columns.tolist())

    print("\nTypes :")
    print(df.dtypes)

    print("\nStatistiques NDVI :")
    print(
        df[
            [
                "mean_ndvi",
                "median_ndvi",
                "std_ndvi",
                "min_ndvi",
                "max_ndvi",
                "valid_pixel_ratio",
            ]
        ].describe()
    )


if __name__ == "__main__":
    main()