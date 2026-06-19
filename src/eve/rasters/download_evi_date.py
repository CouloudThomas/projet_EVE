from __future__ import annotations

import argparse
import os

from raster_common import download_index_for_date


EVALSCRIPT = """
//VERSION=3

function setup() {
    return {
        input: ["B02", "B04", "B08", "SCL", "dataMask"],
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

    const denominator =
        sample.B08
        + 6.0 * sample.B04
        - 7.5 * sample.B02
        + 1.0;

    if (denominator === 0) {
        return [-9999];
    }

    return [
        2.5 * (sample.B08 - sample.B04) / denominator
    ];
}
"""


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Télécharge un GeoTIFF EVI pour une date précise."
    )
    parser.add_argument(
        "--site",
        default=os.getenv(
            "EVE_SITE_ID",
            "site_001",
        ),
    )
    parser.add_argument("--date", required=True)
    args = parser.parse_args()

    download_index_for_date(
        site_id=args.site,
        acquisition_date=args.date,
        index_name="EVI",
        resolution_m=10,
        evalscript=EVALSCRIPT,
    )


if __name__ == "__main__":
    main()
