from __future__ import annotations

import argparse
import os

from raster_common import download_index_for_date


EVALSCRIPT = """
//VERSION=3

function setup() {
    return {
        input: ["B08", "B11", "SCL", "dataMask"],
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

    const denominator = sample.B08 + sample.B11;

    if (denominator === 0) {
        return [-9999];
    }

    return [(sample.B08 - sample.B11) / denominator];
}
"""


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Télécharge un GeoTIFF NDMI pour une date précise."
    )
    parser.add_argument(
        "--site",
        default=os.getenv(
            "EVE_SITE_ID",
            "site_003",
        ),
    )
    parser.add_argument("--date", required=True)
    args = parser.parse_args()

    download_index_for_date(
        site_id=args.site,
        acquisition_date=args.date,
        index_name="NDMI",
        resolution_m=20,
        evalscript=EVALSCRIPT,
    )


if __name__ == "__main__":
    main()
