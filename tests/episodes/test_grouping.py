from __future__ import annotations

import pandas as pd

from eve.episodes.grouping import (
    add_causal_persistence,
    assign_episode_ids,
)


def test_persistence_uses_current_and_past_only() -> None:
    frame = pd.DataFrame(
        {
            "parcel_id": ["site_004"] * 3,
            "acquisition_date": pd.to_datetime(
                ["2022-08-01", "2022-08-10", "2022-08-20"]
            ),
            "in_scope_season": [True] * 3,
            "ndmi_state": ["low", "low", "low"],
            "satellite_quality": ["good"] * 3,
        }
    )

    result = add_causal_persistence(
        frame,
        lookback_days=15,
        persistent_count=2,
        strong_persistent_count=3,
    )

    assert result["low_signal_count_15d"].tolist() == [1, 2, 2]
    assert result["persistence_state"].tolist() == [
        "single",
        "persistent",
        "persistent",
    ]


def test_candidates_within_fifteen_days_share_episode() -> None:
    frame = pd.DataFrame(
        {
            "parcel_id": ["site_004"] * 3,
            "acquisition_date": pd.to_datetime(
                ["2022-08-01", "2022-08-12", "2022-09-01"]
            ),
            "is_candidate": [True, True, True],
        }
    )

    result = assign_episode_ids(frame, maximum_gap_days=15)

    assert result.loc[0, "episode_id"] == result.loc[1, "episode_id"]
    assert result.loc[1, "episode_id"] != result.loc[2, "episode_id"]


def test_episode_ids_are_deterministic() -> None:
    frame = pd.DataFrame(
        {
            "parcel_id": ["site_004", "site_004"],
            "acquisition_date": pd.to_datetime(
                ["2022-08-01", "2022-09-01"]
            ),
            "is_candidate": [True, True],
        }
    )

    first = assign_episode_ids(frame, maximum_gap_days=15)
    second = assign_episode_ids(frame, maximum_gap_days=15)

    assert first["episode_id"].tolist() == second["episode_id"].tolist()
