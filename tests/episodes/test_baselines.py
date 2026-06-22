from __future__ import annotations

import numpy as np

from eve.episodes.baselines import (
    percentile_rank,
    reference_years_for_target,
)


def test_loyo_excludes_target_year() -> None:
    years = list(range(2016, 2026))
    result = reference_years_for_target(2022, years, "retrospective_loyo")

    assert 2022 not in result
    assert result == [2016, 2017, 2018, 2019, 2020, 2021, 2023, 2024, 2025]


def test_prospective_uses_only_past_years() -> None:
    result = reference_years_for_target(
        2022,
        range(2016, 2026),
        "prospective",
    )

    assert result == [2016, 2017, 2018, 2019, 2020, 2021]


def test_percentile_rank_is_empirical_and_low_tailed() -> None:
    reference = np.array([1.0, 2.0, 3.0, 4.0])

    assert percentile_rank(1.0, reference) == 25.0
    assert percentile_rank(2.5, reference) == 50.0
    assert percentile_rank(0.0, reference) == 0.0
