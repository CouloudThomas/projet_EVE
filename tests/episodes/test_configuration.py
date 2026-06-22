from __future__ import annotations

from pathlib import Path

from eve.episodes.baselines import load_structured_config


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def test_hydro_score_does_not_double_count_dryness_index() -> None:
    config = load_structured_config(
        PROJECT_ROOT / "config" / "episode_rules_v0_1.yaml"
    )
    dimensions = config["hydro_context"]["dimensions"]

    assert dimensions == [
        "water_balance_30d",
        "soil_moisture_7_28cm_mean_30d",
    ]
    assert "dryness_index_30d" not in dimensions


def test_precipitation_is_descriptive_not_a_decision_vote() -> None:
    config = load_structured_config(
        PROJECT_ROOT / "config" / "episode_rules_v0_1.yaml"
    )

    assert config["precipitation"]["decision_vote"] is False
