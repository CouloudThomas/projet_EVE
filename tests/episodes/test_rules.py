from __future__ import annotations

import pandas as pd

from eve.episodes.rules import (
    classify_hydro_context,
    classify_ndmi,
    classify_satellite_quality,
    propose_decision,
)


def base_row(**overrides):
    values = {
        "in_scope_season": True,
        "satellite_quality": "good",
        "ndmi_state": "normal",
        "hydro_context": "normal",
        "vpd_state": "normal",
        "persistence_state": "not_applicable",
    }
    values.update(overrides)
    return pd.Series(values)


def test_poor_observation_never_creates_agronomic_candidate() -> None:
    decision = propose_decision(
        base_row(
            satellite_quality="poor",
            ndmi_state="very_low",
            hydro_context="very_dry",
            vpd_state="very_high",
        )
    )

    assert decision.rule_id == "R1"
    assert decision.is_candidate is False
    assert decision.eve_decision == "abstain_insufficient_observation"


def test_priority_rule_precedes_simple_field_check() -> None:
    decision = propose_decision(
        base_row(
            ndmi_state="very_low",
            hydro_context="very_dry",
            vpd_state="very_high",
            persistence_state="persistent",
        )
    )

    assert decision.rule_id == "R6"
    assert decision.recommended_information_action == "priority_field_check"


def test_normal_date_is_kept_as_wait() -> None:
    decision = propose_decision(base_row())

    assert decision.rule_id == "R2"
    assert decision.eve_decision == "wait"
    assert decision.is_candidate is False


def test_missing_percentile_stays_unknown() -> None:
    config = {
        "low_percentile_lte": 20,
        "very_low_percentile_lte": 10,
    }

    assert classify_ndmi(None, config) == "unknown"


def test_unusable_flag_forces_poor_quality() -> None:
    config = {"usable_valid_ratio_gte": 0.70, "good_valid_ratio_gte": 0.90}

    assert classify_satellite_quality(1.0, False, config) == "poor"


def test_hydro_context_has_only_two_non_redundant_votes() -> None:
    assert classify_hydro_context("low", "low") == "very_dry"
    assert classify_hydro_context("low", "normal") == "dry"
    assert classify_hydro_context("normal", "normal") == "normal"
