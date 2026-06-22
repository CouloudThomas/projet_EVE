from __future__ import annotations

from typing import Any

import pandas as pd


LOW_NDMI_STATES = {"low", "very_low"}
USABLE_QUALITY_STATES = {"usable", "good"}


def add_causal_persistence(
    decision_points: pd.DataFrame,
    *,
    lookback_days: int,
    persistent_count: int,
    strong_persistent_count: int,
) -> pd.DataFrame:
    """Compte les signaux bas courants/passés sans utiliser le futur."""

    result = decision_points.sort_values(
        ["parcel_id", "acquisition_date"]
    ).copy()
    result["low_signal_count_15d"] = 0
    result["persistence_state"] = "not_applicable"

    for _, group in result.groupby("parcel_id", sort=False):
        previous_low_dates: list[pd.Timestamp] = []

        for index, row in group.iterrows():
            current_date = pd.Timestamp(row["acquisition_date"])
            cutoff = current_date - pd.Timedelta(days=lookback_days)
            previous_low_dates = [
                date for date in previous_low_dates if date >= cutoff
            ]
            is_low = (
                bool(row["in_scope_season"])
                and row["ndmi_state"] in LOW_NDMI_STATES
                and row["satellite_quality"] in USABLE_QUALITY_STATES
            )

            if not is_low:
                continue

            previous_low_dates.append(current_date)
            count = len(previous_low_dates)
            result.at[index, "low_signal_count_15d"] = count

            if count >= strong_persistent_count:
                state = "strong_persistent"
            elif count >= persistent_count:
                state = "persistent"
            else:
                state = "single"

            result.at[index, "persistence_state"] = state

    return result


def assign_episode_ids(
    decision_points: pd.DataFrame,
    *,
    maximum_gap_days: int,
) -> pd.DataFrame:
    """Regroupe les points candidats séparés d'au plus maximum_gap_days."""

    result = decision_points.copy()
    result["episode_id"] = pd.NA

    for parcel_id, group in result[result["is_candidate"]].groupby(
        "parcel_id",
        sort=True,
    ):
        group = group.sort_values("acquisition_date")
        episode_number = 0
        previous_date: pd.Timestamp | None = None
        current_episode_id: str | None = None

        for index, row in group.iterrows():
            current_date = pd.Timestamp(row["acquisition_date"])
            starts_new = (
                previous_date is None
                or (current_date - previous_date).days > maximum_gap_days
            )

            if starts_new:
                episode_number += 1
                current_episode_id = (
                    f"{parcel_id}_episode_"
                    f"{current_date:%Y%m%d}_{episode_number:02d}"
                )

            result.at[index, "episode_id"] = current_episode_id
            previous_date = current_date

    return result


def _strongest_value(series: pd.Series, order: dict[str, int]) -> str:
    values = [value for value in series.dropna() if value in order]
    if not values:
        return "unknown"
    return max(values, key=lambda value: order[value])


def build_episode_table(
    decision_points: pd.DataFrame,
    *,
    cost_model_version: str,
    cost_status: str,
) -> pd.DataFrame:
    """Agrège les points candidats en événements temporels traçables."""

    candidates = decision_points[
        decision_points["is_candidate"]
        & decision_points["episode_id"].notna()
    ].copy()

    if candidates.empty:
        return pd.DataFrame()

    urgency_order = {"none": 0, "low": 1, "medium": 2, "high": 3}
    hydro_order = {"unknown": 0, "normal": 1, "dry": 2, "very_dry": 3}
    vpd_order = {"unknown": 0, "normal": 1, "high": 2, "very_high": 3}
    persistence_order = {
        "not_applicable": 0,
        "single": 1,
        "persistent": 2,
        "strong_persistent": 3,
    }

    episodes: list[dict[str, Any]] = []

    for episode_id, group in candidates.groupby("episode_id", sort=True):
        group = group.sort_values("acquisition_date")
        peak_index = group["ndmi_percentile"].astype(float).idxmin()
        peak = group.loc[peak_index]
        maximum_urgency = _strongest_value(group["urgency"], urgency_order)
        urgency_group = group[group["urgency"].eq(maximum_urgency)]
        action_row = urgency_group.iloc[-1]
        proposed_action = action_row["recommended_information_action"]
        codes = sorted(
            {
                code
                for value in group["justification_codes"].dropna()
                for code in str(value).split("|")
                if code
            }
        )
        start = pd.Timestamp(group["acquisition_date"].min())
        end = pd.Timestamp(group["acquisition_date"].max())

        episodes.append(
            {
                "episode_id": episode_id,
                "parcel_id": group.iloc[0]["parcel_id"],
                "episode_start_date": start.date().isoformat(),
                "episode_end_date": end.date().isoformat(),
                "peak_date": pd.Timestamp(peak["acquisition_date"])
                .date()
                .isoformat(),
                "duration_days": int((end - start).days),
                "decision_point_count": int(len(group)),
                "dominant_index": "NDMI_BUFFER_10M",
                "worst_ndmi_value": float(group["ndmi_value"].min()),
                "worst_ndmi_anomaly": float(
                    group["ndmi_anomaly_value"].min()
                ),
                "lowest_ndmi_percentile": float(
                    group["ndmi_percentile"].min()
                ),
                "best_valid_pixel_ratio": float(
                    group["valid_pixel_ratio"].max()
                ),
                "dominant_hydro_context": _strongest_value(
                    group["hydro_context"], hydro_order
                ),
                "highest_vpd_state": _strongest_value(
                    group["vpd_state"], vpd_order
                ),
                "persistence_state": _strongest_value(
                    group["persistence_state"], persistence_order
                ),
                "maximum_urgency": maximum_urgency,
                "proposed_action": proposed_action,
                "justification_codes": "|".join(codes),
                "rules_version": group.iloc[0]["rules_version"],
                "baseline_version": group.iloc[0]["baseline_version"],
                "evaluation_mode": group.iloc[0]["evaluation_mode"],
                "cost_model_version": cost_model_version,
                "cost_status": cost_status,
                "estimated_time_min_low": int(
                    action_row["estimated_time_min_low"]
                ),
                "estimated_time_min_high": int(
                    action_row["estimated_time_min_high"]
                ),
                "estimated_cost_eur_low": float(
                    action_row["estimated_cost_eur_low"]
                ),
                "estimated_cost_eur_high": float(
                    action_row["estimated_cost_eur_high"]
                ),
                "delay_cost_estimated": bool(
                    action_row["delay_cost_estimated"]
                ),
                "verification_status": "not_started",
                "verification_method": "none",
                "verification_result": "not_available",
                "human_decision_available": False,
                "human_final_decision": pd.NA,
                "generated_at": group.iloc[0]["generated_at"],
            }
        )

    return pd.DataFrame(episodes)
