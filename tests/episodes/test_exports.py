from __future__ import annotations

import pandas as pd

from eve.episodes.build_episode_store import build_episode_export_tables


def _episode(
    episode_id: str,
    action: str,
    *,
    urgency: str = "medium",
    ndmi_percentile: float = 12.0,
    duration_days: int = 0,
) -> dict[str, object]:
    return {
        "episode_id": episode_id,
        "episode_start_date": "2022-08-01",
        "episode_end_date": "2022-08-01",
        "duration_days": duration_days,
        "decision_point_count": 1,
        "lowest_ndmi_percentile": ndmi_percentile,
        "worst_ndmi_anomaly": -0.1,
        "maximum_urgency": urgency,
        "proposed_action": action,
    }


def test_episode_exports_keep_field_checks_visible() -> None:
    episodes = pd.DataFrame(
        [
            _episode("priority", "priority_field_check", urgency="high"),
            _episode("field", "field_check", ndmi_percentile=18.0),
            _episode("review_extreme", "review_raster", ndmi_percentile=3.0),
            _episode("review_long", "review_raster", duration_days=45),
            _episode("review_moderate", "review_raster", ndmi_percentile=12.0),
        ]
    )

    exports = build_episode_export_tables(episodes)

    assert set(exports) == {
        "information_queue",
        "audit_shortlist",
        "review_raster",
        "field_check",
        "priority_field_check",
    }
    assert exports["field_check"]["episode_id"].tolist() == ["field"]
    assert "field" in set(exports["audit_shortlist"]["episode_id"])
    assert "priority" in set(exports["audit_shortlist"]["episode_id"])
    assert "review_extreme" in set(exports["audit_shortlist"]["episode_id"])
    assert "review_long" in set(exports["audit_shortlist"]["episode_id"])
    assert "review_moderate" not in set(exports["audit_shortlist"]["episode_id"])
    assert "audit_selection_reasons" in exports["audit_shortlist"].columns
