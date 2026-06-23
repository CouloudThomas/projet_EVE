from __future__ import annotations

import json

import pandas as pd

from eve.build_eve_outputs import _augment_summary, write_official_outputs


def test_write_official_outputs_uses_only_four_canonical_files(tmp_path) -> None:
    site_id = "site_test"
    decision_points = pd.DataFrame(
        {
            "decision_point_id": ["site_test_20220801"],
            "acquisition_date": ["2022-08-01"],
        }
    )
    episodes = pd.DataFrame({"episode_id": ["site_test_episode_20220801_01"]})
    manual_audit = episodes.copy()
    summary = {"eve_pipeline_version": "v0.2_official"}

    paths = write_official_outputs(
        decision_points,
        episodes,
        summary,
        manual_audit,
        processed_dir=tmp_path,
        site_id=site_id,
    )

    assert set(paths) == {"decision_points", "episodes", "summary", "manual_audit"}
    assert sorted(path.name for path in tmp_path.iterdir()) == [
        "eve_decision_points_site_test.csv",
        "eve_episodes_site_test.csv",
        "eve_manual_audit_site_test.csv",
        "eve_summary_site_test.json",
    ]
    saved_summary = json.loads(paths["summary"].read_text(encoding="utf-8"))
    assert saved_summary["eve_pipeline_version"] == "v0.2_official"


def test_summary_declares_v0_2_as_official_layer(tmp_path) -> None:
    paths = {
        "decision_points": tmp_path / "eve_decision_points_site_test.csv",
        "episodes": tmp_path / "eve_episodes_site_test.csv",
        "summary": tmp_path / "eve_summary_site_test.json",
        "manual_audit": tmp_path / "eve_manual_audit_site_test.csv",
    }

    summary = _augment_summary(
        {"episodes": 1},
        site_id="site_test",
        generated_at="2026-06-23T12:00:00+00:00",
        evaluation_mode="retrospective_loyo",
        baseline_artifact={"baseline_version": "baseline_test"},
        v0_1_rules_config={"rules_version": "v0.1"},
        v0_2_rules_config={"rules_version": "v0.2"},
        official_paths=paths,
    )

    assert summary["eve_pipeline_version"] == "v0.2_official"
    assert summary["source_decision_rules_version"] == "v0.1"
    assert summary["robustness_rules_version"] == "v0.2"
    assert "pas des diagnostics" in summary["interpretation_warning"]
