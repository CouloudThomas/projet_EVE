from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from eve.episodes.baselines import load_structured_config
from eve.episodes.build_episode_store import (
    DEFAULT_COSTS_CONFIG as DEFAULT_V0_1_COSTS_CONFIG,
    DEFAULT_RULES_CONFIG as DEFAULT_V0_1_RULES_CONFIG,
    build_decision_points,
    load_sources,
)
from eve.episodes.grouping import build_episode_table
from eve.robustness.build_v0_2_episode_store import (
    DEFAULT_COSTS_CONFIG as DEFAULT_V0_2_COSTS_CONFIG,
    DEFAULT_RULES_CONFIG as DEFAULT_V0_2_RULES_CONFIG,
    DEFAULT_SIMPLE_BASELINES_CONFIG,
    build_manual_audit_table,
    enrich_decision_points,
    enrich_episodes,
)
from eve.robustness.robustness_metrics import build_robustness_summary


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Construit les 4 sorties officielles EVE. "
            "La V0.1 est calculée en mémoire, la sortie officielle est V0.2."
        )
    )
    parser.add_argument("--site", default="site_004")
    parser.add_argument("--processed-dir", type=Path, default=DEFAULT_PROCESSED_DIR)
    parser.add_argument("--v0-1-rules-config", type=Path, default=DEFAULT_V0_1_RULES_CONFIG)
    parser.add_argument("--v0-1-costs-config", type=Path, default=DEFAULT_V0_1_COSTS_CONFIG)
    parser.add_argument("--v0-2-rules-config", type=Path, default=DEFAULT_V0_2_RULES_CONFIG)
    parser.add_argument("--v0-2-costs-config", type=Path, default=DEFAULT_V0_2_COSTS_CONFIG)
    parser.add_argument(
        "--simple-baselines-config",
        type=Path,
        default=DEFAULT_SIMPLE_BASELINES_CONFIG,
    )
    parser.add_argument(
        "--evaluation-mode",
        choices=("retrospective_loyo", "prospective"),
        default=None,
    )
    return parser.parse_args()


def _official_paths(processed_dir: Path, site_id: str) -> dict[str, Path]:
    return {
        "decision_points": processed_dir / f"eve_decision_points_{site_id}.csv",
        "episodes": processed_dir / f"eve_episodes_{site_id}.csv",
        "summary": processed_dir / f"eve_summary_{site_id}.json",
        "manual_audit": processed_dir / f"eve_manual_audit_{site_id}.csv",
    }


def _augment_summary(
    summary: dict[str, Any],
    *,
    site_id: str,
    generated_at: str,
    evaluation_mode: str,
    baseline_artifact: dict[str, Any],
    v0_1_rules_config: dict[str, Any],
    v0_2_rules_config: dict[str, Any],
    official_paths: dict[str, Path],
) -> dict[str, Any]:
    def display_path(path: Path) -> str:
        try:
            return str(path.relative_to(PROJECT_ROOT))
        except ValueError:
            return str(path)

    result = dict(summary)
    result.update(
        {
            "site_id": site_id,
            "eve_pipeline_version": "v0.2_official",
            "generated_at": generated_at,
            "evaluation_mode": evaluation_mode,
            "source_decision_rules_version": v0_1_rules_config["rules_version"],
            "robustness_rules_version": v0_2_rules_config["rules_version"],
            "baseline_version": baseline_artifact.get("baseline_version"),
            "official_outputs": {
                name: display_path(path)
                for name, path in official_paths.items()
            },
            "interpretation_warning": (
                "Les épisodes V0.2 sont des hypothèses d'observation et de "
                "qualification, pas des diagnostics causalement validés."
            ),
        }
    )
    return result


def write_official_outputs(
    decision_points: pd.DataFrame,
    episodes: pd.DataFrame,
    summary: dict[str, Any],
    manual_audit: pd.DataFrame,
    *,
    processed_dir: Path,
    site_id: str,
) -> dict[str, Path]:
    processed_dir.mkdir(parents=True, exist_ok=True)
    paths = _official_paths(processed_dir, site_id)

    decision_output = decision_points.copy()
    decision_output["acquisition_date"] = pd.to_datetime(
        decision_output["acquisition_date"]
    ).dt.date
    decision_output.to_csv(paths["decision_points"], index=False, encoding="utf-8")
    episodes.to_csv(paths["episodes"], index=False, encoding="utf-8")
    manual_audit.to_csv(paths["manual_audit"], index=False, encoding="utf-8")
    with paths["summary"].open("w", encoding="utf-8") as file:
        json.dump(summary, file, ensure_ascii=False, indent=2)
        file.write("\n")
    return paths


def run(
    *,
    site_id: str,
    processed_dir: Path,
    v0_1_rules_config_file: Path,
    v0_1_costs_config_file: Path,
    v0_2_rules_config_file: Path,
    v0_2_costs_config_file: Path,
    simple_baselines_config_file: Path,
    evaluation_mode: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any], pd.DataFrame]:
    generated_at = datetime.now(timezone.utc).replace(microsecond=0).isoformat()

    v0_1_rules_config = load_structured_config(v0_1_rules_config_file)
    v0_1_costs_config = load_structured_config(v0_1_costs_config_file)
    v0_2_rules_config = load_structured_config(v0_2_rules_config_file)
    v0_2_costs_config = load_structured_config(v0_2_costs_config_file)
    simple_config = load_structured_config(simple_baselines_config_file)
    selected_mode = evaluation_mode or v0_1_rules_config["evaluation_mode"]

    vegetation_file = processed_dir / f"vegetation_weather_{site_id}.csv"
    weather_file = processed_dir / f"weather_daily_{site_id}.csv"
    vegetation, weather = load_sources(
        vegetation_file,
        weather_file,
        v0_1_rules_config,
        site_id,
    )
    decision_points_v0_1, baseline_artifact = build_decision_points(
        vegetation,
        weather,
        site_id=site_id,
        rules_config=v0_1_rules_config,
        costs_config=v0_1_costs_config,
        evaluation_mode=selected_mode,
        generated_at=generated_at,
    )
    episodes_v0_1 = build_episode_table(
        decision_points_v0_1,
        cost_model_version=v0_1_costs_config["cost_model_version"],
        cost_status=v0_1_costs_config["cost_status"],
    )

    decision_points_v0_2 = enrich_decision_points(
        decision_points_v0_1,
        rules_config=v0_2_rules_config,
        simple_config=simple_config,
        generated_at=generated_at,
    )
    episodes_v0_2 = enrich_episodes(
        episodes_v0_1,
        decision_points_v0_2,
        rules_config=v0_2_rules_config,
        costs_config=v0_2_costs_config,
        generated_at=generated_at,
    )
    manual_audit = build_manual_audit_table(episodes_v0_2)
    robustness_summary = build_robustness_summary(
        decision_points_v0_2,
        episodes_v0_2,
        manual_audit,
        site_id=site_id,
        generated_at=generated_at,
    )
    official_paths = _official_paths(processed_dir, site_id)
    summary = _augment_summary(
        robustness_summary,
        site_id=site_id,
        generated_at=generated_at,
        evaluation_mode=selected_mode,
        baseline_artifact=baseline_artifact,
        v0_1_rules_config=v0_1_rules_config,
        v0_2_rules_config=v0_2_rules_config,
        official_paths=official_paths,
    )
    write_official_outputs(
        decision_points_v0_2,
        episodes_v0_2,
        summary,
        manual_audit,
        processed_dir=processed_dir,
        site_id=site_id,
    )
    return decision_points_v0_2, episodes_v0_2, summary, manual_audit


def main() -> None:
    args = parse_arguments()
    decision_points, episodes, summary, manual_audit = run(
        site_id=args.site,
        processed_dir=args.processed_dir,
        v0_1_rules_config_file=args.v0_1_rules_config,
        v0_1_costs_config_file=args.v0_1_costs_config,
        v0_2_rules_config_file=args.v0_2_rules_config,
        v0_2_costs_config_file=args.v0_2_costs_config,
        simple_baselines_config_file=args.simple_baselines_config,
        evaluation_mode=args.evaluation_mode,
    )
    print("EVE sorties officielles générées.")
    print(f"Site : {args.site}")
    print(f"Version officielle : {summary['eve_pipeline_version']}")
    print(f"Points de décision : {len(decision_points)}")
    print(f"Épisodes : {len(episodes)}")
    print(f"Épisodes audit manuel : {len(manual_audit)}")
    print("Typologie :")
    for label, count in episodes["episode_type_primary"].value_counts().items():
        print(f"- {label}: {count}")
    print("Sorties :")
    for name, path in summary["official_outputs"].items():
        print(f"- {name}: {path}")


if __name__ == "__main__":
    main()
