from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"


def _count(series: pd.Series) -> dict[str, int]:
    return {
        str(key): int(value)
        for key, value in series.fillna("missing").astype(str).value_counts().items()
    }


def _safe_value(value: Any) -> Any:
    if pd.isna(value):
        return None
    if isinstance(value, pd.Timestamp):
        return value.date().isoformat()
    if hasattr(value, "item"):
        return value.item()
    return value


def _records(frame: pd.DataFrame, columns: list[str], limit: int = 10) -> list[dict[str, Any]]:
    available = [column for column in columns if column in frame.columns]
    records = frame[available].head(limit).to_dict(orient="records")
    return [
        {key: _safe_value(value) for key, value in record.items()}
        for record in records
    ]


def _sum_costs(episodes: pd.DataFrame) -> dict[str, float | int]:
    if episodes.empty:
        return {
            "episode_count": 0,
            "time_min_low": 0,
            "time_min_high": 0,
            "cost_eur_low": 0.0,
            "cost_eur_high": 0.0,
        }

    return {
        "episode_count": int(len(episodes)),
        "time_min_low": int(episodes["estimated_time_min_low"].sum()),
        "time_min_high": int(episodes["estimated_time_min_high"].sum()),
        "cost_eur_low": float(episodes["estimated_cost_eur_low"].sum()),
        "cost_eur_high": float(episodes["estimated_cost_eur_high"].sum()),
    }


def build_monitoring_report(
    decision_points: pd.DataFrame,
    episodes: pd.DataFrame,
    *,
    site_id: str,
    generated_at: str | None = None,
) -> dict[str, Any]:
    """Construit un rapport minimal de santé scientifique EVE.

    Ce rapport ne valide pas la vérité terrain. Il vérifie que les sorties sont
    cohérentes, lisibles et surveillables après chaque run.
    """

    generated_at = generated_at or datetime.now(timezone.utc).replace(
        microsecond=0
    ).isoformat()
    decisions = decision_points.copy()
    decisions["acquisition_date"] = pd.to_datetime(decisions["acquisition_date"])
    episode_table = episodes.copy()
    if not episode_table.empty:
        episode_table["episode_start_date"] = pd.to_datetime(
            episode_table["episode_start_date"]
        )
        episode_table["episode_end_date"] = pd.to_datetime(
            episode_table["episode_end_date"]
        )

    candidates = decisions[decisions["is_candidate"] == True]  # noqa: E712
    in_scope = decisions[decisions["in_scope_season"] == True]  # noqa: E712
    eligible = decisions[decisions["eligible_for_analysis"] == True]  # noqa: E712
    priority = episode_table[
        episode_table["proposed_action"].eq("priority_field_check")
    ]

    candidate_episode_ids = set(candidates["episode_id"].dropna().astype(str))
    episode_ids = set(episode_table["episode_id"].dropna().astype(str))
    invariants = {
        "duplicate_decision_point_ids": int(
            decisions["decision_point_id"].duplicated().sum()
        ),
        "candidate_without_episode": int(
            (
                (decisions["is_candidate"] == True)  # noqa: E712
                & decisions["episode_id"].isna()
            ).sum()
        ),
        "non_candidate_with_episode": int(
            (
                (decisions["is_candidate"] == False)  # noqa: E712
                & decisions["episode_id"].notna()
            ).sum()
        ),
        "poor_satellite_candidate": int(
            (
                (decisions["is_candidate"] == True)  # noqa: E712
                & decisions["satellite_quality"].eq("poor")
            ).sum()
        ),
        "ndmi_percentile_out_of_bounds": int(
            (
                decisions["ndmi_percentile"].notna()
                & ~decisions["ndmi_percentile"].between(0, 100)
            ).sum()
        ),
        "episode_count_mismatch": int(len(candidate_episode_ids ^ episode_ids)),
    }
    critical_failures = {
        key: value for key, value in invariants.items() if value != 0
    }

    warnings: list[str] = []
    poor_share = (
        float((in_scope["satellite_quality"] == "poor").mean())
        if not in_scope.empty
        else 0.0
    )
    if poor_share >= 0.40:
        warnings.append(
            "Plus de 40 % des observations en saison sont de qualité satellite pauvre."
        )
    if not episode_table.empty and int(episode_table["duration_days"].max()) >= 45:
        warnings.append(
            "Au moins un épisode dure 45 jours ou plus : vérifier un éventuel chaînage."
        )
    yearly_funnel = (
        decisions.assign(year=decisions["acquisition_date"].dt.year)
        .groupby("year")
        .agg(
            decision_points=("decision_point_id", "count"),
            in_scope=("in_scope_season", "sum"),
            eligible=("eligible_for_analysis", "sum"),
            candidates=("is_candidate", "sum"),
        )
    )
    yearly_funnel["candidate_rate_eligible"] = (
        yearly_funnel["candidates"] / yearly_funnel["eligible"]
    )

    top_longest = (
        episode_table.sort_values(
            ["duration_days", "decision_point_count"],
            ascending=False,
        )
        if not episode_table.empty
        else episode_table
    )
    most_extreme = (
        episode_table.sort_values(
            ["lowest_ndmi_percentile", "worst_ndmi_anomaly"],
            ascending=True,
        )
        if not episode_table.empty
        else episode_table
    )

    report = {
        "site_id": site_id,
        "generated_at": generated_at,
        "status": "fail" if critical_failures else ("warn" if warnings else "ok"),
        "warnings": warnings,
        "critical_failures": critical_failures,
        "summary": {
            "decision_points": int(len(decisions)),
            "date_min": decisions["acquisition_date"].min().date().isoformat(),
            "date_max": decisions["acquisition_date"].max().date().isoformat(),
            "in_scope": int(len(in_scope)),
            "eligible": int(len(eligible)),
            "candidates": int(len(candidates)),
            "episodes": int(len(episode_table)),
            "candidate_rate_among_eligible": (
                float(len(candidates) / len(eligible)) if len(eligible) else None
            ),
            "poor_satellite_share_in_scope": poor_share,
        },
        "counts": {
            "eligibility_reason": _count(decisions["eligibility_reason"]),
            "rule_id": _count(decisions["rule_id"]),
            "decision_action": _count(decisions["recommended_information_action"]),
            "decision_urgency": _count(decisions["urgency"]),
            "episode_action": (
                _count(episode_table["proposed_action"])
                if not episode_table.empty
                else {}
            ),
            "episode_hydro_context": (
                _count(episode_table["dominant_hydro_context"])
                if not episode_table.empty
                else {}
            ),
            "episode_vpd_state": (
                _count(episode_table["highest_vpd_state"])
                if not episode_table.empty
                else {}
            ),
            "episode_persistence": (
                _count(episode_table["persistence_state"])
                if not episode_table.empty
                else {}
            ),
        },
        "yearly_funnel": [
            {key: _safe_value(value) for key, value in record.items()}
            for record in yearly_funnel.reset_index().to_dict(orient="records")
        ],
        "episode_costs": {
            "all_episodes": _sum_costs(episode_table),
            "priority_field_check": _sum_costs(priority),
        },
        "top_longest_episodes": _records(
            top_longest,
            [
                "episode_id",
                "episode_start_date",
                "episode_end_date",
                "duration_days",
                "decision_point_count",
                "proposed_action",
                "dominant_hydro_context",
                "highest_vpd_state",
                "lowest_ndmi_percentile",
            ],
        ),
        "most_extreme_ndmi_episodes": _records(
            most_extreme,
            [
                "episode_id",
                "episode_start_date",
                "episode_end_date",
                "duration_days",
                "decision_point_count",
                "proposed_action",
                "dominant_hydro_context",
                "highest_vpd_state",
                "lowest_ndmi_percentile",
                "worst_ndmi_anomaly",
            ],
        ),
        "invariants": invariants,
    }
    return report


def _format_markdown_value(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def _markdown_table(headers: list[str], rows: list[list[Any]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(_format_markdown_value(value) for value in row)
            + " |"
        )
    return "\n".join(lines)


def monitoring_markdown(report: dict[str, Any]) -> str:
    summary = report["summary"]
    lines = [
        f"# Monitoring scientifique EVE — {report['site_id']}",
        "",
        f"- Généré : {report['generated_at']}",
        f"- Statut : `{report['status']}`",
        "",
        "## Résumé",
        "",
        _markdown_table(
            ["Indicateur", "Valeur"],
            [
                ["Points de décision", summary["decision_points"]],
                ["Période", f"{summary['date_min']} → {summary['date_max']}"],
                ["Points en saison", summary["in_scope"]],
                ["Points éligibles", summary["eligible"]],
                ["Candidats", summary["candidates"]],
                ["Épisodes", summary["episodes"]],
                [
                    "Taux candidat / éligible",
                    (
                        f"{summary['candidate_rate_among_eligible']:.3f}"
                        if summary["candidate_rate_among_eligible"] is not None
                        else "n/a"
                    ),
                ],
                [
                    "Part observations pauvres en saison",
                    f"{summary['poor_satellite_share_in_scope']:.3f}",
                ],
            ],
        ),
        "",
        "## Alertes",
        "",
    ]
    if report["warnings"]:
        lines.extend(f"- {warning}" for warning in report["warnings"])
    else:
        lines.append("- Aucune alerte non bloquante.")
    lines.extend(["", "## Invariants", ""])
    lines.append(
        _markdown_table(
            ["Contrôle", "Valeur"],
            [[key, value] for key, value in report["invariants"].items()],
        )
    )
    lines.extend(["", "## Répartition des règles", ""])
    lines.append(
        _markdown_table(
            ["Règle", "Nombre"],
            [[key, value] for key, value in report["counts"]["rule_id"].items()],
        )
    )
    lines.extend(["", "## Actions épisodes", ""])
    lines.append(
        _markdown_table(
            ["Action", "Nombre"],
            [
                [key, value]
                for key, value in report["counts"]["episode_action"].items()
            ],
        )
    )
    lines.extend(["", "## Épisodes les plus longs", ""])
    lines.append(
        _markdown_table(
            [
                "Épisode",
                "Début",
                "Fin",
                "Durée",
                "Points",
                "Action",
                "Hydro",
                "VPD",
                "NDMI pct min",
            ],
            [
                [
                    row.get("episode_id"),
                    row.get("episode_start_date"),
                    row.get("episode_end_date"),
                    row.get("duration_days"),
                    row.get("decision_point_count"),
                    row.get("proposed_action"),
                    row.get("dominant_hydro_context"),
                    row.get("highest_vpd_state"),
                    row.get("lowest_ndmi_percentile"),
                ]
                for row in report["top_longest_episodes"]
            ],
        )
    )
    lines.extend(["", "## Coûts théoriques", ""])
    lines.append(
        _markdown_table(
            ["Périmètre", "Épisodes", "Temps min", "Temps max", "Coût min", "Coût max"],
            [
                [
                    "Tous épisodes",
                    report["episode_costs"]["all_episodes"]["episode_count"],
                    report["episode_costs"]["all_episodes"]["time_min_low"],
                    report["episode_costs"]["all_episodes"]["time_min_high"],
                    report["episode_costs"]["all_episodes"]["cost_eur_low"],
                    report["episode_costs"]["all_episodes"]["cost_eur_high"],
                ],
                [
                    "Priorité terrain",
                    report["episode_costs"]["priority_field_check"]["episode_count"],
                    report["episode_costs"]["priority_field_check"]["time_min_low"],
                    report["episode_costs"]["priority_field_check"]["time_min_high"],
                    report["episode_costs"]["priority_field_check"]["cost_eur_low"],
                    report["episode_costs"]["priority_field_check"]["cost_eur_high"],
                ],
            ],
        )
    )
    lines.append("")
    return "\n".join(lines)


def write_monitoring_artifacts(
    report: dict[str, Any],
    *,
    output_dir: Path,
    site_id: str,
) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_file = output_dir / f"eve_scientific_monitoring_{site_id}.json"
    markdown_file = output_dir / f"eve_scientific_monitoring_{site_id}.md"
    with json_file.open("w", encoding="utf-8") as file:
        json.dump(report, file, ensure_ascii=False, indent=2)
        file.write("\n")
    markdown_file.write_text(monitoring_markdown(report), encoding="utf-8")
    return json_file, markdown_file


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Construit le monitoring scientifique minimal EVE."
    )
    parser.add_argument("--site", default="site_004")
    parser.add_argument("--processed-dir", type=Path, default=DEFAULT_PROCESSED_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_arguments()
    decision_file = args.processed_dir / f"eve_decision_points_{args.site}.csv"
    episode_file = args.processed_dir / f"eve_episodes_{args.site}.csv"
    decision_points = pd.read_csv(decision_file)
    episodes = pd.read_csv(episode_file)
    report = build_monitoring_report(
        decision_points,
        episodes,
        site_id=args.site,
    )
    json_file, markdown_file = write_monitoring_artifacts(
        report,
        output_dir=args.processed_dir,
        site_id=args.site,
    )
    print(f"Monitoring JSON : {json_file}")
    print(f"Monitoring Markdown : {markdown_file}")
    print(f"Statut : {report['status']}")


if __name__ == "__main__":
    main()
