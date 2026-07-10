from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_PATH = PROJECT_ROOT / "notebooks" / "explore_eve_results.ipynb"
DATA_DIR = PROJECT_ROOT / "data" / "processed"
DEFAULT_SITE_ID = "site_004"


def md(source: str) -> dict:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": [line + "\n" for line in source.strip().splitlines()],
    }


def code(source: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [line + "\n" for line in source.strip().splitlines()],
    }


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Génère le notebook principal d'analyse des résultats EVE V0.2."
    )
    parser.add_argument(
        "--site",
        default=DEFAULT_SITE_ID,
        help="Identifiant du site à analyser dans le notebook.",
    )
    return parser.parse_args()


def main(site_id: str | None = None) -> None:
    if site_id is None:
        site_id = parse_arguments().site

    decision_points = pd.read_csv(DATA_DIR / f"eve_decision_points_{site_id}.csv")
    episodes = pd.read_csv(DATA_DIR / f"eve_episodes_{site_id}.csv")
    manual_audit = pd.read_csv(DATA_DIR / f"eve_manual_audit_{site_id}.csv")
    summary = json.loads(
        (DATA_DIR / f"eve_summary_{site_id}.json").read_text(encoding="utf-8")
    )
    type_distribution = episodes["episode_type_primary"].value_counts().to_dict()

    config_cell = """
from pathlib import Path
import json

import matplotlib.pyplot as plt
import pandas as pd

PROJECT_ROOT = Path("..").resolve()
DATA_DIR = PROJECT_ROOT / "data" / "processed"
SITE_ID = "__SITE_ID__"

paths = {
    "decision_points": DATA_DIR / f"eve_decision_points_{SITE_ID}.csv",
    "episodes": DATA_DIR / f"eve_episodes_{SITE_ID}.csv",
    "summary": DATA_DIR / f"eve_summary_{SITE_ID}.json",
    "manual_audit": DATA_DIR / f"eve_manual_audit_{SITE_ID}.csv",
}

for name, path in paths.items():
    print(name, path, "OK" if path.exists() else "MISSING")
""".replace("__SITE_ID__", site_id)

    notebook = {
        "cells": [
            md(
                f"""
# EVE — résultats officiels V0.2 — {site_id}

Ce notebook est le notebook principal simplifié du projet.

Il lit uniquement les 4 sorties officielles :

- `eve_decision_points_{site_id}.csv`
- `eve_episodes_{site_id}.csv`
- `eve_summary_{site_id}.json`
- `eve_manual_audit_{site_id}.csv`

La V0.1 est désormais une étape interne du pipeline ; la sortie officielle analysée ici est la V0.2.
"""
            ),
            md(
                f"""
## Résumé actuel

- Points de décision : {len(decision_points)}
- Épisodes : {len(episodes)}
- Épisodes audit manuel : {len(manual_audit)}
- Typologie : {type_distribution}
- Version officielle : {summary.get("eve_pipeline_version")}
"""
            ),
            code(config_cell),
            code(
                """
decision_points = pd.read_csv(paths["decision_points"], parse_dates=["acquisition_date"])
episodes = pd.read_csv(paths["episodes"], parse_dates=["episode_start_date", "episode_end_date", "peak_date"])
manual_audit = pd.read_csv(paths["manual_audit"], parse_dates=["episode_start_date", "episode_end_date", "peak_date"])
summary = json.loads(paths["summary"].read_text(encoding="utf-8"))

print("Decision points:", decision_points.shape)
print("Episodes:", episodes.shape)
print("Manual audit:", manual_audit.shape)
summary
"""
            ),
            md("## 1. Funnel officiel"),
            code(
                """
kpis = pd.DataFrame(
    [
        ["eligible_points", summary["eligible_points"]],
        ["candidate_dates", summary["candidate_dates"]],
        ["episodes", summary["episodes"]],
        ["manual_audit_episodes", summary["manual_audit_episodes"]],
        ["poor_quality_abstentions", summary["poor_quality_abstentions"]],
        ["wait_decisions", summary["wait_decisions"]],
        ["potential_observations_avoided_proxy", summary["potential_observations_avoided_vs_all_eligible_review_proxy"]],
    ],
    columns=["metric", "value"],
)
display(kpis)

fig, ax = plt.subplots(figsize=(9, 4))
ax.bar(kpis["metric"], kpis["value"])
ax.set_title("Funnel EVE officiel")
ax.tick_params(axis="x", rotation=35)
ax.set_ylabel("count")
plt.tight_layout()
plt.show()
"""
            ),
            md("## 2. Typologie des épisodes"),
            code(
                """
type_counts = episodes["episode_type_primary"].value_counts()
display(type_counts.to_frame("episodes"))

fig, ax = plt.subplots(figsize=(9, 4))
type_counts.plot(kind="bar", ax=ax)
ax.set_title("Typologie primaire des épisodes")
ax.set_ylabel("episodes")
ax.tick_params(axis="x", rotation=30)
plt.tight_layout()
plt.show()

display(
    episodes[
        [
            "episode_id",
            "episode_start_date",
            "episode_end_date",
            "proposed_action",
            "episode_type_primary",
            "episode_type_flags",
            "episode_type_confidence",
            "lowest_ndmi_percentile",
            "dominant_hydro_context",
            "highest_vpd_state",
            "episode_phenology_stage_main",
        ]
    ].sort_values(["episode_type_primary", "lowest_ndmi_percentile"])
)
"""
            ),
            md("## 3. Réponse végétale vs pression climatique"),
            code(
                """
fig, ax = plt.subplots(figsize=(8, 6))
for label, group in episodes.groupby("episode_type_primary"):
    ax.scatter(
        group["episode_vegetation_anomaly_score_max"],
        group["episode_climatic_pressure_score_max"],
        s=40 + group["decision_point_count"] * 10,
        alpha=0.75,
        label=label,
    )
ax.axvline(0.6, linestyle="--", color="grey", linewidth=1)
ax.axhline(0.6, linestyle="--", color="grey", linewidth=1)
ax.set_xlabel("score anomalie végétale")
ax.set_ylabel("score pression climatique")
ax.set_title("Réponse végétale vs pression climatique")
ax.legend(fontsize=8)
plt.tight_layout()
plt.show()
"""
            ),
            md("## 4. Action opérationnelle vs typologie"),
            code(
                """
crosstab = pd.crosstab(episodes["proposed_action"], episodes["episode_type_primary"])
display(crosstab)

fig, ax = plt.subplots(figsize=(9, 4))
crosstab.plot(kind="bar", stacked=True, ax=ax)
ax.set_title("Actions recommandées et typologie")
ax.set_ylabel("episodes")
ax.tick_params(axis="x", rotation=0)
plt.tight_layout()
plt.show()
"""
            ),
            md("## 5. Phénologie et prudence d'interprétation"),
            code(
                """
phenology_table = pd.crosstab(
    episodes["episode_phenology_stage_main"],
    episodes["episode_phenology_interpretation_risk_max"],
)
display(phenology_table)

fig, ax = plt.subplots(figsize=(8, 4))
phenology_table.plot(kind="bar", stacked=True, ax=ax)
ax.set_title("Phénologie et risque d'interprétation")
ax.set_ylabel("episodes")
ax.tick_params(axis="x", rotation=30)
plt.tight_layout()
plt.show()
"""
            ),
            md("## 6. EVE vs règle simple combinée"),
            code(
                """
relation_counts = episodes["eve_vs_simple_baseline_relation"].value_counts()
display(relation_counts.to_frame("episodes"))

fig, ax = plt.subplots(figsize=(7, 4))
relation_counts.plot(kind="bar", ax=ax)
ax.set_title("Relation EVE vs règle simple")
ax.set_ylabel("episodes")
ax.tick_params(axis="x", rotation=0)
plt.tight_layout()
plt.show()
"""
            ),
            md("## 7. Coûts physiques et dépendance numérique"),
            code(
                """
cost_summary = (
    episodes.groupby(["proposed_action", "digital_dependency_level"])
    .agg(
        episodes=("episode_id", "count"),
        time_low=("physical_time_min_low", "sum"),
        time_high=("physical_time_min_high", "sum"),
        electricity_low=("physical_electricity_kwh_low", "sum"),
        electricity_high=("physical_electricity_kwh_high", "sum"),
    )
    .reset_index()
)
display(cost_summary)
"""
            ),
            md("## 8. Short-list d'audit manuel"),
            code(
                """
audit_cols = [
    "episode_id",
    "episode_start_date",
    "episode_end_date",
    "duration_days",
    "proposed_action",
    "episode_type_primary",
    "manual_audit_reasons",
    "lowest_ndmi_percentile",
    "episode_consistency_score_max",
    "dominant_hydro_context",
    "highest_vpd_state",
    "episode_phenology_stage_main",
]
display(manual_audit[audit_cols])
"""
            ),
            md("## 9. Inspection d'un épisode"),
            code(
                """
def show_episode(episode_id: str):
    episode = episodes[episodes["episode_id"].eq(episode_id)]
    points = decision_points[decision_points["episode_id"].eq(episode_id)].copy()
    display(episode.T)
    display(
        points[
            [
                "acquisition_date",
                "ndmi_value",
                "ndmi_percentile",
                "vegetation_anomaly_score",
                "climatic_pressure_score",
                "consistency_score",
                "hydro_context",
                "vpd_state",
                "phenology_stage",
                "satellite_quality",
                "recommended_information_action",
                "simple_combined_alert",
                "degraded_mode_recommendation",
                "justification_text",
            ]
        ]
    )

if not episodes.empty:
    show_episode(episodes.iloc[0]["episode_id"])
"""
            ),
            md(
                """
## 10. Lecture scientifique

Ce notebook doit servir à analyser les sorties officielles, pas à produire des fichiers.

Interprétation prudente :

- `hydroclimatic_candidate` = hypothèse cohérente végétation + climat.
- `unexplained_vegetation_anomaly` = anomalie végétale à expliquer, pas diagnostic.
- `phenology_transition` = signal à forte prudence phénologique.
- Les métriques d'évitement sont des proxys, pas des gains réels mesurés.
"""
            ),
        ],
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {
                "name": "python",
                "pygments_lexer": "ipython3",
            },
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }

    NOTEBOOK_PATH.parent.mkdir(parents=True, exist_ok=True)
    NOTEBOOK_PATH.write_text(
        json.dumps(notebook, ensure_ascii=False, indent=1),
        encoding="utf-8",
    )
    print(NOTEBOOK_PATH)


if __name__ == "__main__":
    main()
