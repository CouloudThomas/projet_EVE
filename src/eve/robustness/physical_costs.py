from __future__ import annotations

from typing import Any


PHYSICAL_COST_COLUMNS = [
    "physical_time_min_low",
    "physical_time_min_high",
    "physical_travel_km_low",
    "physical_travel_km_high",
    "physical_electricity_kwh_low",
    "physical_electricity_kwh_high",
    "estimated_kgco2e_low",
    "estimated_kgco2e_high",
    "material_mobilized",
    "digital_dependency_level",
    "physical_cost_model_version",
    "physical_cost_status",
    "cost_confidence",
]


def physical_cost_for_action(action: str, config: dict[str, Any]) -> dict[str, Any]:
    actions = config["actions"]
    if action not in actions:
        action = "none"
    source = actions[action]
    result = {
        "physical_time_min_low": source.get("physical_time_min_low"),
        "physical_time_min_high": source.get("physical_time_min_high"),
        "physical_travel_km_low": source.get("physical_travel_km_low"),
        "physical_travel_km_high": source.get("physical_travel_km_high"),
        "physical_electricity_kwh_low": source.get("physical_electricity_kwh_low"),
        "physical_electricity_kwh_high": source.get("physical_electricity_kwh_high"),
        "estimated_kgco2e_low": source.get("estimated_kgco2e_low"),
        "estimated_kgco2e_high": source.get("estimated_kgco2e_high"),
        "material_mobilized": source.get("material_mobilized"),
        "digital_dependency_level": source.get("digital_dependency_level"),
        "physical_cost_model_version": config["cost_model_version"],
        "physical_cost_status": config["cost_status"],
        "cost_confidence": source.get("cost_confidence", "unknown"),
    }
    return result
