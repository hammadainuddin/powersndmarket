"""
config_writer.py
----------------
Translate Streamlit dashboard state (Python dicts) into:
  1. Calliope YAML override files (saved to model/overrides/)
  2. Parameter dicts consumed directly by build_model.py / myopic_chain.py

Also provides helpers for reading existing scenario configs.
"""

from pathlib import Path
from typing import Optional

import yaml

PROJECT_ROOT = Path(__file__).parent.parent.parent
OVERRIDES_DIR = PROJECT_ROOT / "model" / "overrides"
OVERRIDES_DIR.mkdir(parents=True, exist_ok=True)


def demand_growth_from_ui(peninsular: float, sabah: float, sarawak: float) -> dict:
    """Convert dashboard sliders (% CAGR) to decimal growth rates."""
    return {
        "peninsular": peninsular / 100.0,
        "sabah": sabah / 100.0,
        "sarawak": sarawak / 100.0,
    }


def tech_costs_from_ui(cost_table: dict) -> dict:
    """
    Convert dashboard cost table to Calliope-compatible override dict.

    Args:
        cost_table: {
            tech_name: {
                'capex_usd_kw': float,
                'opex_fixed_usd_kw_yr': float,
                'opex_var_usd_mwh': float,
            }
        }

    Returns:
        {tech_name: {'energy_cap': ..., 'om_annual': ..., 'om_con': ...}}
    """
    result = {}
    for tech, costs in cost_table.items():
        entry = {}
        if "capex_usd_kw" in costs:
            entry["energy_cap"] = costs["capex_usd_kw"] * 1000  # Convert $/kW -> $/MW
        if "opex_fixed_usd_kw_yr" in costs:
            entry["om_annual"] = costs["opex_fixed_usd_kw_yr"] * 1000  # $/kW/yr -> $/MW/yr
        if "opex_var_usd_mwh" in costs:
            entry["om_con"] = costs["opex_var_usd_mwh"] / 1000  # $/MWh -> $/kWh (Calliope units)
        if entry:
            result[tech] = entry
    return result


def carbon_price_from_ui(trajectory: str, custom_value: Optional[float] = None) -> Optional[float]:
    """Convert dashboard carbon price selection to USD/tCO2 for 2025 anchor."""
    presets = {
        "none": None,
        "low": 10.0,
        "medium": 50.0,
        "high": 100.0,
    }
    if trajectory == "custom":
        return custom_value
    return presets.get(trajectory)


def re_targets_from_ui(target_table: dict) -> dict:
    """
    Convert dashboard RE target table to fraction dict.

    Args:
        target_table: {2025: 31, 2030: 35, 2035: 40, 2040: 52, 2045: 62, 2050: 70}

    Returns:
        {2025: 0.31, 2030: 0.35, ...}
    """
    return {int(yr): pct / 100.0 for yr, pct in target_table.items()}


def save_custom_scenario(
    name: str,
    base_scenario: str,
    demand_growth: dict,
    tech_costs: dict,
    carbon_price_trajectory: str,
    re_targets: dict,
    toggles: dict,
) -> Path:
    """
    Save a custom scenario as a YAML override file.

    Args:
        name: scenario name (will be slugified for filename)
        base_scenario: 'baseline', 'netr_target', or 'accelerated_re'
        demand_growth: {region: decimal_cagr}
        tech_costs: Calliope-format cost overrides
        carbon_price_trajectory: 'none'/'low'/'medium'/'high'
        re_targets: {year: fraction}
        toggles: {
            'early_coal_retirement': bool,
            'enable_hvdc_2030': bool,
            'hydrogen_peakers': bool,
            'retire_diesel_2030': bool,
        }

    Returns:
        Path to saved YAML file
    """
    slug = name.lower().replace(" ", "_").replace("-", "_")
    out_path = OVERRIDES_DIR / f"{slug}.yaml"

    config = {
        "overrides": {
            slug: {
                "model.name": f"Malaysia {name}",
                "_base_scenario": base_scenario,
                "_demand_growth": demand_growth,
                "_tech_costs": tech_costs,
                "_carbon_price_trajectory": carbon_price_trajectory,
                "_re_targets": {str(k): v for k, v in re_targets.items()},
                "_toggles": toggles,
            }
        }
    }

    with open(out_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)

    return out_path


def load_custom_scenario(name: str) -> dict:
    """Load a previously saved custom scenario."""
    slug = name.lower().replace(" ", "_").replace("-", "_")
    path = OVERRIDES_DIR / f"{slug}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"Scenario not found: {path}")
    with open(path) as f:
        data = yaml.safe_load(f)
    return data["overrides"][slug]


def list_saved_scenarios() -> list:
    """List all saved custom scenario names."""
    yamls = list(OVERRIDES_DIR.glob("*.yaml"))
    # Exclude built-in cost sensitivity files
    exclude = {"cost_low", "cost_high"}
    return [p.stem for p in yamls if p.stem not in exclude]
