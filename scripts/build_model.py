"""
build_model.py
--------------
Assemble a Calliope model object for a given region and milestone year.

Handles:
- Loading region-specific location YAML
- Applying demand growth scaling
- Applying myopic capacity constraints (min/max from previous year)
- Applying scenario overrides
- Unlocking HVDC cable from 2035
- Setting RE group_share constraints
- Returning a ready-to-run calliope.Model object

Usage (Python API):
    from scripts.build_model import build_model
    model = build_model("peninsular", 2030, scenario="netr_target",
                        prev_caps={}, demand_growth=0.04)
    model.run()
"""

import copy
import json
from pathlib import Path
from typing import Optional

import calliope
import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).parent.parent
MODEL_DIR = PROJECT_ROOT / "model"
ANNUAL_INPUTS_PATH = PROJECT_ROOT / "data" / "annual_inputs.json"


def _load_annual_inputs() -> Optional[dict]:
    """Load annual_inputs.json if it exists (written by the Data dashboard page)."""
    if ANNUAL_INPUTS_PATH.exists():
        with open(ANNUAL_INPUTS_PATH) as f:
            return json.load(f)
    return None

# RE share targets per milestone year (fraction of generation)
# Baseline: NETR trajectory (can be overridden by scenario)
RE_TARGETS = {
    # Baseline: gradual growth aligned with existing policy trajectory (no new mandates)
    "baseline": {2025: 0.20, 2030: 0.25, 2035: 0.30, 2040: 0.35, 2045: 0.40, 2050: 0.45},
    # NETR: policy-aligned, 70% RE by 2050 per NETR roadmap
    "netr_target": {2025: 0.31, 2030: 0.35, 2035: 0.40, 2040: 0.52, 2045: 0.62, 2050: 0.70},
    # Accelerated: aggressive decarbonisation path
    "accelerated_re": {2025: 0.35, 2030: 0.45, 2035: 0.55, 2040: 0.65, 2045: 0.75, 2050: 0.82},
}

# RE technology names (used in group_share constraint)
RE_TECHS = [
    "solar_utility", "solar_rooftop", "wind_onshore",
    "hydro_large_existing", "hydro_ror", "biomass_plant",
]

# Demand growth rates (annual CAGR by region)
DEFAULT_DEMAND_GROWTH = {
    "peninsular": 0.035,  # 3.5% per year
    "sabah": 0.045,        # 4.5% (faster developing region)
    "sarawak": 0.040,      # 4.0% (industrialisation push)
}

# Coal retirement schedule: (min_MW, max_MW) per region per milestone year
# 2025: min=max=initial (existing fleet forced in); later years: min=0, max=ceiling
COAL_RETIREMENT = {
    "peninsular": {2025: (12800, 12800), 2030: (0, 10000), 2035: (0, 7000),
                   2040: (0, 3000), 2045: (0, 0), 2050: (0, 0)},
    "sarawak":    {2025: (920, 920), 2030: (0, 920), 2035: (0, 500),
                   2040: (0, 0), 2045: (0, 0), 2050: (0, 0)},
    "sabah":      {yr: (0, 0) for yr in [2025, 2030, 2035, 2040, 2045, 2050]},
}

# Solar deployment caps (MW) per region per milestone year
# Based on realistic deployment trajectories (2025 ~1.5 GW actual, growing to NETR targets)
SOLAR_MAX_CAPACITY = {
    "peninsular": {2025: 7000, 2030: 15000, 2035: 22000, 2040: 30000, 2045: 37000, 2050: 40000},
    "sabah":      {2025: 800,  2030: 1500,  2035: 2000,  2040: 2500,  2045: 3000,  2050: 3000},
    "sarawak":    {2025: 500,  2030: 2000,  2035: 4000,  2040: 6000,  2045: 8000,  2050: 10000},
}

# Carbon price trajectory (USD/tCO2) — used as cost for CO2 carrier if enabled
CARBON_PRICE = {
    "baseline": {2025: 0, 2030: 0, 2035: 0, 2040: 0, 2045: 0, 2050: 0},
    "netr_target": {2025: 10, 2030: 20, 2035: 35, 2040: 50, 2045: 65, 2050: 80},
    "accelerated_re": {2025: 25, 2030: 50, 2035: 80, 2040: 110, 2045: 135, 2050: 150},
}


def compute_demand_scale(region: str, milestone_year: int, base_year: int = 2025,
                         growth_rate: Optional[float] = None) -> float:
    """Compound annual growth from base year to milestone year."""
    rate = growth_rate if growth_rate is not None else DEFAULT_DEMAND_GROWTH[region]
    return (1 + rate) ** (milestone_year - base_year)


def build_override_dict(
    region: str,
    milestone_year: int,
    scenario: str = "baseline",
    prev_caps: dict = None,
    demand_growth: Optional[float] = None,
    carbon_price_override: Optional[float] = None,
    tech_costs: Optional[dict] = None,
) -> dict:
    """
    Build a dictionary of Calliope overrides for a specific region/year/scenario.

    Args:
        region: 'peninsular', 'sabah', or 'sarawak'
        milestone_year: e.g. 2030
        scenario: 'baseline', 'netr_target', or 'accelerated_re'
        prev_caps: {tech_name: capacity_MW} from previous milestone year's results
        demand_growth: annual CAGR (overrides default)
        carbon_price_override: USD/tCO2 (overrides scenario default)
        tech_costs: {tech_name: {'energy_cap': ..., 'om_annual': ...}} dashboard overrides

    Returns:
        dict suitable for calliope override_dict parameter
    """
    if prev_caps is None:
        prev_caps = {}

    overrides = {"locations": {region: {"techs": {}}}, "links": {}}
    loc = overrides["locations"][region]["techs"]

    # --- Load annual inputs from dashboard Data page (if saved) ---
    annual = _load_annual_inputs()
    yr_str = str(milestone_year)

    # --- Coal retirement ---
    if annual:
        coal_cap = annual["capacity"][region]["coal_existing"][yr_str]
        # Max = capacity from the previous milestone year (can't un-retire coal)
        prev_years = [y for y in [2025, 2030, 2035, 2040, 2045, 2050] if y < milestone_year]
        if prev_years:
            coal_prev = annual["capacity"][region]["coal_existing"][str(max(prev_years))]
            coal_max = max(coal_cap, coal_prev)  # can't exceed prior year
        else:
            coal_max = coal_cap
        coal_min = coal_cap  # must retire at least to the scheduled floor
        if coal_min == coal_max:
            loc["coal_existing"] = {"constraints": {"energy_cap_equals": coal_cap}}
        else:
            loc["coal_existing"] = {"constraints": {"energy_cap_min": coal_min, "energy_cap_max": coal_max}}
    else:
        coal_min, coal_max = COAL_RETIREMENT.get(region, {}).get(milestone_year, (0, 0))
        if coal_min == coal_max:
            loc["coal_existing"] = {"constraints": {"energy_cap_equals": coal_max}}
        else:
            loc["coal_existing"] = {"constraints": {"energy_cap_min": coal_min, "energy_cap_max": coal_max}}
    loc["coal_new"] = {"constraints": {"energy_cap_max": 0}}  # No new coal in all scenarios

    # --- Capacity caps from annual inputs ---
    if annual:
        cap_data = annual["capacity"][region]

        def _set_cap(tech_key, loc_tech):
            val = cap_data.get(tech_key, {}).get(yr_str)
            if val is not None:
                if loc_tech not in loc:
                    loc[loc_tech] = {"constraints": {}}
                loc[loc_tech]["constraints"]["energy_cap_max"] = val

        def _set_min(tech_key, loc_tech):
            val = cap_data.get(tech_key, {}).get(yr_str)
            if val is not None and val > 0:
                if loc_tech not in loc:
                    loc[loc_tech] = {"constraints": {}}
                loc[loc_tech]["constraints"]["energy_cap_min"] = val

        _set_cap("solar_utility_max", "solar_utility")
        _set_cap("solar_rooftop_max", "solar_rooftop")
        _set_cap("wind_onshore_max", "wind_onshore")
        _set_cap("biomass_max", "biomass_plant")
        _set_cap("battery_max", "battery_storage")
        _set_min("gas_ccgt", "gas_ccgt")
        _set_min("gas_ocgt", "gas_ocgt")
        _set_min("hydro_large_existing", "hydro_large_existing")
        _set_min("hydro_ror_min", "hydro_ror")
        _set_min("diesel_genset", "diesel_genset")

        # Fuel cost overrides derived from gas/coal prices
        fp = annual["fuel_prices"].get(yr_str, {})
        gas_price = fp.get("gas_usd_gj", 10.0)
        coal_price = fp.get("coal_usd_tonne", 100.0)
        coal_gj_per_t = 26.0  # GJ/tonne for thermal coal
        for tech, eff, var_om in [
            ("gas_ccgt",      0.52, 3.0),
            ("gas_ocgt",      0.35, 4.0),
            ("coal_existing", 0.38, 4.0),
            ("coal_new",      0.42, 3.0),
        ]:
            if "gas" in tech:
                fuel_cost = 3.6 * gas_price / eff
            else:
                fuel_cost = 3.6 * coal_price / (coal_gj_per_t * eff)
            om_con = round(fuel_cost + var_om, 2)
            if tech not in loc:
                loc[tech] = {}
            loc[tech].setdefault("costs", {}).setdefault("monetary", {})["om_con"] = om_con
    else:
        # --- Solar deployment caps (fallback to hardcoded) ---
        solar_max = SOLAR_MAX_CAPACITY.get(region, {}).get(milestone_year)
        if solar_max is not None:
            if "solar_utility" not in loc:
                loc["solar_utility"] = {"constraints": {}}
            loc["solar_utility"]["constraints"]["energy_cap_max"] = solar_max
            rooftop_max = int(solar_max * 0.15)
            if "solar_rooftop" not in loc:
                loc["solar_rooftop"] = {"constraints": {}}
            loc["solar_rooftop"]["constraints"]["energy_cap_max"] = rooftop_max

    # --- Carry forward minimum capacity from previous year ---
    # Exclude: coal (has retirement schedule), demand (energy_cap_min not allowed on demand)
    _no_carryforward = {"coal_existing", "coal_new", "demand_electricity"}
    for tech, cap in prev_caps.items():
        if cap > 0 and tech not in _no_carryforward:
            if tech not in loc:
                loc[tech] = {"constraints": {}}
            # Existing capacity is a floor — model can build on top
            # Clamp to energy_cap_max if one is set (prevents infeasibility)
            existing_max = loc[tech]["constraints"].get("energy_cap_max")
            if existing_max is not None and cap > existing_max:
                cap = existing_max
            loc[tech]["constraints"]["energy_cap_min"] = cap

    # --- Technology cost overrides (from dashboard) ---
    if tech_costs:
        for tech, cost_dict in tech_costs.items():
            if tech not in loc:
                loc[tech] = {"costs": {"monetary": {}}}
            else:
                loc[tech].setdefault("costs", {}).setdefault("monetary", {})
            for cost_key, cost_val in cost_dict.items():
                loc[tech]["costs"]["monetary"][cost_key] = cost_val

    # --- HVDC Sabah-Sarawak ---
    if region in ("sabah", "sarawak"):
        if annual:
            hvdc_cap = annual["hvdc_mw"].get(yr_str, 0)
        else:
            hvdc_cap = 1000 if (milestone_year >= 2035 and scenario != "baseline") else 0
        if hvdc_cap > 0:
            overrides["links"]["sabah,sarawak"] = {
                "techs": {"hvdc_sabah_sarawak": {"constraints": {"energy_cap_max": hvdc_cap}}}
            }

    # --- RE group_share constraint ---
    re_target = RE_TARGETS.get(scenario, RE_TARGETS["baseline"]).get(milestone_year, 0)

    return overrides, re_target


def build_model(
    region: str,
    milestone_year: int,
    scenario: str = "baseline",
    prev_caps: dict = None,
    demand_growth: Optional[float] = None,
    carbon_price_override: Optional[float] = None,
    tech_costs: Optional[dict] = None,
    n_rep_days: int = 24,
) -> calliope.Model:
    """
    Build and return a Calliope Model object ready to run.

    Args:
        region: 'peninsular', 'sabah', or 'sarawak'
        milestone_year: 2025, 2030, ..., 2050
        scenario: 'baseline', 'netr_target', or 'accelerated_re'
        prev_caps: capacity carried forward from prior milestone year
        demand_growth: annual demand CAGR (None = use regional default)
        carbon_price_override: override carbon price USD/tCO2
        tech_costs: technology cost overrides from dashboard
        n_rep_days: number of representative days (default 24)
    """
    # Build override dict
    override_dict, re_target = build_override_dict(
        region, milestone_year, scenario, prev_caps,
        demand_growth, carbon_price_override, tech_costs,
    )

    # Add group_share constraint for RE target
    if re_target > 0:
        override_dict["group_constraints"] = {
            "min_renewable_share": {
                "locs": [region],
                "techs": RE_TECHS,
                "energy_cap_share_min": re_target,
            }
        }

    # Determine representative day end date (n_rep_days days starting Jan 1)
    # Time series data is always indexed from 2025 (representative year).
    # The milestone_year only affects capacity constraints and demand scaling.
    import datetime
    TS_BASE_YEAR = 2025
    start_dt = datetime.date(TS_BASE_YEAR, 1, 1)
    end_dt = start_dt + datetime.timedelta(days=n_rep_days - 1)
    subset_end = f"{end_dt.year}-{end_dt.month:02d}-{end_dt.day:02d} 23:00:00"
    subset_start = f"{TS_BASE_YEAR}-01-01 00:00:00"

    # Write a per-run model YAML inside model/ so relative imports resolve correctly
    run_yaml_path = MODEL_DIR / f"_run_{region}_{milestone_year}_{scenario}.yaml"

    run_config = {
        "import": [
            "techs/supply.yaml",
            "techs/storage.yaml",
            "techs/demand.yaml",
            "techs/transmission.yaml",
            f"locations/{region}.yaml",
        ],
        "model": {
            "name": f"Malaysia {region.title()} {milestone_year} ({scenario})",
            "calliope_version": "0.6.10",
            "timeseries_data_path": f"timeseries/{region}",
            "subset_time": [subset_start, subset_end],
        },
        "run": {
            "mode": "plan",
            "solver": "appsi_highs",
            "solver_options": {},
            "backend": "pyomo",
            "objective_options": {"cost_class": {"monetary": 1}, "sense": "minimize"},
        },
    }

    with open(run_yaml_path, "w") as f:
        yaml.dump(run_config, f, default_flow_style=False, sort_keys=False)

    model = calliope.Model(
        str(run_yaml_path),
        override_dict=override_dict,
    )
    return model


def scale_demand_csv(region: str, milestone_year: int,
                     demand_growth: Optional[float] = None) -> None:
    """
    Scale the clustered demand CSV for a given milestone year and save as
    demand_{region}_{milestone_year}.csv for Calliope to reference.
    """
    base_path = MODEL_DIR / "timeseries" / f"demand_{region}.csv"
    if not base_path.exists():
        raise FileNotFoundError(f"Base demand CSV not found: {base_path}")

    scale = compute_demand_scale(region, milestone_year, growth_rate=demand_growth)
    df = pd.read_csv(base_path, index_col=0, parse_dates=True)
    # Ensure column is named after the region (Calliope requirement)
    df.columns = [region]
    df_scaled = df * scale
    # Save into region subdirectory as demand.csv (overwritten per milestone year run)
    region_ts_dir = MODEL_DIR / "timeseries" / region
    region_ts_dir.mkdir(parents=True, exist_ok=True)
    out_path = region_ts_dir / "demand.csv"
    df_scaled.to_csv(out_path)


if __name__ == "__main__":
    # Quick test: build Peninsular 2025 baseline
    m = build_model("peninsular", 2025, scenario="baseline")
    print(m)
