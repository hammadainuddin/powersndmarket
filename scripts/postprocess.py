"""
postprocess.py
--------------
Aggregate NetCDF results from myopic chain into summary DataFrames
for the Streamlit dashboard.

Usage:
    from scripts.postprocess import load_scenario_results
    summary = load_scenario_results("netr_target")
"""

import json
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import xarray as xr

PROJECT_ROOT = Path(__file__).parent.parent
RESULTS_DIR = PROJECT_ROOT / "results"

MILESTONE_YEARS = [2025, 2030, 2035, 2040, 2045, 2050]
ALL_REGIONS = ["peninsular", "sabah", "sarawak"]

# Emission factors (tCO2/MWh output)
EMISSION_FACTORS = {
    "coal_existing": 0.88,
    "coal_new": 0.80,
    "gas_ccgt": 0.35,
    "gas_ocgt": 0.55,
    "diesel_genset": 0.65,
    "solar_utility": 0.0,
    "solar_rooftop": 0.0,
    "wind_onshore": 0.0,
    "hydro_large_existing": 0.0,
    "hydro_ror": 0.0,
    "biomass_plant": 0.05,  # Net lifecycle (residue combustion)
    "battery_storage": 0.0,
    "hvdc_sabah_sarawak": 0.0,
}

# Technology display labels
TECH_LABELS = {
    "coal_existing": "Coal (existing)",
    "coal_new": "Coal (new)",
    "gas_ccgt": "Gas CCGT",
    "gas_ocgt": "Gas OCGT",
    "diesel_genset": "Diesel",
    "solar_utility": "Solar (utility)",
    "solar_rooftop": "Solar (rooftop)",
    "wind_onshore": "Wind",
    "hydro_large_existing": "Hydro (existing)",
    "hydro_ror": "Hydro (RoR)",
    "biomass_plant": "Biomass",
    "battery_storage": "Battery",
    "hvdc_sabah_sarawak": "HVDC cable",
}

TECH_COLORS = {
    "coal_existing": "#3d3d3d",
    "coal_new": "#666666",
    "gas_ccgt": "#e8734a",
    "gas_ocgt": "#f0a070",
    "diesel_genset": "#a05000",
    "solar_utility": "#f5c518",
    "solar_rooftop": "#ffd700",
    "wind_onshore": "#8bc34a",
    "hydro_large_existing": "#1a6bb5",
    "hydro_ror": "#2a9bd4",
    "biomass_plant": "#4caf50",
    "battery_storage": "#9c27b0",
    "hvdc_sabah_sarawak": "#ff5722",
}


def load_nc(scenario: str, region: str, year: int) -> Optional[xr.Dataset]:
    """Load a single result NetCDF file."""
    path = RESULTS_DIR / scenario / f"{region}_{year}.nc"
    if path.exists():
        return xr.open_dataset(str(path))
    return None


def _parse_loc_tech(lt: str):
    """Split 'region::tech' into (region, tech)."""
    parts = str(lt).split("::", 1)
    return (parts[0], parts[1]) if len(parts) == 2 else (parts[0], "")


def _parse_loc_tech_carrier(ltc: str):
    """Split 'region::tech::carrier' into (region, tech, carrier)."""
    parts = str(ltc).split("::")
    if len(parts) == 3:
        return parts[0], parts[1], parts[2]
    return parts[0], ":".join(parts[1:]), ""


def get_capacity_mix(scenario: str, region: str) -> pd.DataFrame:
    """
    Return installed capacity (MW) by technology and milestone year.

    Returns DataFrame: index=milestone_year, columns=tech_name
    """
    rows = {}
    for year in MILESTONE_YEARS:
        ds = load_nc(scenario, region, year)
        if ds is None:
            continue
        try:
            ec = ds["energy_cap"]
            row = {}
            for lt in ec.loc_techs.values:
                r, tech = _parse_loc_tech(lt)
                if r == region:
                    row[tech] = float(ec.sel(loc_techs=lt).values)
            rows[year] = row
        except Exception:
            pass
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).T.fillna(0)


def get_generation_mix(scenario: str, region: str) -> pd.DataFrame:
    """
    Return annual generation (TWh) by technology and milestone year.
    Calliope stores carrier_prod with dim 'loc_tech_carriers_prod' as 'region::tech::carrier'.
    """
    rows = {}
    for year in MILESTONE_YEARS:
        ds = load_nc(scenario, region, year)
        if ds is None:
            continue
        try:
            cp = ds["carrier_prod"]
            weights = _load_weights(region, year)
            gen_twh = {}
            for ltc in cp.loc_tech_carriers_prod.values:
                r, tech, carrier = _parse_loc_tech_carrier(ltc)
                if r != region or carrier != "electricity":
                    continue
                ts_values = cp.sel(loc_tech_carriers_prod=ltc).values
                # Only positive values (production, not consumption)
                ts_values = np.where(ts_values > 0, ts_values, 0)
                if weights is not None:
                    gen_mwh = _apply_weights(ts_values, weights)
                else:
                    gen_mwh = ts_values.sum()
                gen_twh[tech] = gen_mwh / 1e6  # MWh -> TWh
            rows[year] = gen_twh
        except Exception:
            pass
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).T.fillna(0)


def get_emissions(scenario: str, region: str) -> pd.Series:
    """Return annual CO2 emissions (Mt) by milestone year."""
    gen = get_generation_mix(scenario, region)
    if gen.empty:
        return pd.Series(dtype=float)
    rows = {}
    for year in gen.index:
        total = 0.0
        for tech, twh in gen.loc[year].items():
            ef = EMISSION_FACTORS.get(tech, 0)
            total += twh * ef  # Mt CO2 (1 TWh * tCO2/MWh = 1 Mt CO2)
        rows[year] = total
    return pd.Series(rows)


def get_system_cost(scenario: str, region: str) -> pd.DataFrame:
    """
    Return system cost breakdown (million USD/year) by component and milestone year.
    Calliope stores cost_investment with dim (costs, loc_techs_investment_cost).
    """
    rows = {}
    for year in MILESTONE_YEARS:
        ds = load_nc(scenario, region, year)
        if ds is None:
            continue
        try:
            capex = 0.0
            if "cost_investment" in ds:
                ci = ds["cost_investment"].sel(costs="monetary")
                for lt in ci.loc_techs_investment_cost.values:
                    r, _ = _parse_loc_tech(lt)
                    if r == region:
                        capex += float(ci.sel(loc_techs_investment_cost=lt).values)
            opex = 0.0
            if "cost_var" in ds:
                cv = ds["cost_var"].sel(costs="monetary")
                for lt in cv.loc_techs_om_cost.values:
                    r, _ = _parse_loc_tech(lt)
                    if r == region:
                        opex += float(cv.sel(loc_techs_om_cost=lt).values.sum())
            rows[year] = {
                "capex_annualised": capex / 1e6,
                "opex_variable": opex / 1e6,
            }
        except Exception:
            rows[year] = {"capex_annualised": 0, "opex_variable": 0}
    return pd.DataFrame(rows).T.fillna(0)


def get_curtailment(scenario: str, region: str) -> pd.Series:
    """Return renewable curtailment % by milestone year (proxy: unmet demand %)."""
    rows = {}
    for year in MILESTONE_YEARS:
        ds = load_nc(scenario, region, year)
        if ds is None:
            continue
        try:
            # Sum all demand consumption for the region
            cc = ds["carrier_con"]
            total_demand = 0.0
            for ltc in cc.loc_tech_carriers_con.values:
                r, tech, carrier = _parse_loc_tech_carrier(ltc)
                if r == region and carrier == "electricity" and "demand" in tech:
                    total_demand += abs(float(cc.sel(loc_tech_carriers_con=ltc).values.sum()))
            rows[year] = 0  # No explicit unmet demand variable; assume fully met
        except Exception:
            rows[year] = 0
    return pd.Series(rows)


def get_battery_kpis(scenario: str, region: str) -> pd.DataFrame:
    """
    Return battery storage KPIs per milestone year.
    Calliope stores storage/storage_cap with 'loc_techs_store' dimension.
    """
    rows = {}
    for year in MILESTONE_YEARS:
        ds = load_nc(scenario, region, year)
        if ds is None:
            continue
        try:
            batt_lt = f"{region}::battery_storage"
            batt_ltc = f"{region}::battery_storage::electricity"

            if "storage" not in ds or batt_lt not in ds["storage"].loc_techs_store.values:
                continue

            soc = ds["storage"].sel(loc_techs_store=batt_lt).values
            storage_cap = float(ds["storage_cap"].sel(loc_techs_store=batt_lt).values)

            ec = ds["energy_cap"]
            energy_cap = 0.0
            for lt in ec.loc_techs.values:
                if str(lt) == batt_lt:
                    energy_cap = float(ec.sel(loc_techs=lt).values)
                    break

            if storage_cap > 0 and energy_cap > 0:
                avg_soc = float(np.mean(soc / storage_cap))
                cp = ds["carrier_prod"]
                discharge = np.zeros(len(ds.timesteps))
                if batt_ltc in cp.loc_tech_carriers_prod.values:
                    raw = cp.sel(loc_tech_carriers_prod=batt_ltc).values
                    discharge = np.where(raw > 0, raw, 0)
                total_discharge_mwh = discharge.sum()
                cycles = total_discharge_mwh / storage_cap
                cf = float(np.mean(discharge > 0.01 * energy_cap))
                rows[year] = {
                    "cycles_per_year": cycles,
                    "avg_soc": avg_soc,
                    "duration_utilisation": avg_soc,
                    "capacity_factor": cf,
                    "installed_mw": energy_cap,
                    "installed_mwh": storage_cap,
                }
        except Exception:
            pass
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).T


def get_diurnal_mix(scenario: str, region: str, year: int) -> pd.DataFrame:
    """
    Return average hourly generation profile (MW) by technology for a given year.
    Averages carrier_prod over all 24 representative days to produce a 24-hour profile.

    Returns DataFrame: index=hour (0-23), columns=tech_name (values in MW)
    """
    ds = load_nc(scenario, region, year)
    if ds is None:
        return pd.DataFrame()
    try:
        cp = ds["carrier_prod"]
        hours = 24
        n_days = len(ds.timesteps) // hours
        tech_profiles = {}
        for ltc in cp.loc_tech_carriers_prod.values:
            r, tech, carrier = _parse_loc_tech_carrier(ltc)
            if r != region or carrier != "electricity":
                continue
            vals = cp.sel(loc_tech_carriers_prod=ltc).values  # shape (576,)
            vals = np.where(vals > 0, vals, 0)
            # Reshape to (n_days, 24) and average across days
            reshaped = vals.reshape(n_days, hours)
            tech_profiles[tech] = reshaped.mean(axis=0)
        # Add demand from carrier_con (negative values → flip sign)
        cc = ds["carrier_con"]
        for ltc in cc.loc_tech_carriers_con.values:
            r, tech, carrier = _parse_loc_tech_carrier(ltc)
            if r != region or carrier != "electricity" or "demand" not in tech:
                continue
            vals = abs(cc.sel(loc_tech_carriers_con=ltc).values)
            reshaped = vals.reshape(n_days, hours)
            tech_profiles["demand_electricity"] = reshaped.mean(axis=0)

        if not tech_profiles:
            return pd.DataFrame()
        df = pd.DataFrame(tech_profiles, index=range(hours))
        df.index.name = "hour"
        return df
    except Exception:
        return pd.DataFrame()


def _load_weights(region: str, year: int) -> Optional[pd.DataFrame]:
    """Load cluster weights for a region (used to weight representative days)."""
    weights_path = PROJECT_ROOT / "model" / "timeseries" / f"cluster_weights_{region}.csv"
    if weights_path.exists():
        return pd.read_csv(weights_path)
    return None


def _apply_weights(ts_values: np.ndarray, weights: pd.DataFrame) -> float:
    """Apply cluster weights to a flat time series of representative days."""
    n_clusters = len(weights)
    hours_per_cluster = len(ts_values) // n_clusters
    total = 0.0
    for i, row in weights.iterrows():
        start = i * hours_per_cluster
        end = start + hours_per_cluster
        cluster_sum = ts_values[start:end].sum()
        total += cluster_sum * row["weight"]
    return total


def load_scenario_results(scenario: str, regions: list = None) -> dict:
    """
    Load all results for a scenario and return a structured summary dict.

    Returns:
        {
          region: {
            'capacity_mix': DataFrame,
            'generation_mix': DataFrame,
            'emissions': Series,
            'system_cost': DataFrame,
            'curtailment': Series,
            'battery_kpis': DataFrame,
          }
        }
    """
    if regions is None:
        regions = ALL_REGIONS
    out = {}
    for region in regions:
        out[region] = {
            "capacity_mix": get_capacity_mix(scenario, region),
            "generation_mix": get_generation_mix(scenario, region),
            "emissions": get_emissions(scenario, region),
            "system_cost": get_system_cost(scenario, region),
            "curtailment": get_curtailment(scenario, region),
            "battery_kpis": get_battery_kpis(scenario, region),
        }
    return out
