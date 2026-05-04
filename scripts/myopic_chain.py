"""
myopic_chain.py
---------------
Myopic rolling optimization engine for Malaysia energy model.

For each milestone year:
  1. Build Calliope models for all three regions (or selected region)
  2. Run each region model
  3. Extract newly built capacity from results
  4. Pass forward as minimum capacity constraints to next milestone year

The three regional models in each milestone year run in parallel via run_parallel.py.
This module handles the sequential chaining logic.

Usage:
    from scripts.myopic_chain import run_myopic_chain
    results = run_myopic_chain(scenario="netr_target", regions=["peninsular", "sabah", "sarawak"])
"""

import json
import logging
from pathlib import Path
from typing import Optional

import pandas as pd
import xarray as xr

PROJECT_ROOT = Path(__file__).parent.parent
RESULTS_DIR = PROJECT_ROOT / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

MILESTONE_YEARS = [2025, 2030, 2035, 2040, 2045, 2050]
ALL_REGIONS = ["peninsular", "sabah", "sarawak"]

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


def extract_capacities(model_results: xr.Dataset, region: str) -> dict:
    """
    Extract installed capacity (energy_cap, in MW) per technology from model results.

    Returns:
        {tech_name: capacity_MW}
    """
    caps = {}
    try:
        # model_results is an xr.Dataset; energy_cap has dim 'loc_techs'
        # with combined coordinate strings like "peninsular::solar_utility"
        energy_cap = model_results["energy_cap"]
        prefix = f"{region}::"
        for loc_tech in energy_cap.loc_techs.values:
            if str(loc_tech).startswith(prefix):
                tech = str(loc_tech).split("::", 1)[1]
                val = float(energy_cap.sel(loc_techs=loc_tech).values)
                if val > 0.01:
                    caps[tech] = val
    except Exception as e:
        log.warning(f"Could not extract capacities for {region}: {e}")
    return caps


def save_results(model_results: xr.Dataset, region: str, year: int,
                 scenario: str, run_dir: Path) -> Path:
    """Save model results to NetCDF."""
    out_path = run_dir / f"{region}_{year}.nc"
    # Fix attrs that may contain None (not serializable to NetCDF)
    clean_attrs = {
        k: (str(v) if v is None else v)
        for k, v in model_results.attrs.items()
    }
    model_results = model_results.assign_attrs(clean_attrs)
    model_results.to_netcdf(str(out_path))
    log.info(f"Saved results: {out_path}")
    return out_path


def run_single_region(
    region: str,
    year: int,
    scenario: str,
    prev_caps: dict,
    run_dir: Path,
    demand_growth: Optional[float] = None,
    carbon_price_override: Optional[float] = None,
    tech_costs: Optional[dict] = None,
) -> dict:
    """
    Build and run a single region model for a given milestone year.

    Returns:
        {'region': str, 'year': int, 'caps': dict, 'result_path': Path, 'success': bool}
    """
    from scripts.build_model import build_model, scale_demand_csv

    log.info(f"Starting: {region} {year} ({scenario})")

    try:
        # Scale demand CSV for this milestone year
        scale_demand_csv(region, year, demand_growth=demand_growth)

        # Build and run model
        model = build_model(
            region, year, scenario=scenario, prev_caps=prev_caps,
            demand_growth=demand_growth, carbon_price_override=carbon_price_override,
            tech_costs=tech_costs,
        )
        model.run()

        # Extract results
        caps = extract_capacities(model.results, region)
        result_path = save_results(model.results, region, year, scenario, run_dir)

        log.info(f"Completed: {region} {year} — top techs: "
                 + ", ".join(f"{k}={v:.0f}MW" for k, v in sorted(caps.items(),
                              key=lambda x: -x[1])[:5]))
        return {"region": region, "year": year, "caps": caps,
                "result_path": result_path, "success": True}

    except Exception as e:
        log.error(f"FAILED: {region} {year}: {e}", exc_info=True)
        return {"region": region, "year": year, "caps": prev_caps,
                "result_path": None, "success": False, "error": str(e)}


def run_myopic_chain(
    scenario: str = "baseline",
    regions: list = None,
    demand_growth: Optional[dict] = None,
    carbon_price_override: Optional[float] = None,
    tech_costs: Optional[dict] = None,
    parallel: bool = True,
    progress_callback=None,
) -> dict:
    """
    Run full myopic chain across all milestone years.

    Args:
        scenario: 'baseline', 'netr_target', or 'accelerated_re'
        regions: list of regions to run (default: all three)
        demand_growth: {region: annual_cagr} e.g. {'peninsular': 0.04}
        carbon_price_override: override carbon price for all years
        tech_costs: technology cost overrides from dashboard
        parallel: run regions in parallel within each milestone year
        progress_callback: optional fn(year, region, status) for dashboard updates

    Returns:
        {region: {'caps_by_year': {year: {tech: cap}}, 'result_paths': [...]}}
    """
    if regions is None:
        regions = ALL_REGIONS

    run_dir = RESULTS_DIR / scenario
    run_dir.mkdir(parents=True, exist_ok=True)

    # State: capacities carried forward per region
    carried_caps = {r: {} for r in regions}
    all_results = {r: {"caps_by_year": {}, "result_paths": []} for r in regions}

    for year in MILESTONE_YEARS:
        log.info(f"\n{'='*60}")
        log.info(f"Milestone year: {year}")
        log.info(f"{'='*60}")

        if parallel:
            # Run all regions in parallel
            from scripts.run_parallel import run_regions_parallel
            year_results = run_regions_parallel(
                regions=regions,
                year=year,
                scenario=scenario,
                prev_caps_by_region=carried_caps,
                run_dir=run_dir,
                demand_growth=demand_growth,
                carbon_price_override=carbon_price_override,
                tech_costs=tech_costs,
            )
        else:
            # Sequential (for debugging)
            year_results = []
            for region in regions:
                dgrowth = demand_growth.get(region) if demand_growth else None
                res = run_single_region(
                    region, year, scenario, carried_caps[region], run_dir,
                    demand_growth=dgrowth, carbon_price_override=carbon_price_override,
                    tech_costs=tech_costs,
                )
                year_results.append(res)

        # Update carried capacities and record results
        for res in year_results:
            region = res["region"]
            carried_caps[region] = res["caps"]
            all_results[region]["caps_by_year"][year] = res["caps"]
            if res["result_path"]:
                all_results[region]["result_paths"].append(str(res["result_path"]))
            if progress_callback:
                progress_callback(year, region, "success" if res["success"] else "failed")

    # Save summary JSON
    summary_path = run_dir / "summary_caps.json"
    with open(summary_path, "w") as f:
        json.dump(
            {r: {str(y): v for y, v in all_results[r]["caps_by_year"].items()}
             for r in all_results},
            f, indent=2
        )
    log.info(f"Run complete. Summary: {summary_path}")
    return all_results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", default="baseline")
    parser.add_argument("--regions", nargs="+", default=["peninsular", "sabah", "sarawak"])
    parser.add_argument("--no-parallel", action="store_true")
    args = parser.parse_args()
    run_myopic_chain(scenario=args.scenario, regions=args.regions, parallel=not args.no_parallel)
