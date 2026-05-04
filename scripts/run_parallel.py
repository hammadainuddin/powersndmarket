"""
run_parallel.py
---------------
Multiprocessing launcher for running three regional models simultaneously
within each milestone year.

Usage (CLI):
    python scripts/run_parallel.py --scenario netr_target
    python scripts/run_parallel.py --scenario baseline --regions peninsular sabah
"""

import argparse
import logging
import multiprocessing as mp
import sys
import time
from pathlib import Path
from typing import Optional

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

log = logging.getLogger(__name__)


def _worker(kwargs: dict) -> dict:
    """Worker function executed in a subprocess."""
    from scripts.myopic_chain import run_single_region
    return run_single_region(**kwargs)


def run_regions_parallel(
    regions: list,
    year: int,
    scenario: str,
    prev_caps_by_region: dict,
    run_dir: Path,
    demand_growth: Optional[dict] = None,
    carbon_price_override: Optional[float] = None,
    tech_costs: Optional[dict] = None,
    n_workers: int = 3,
) -> list:
    """
    Run all regions in parallel using multiprocessing.Pool.

    Args:
        regions: list of region names
        year: milestone year
        scenario: scenario name
        prev_caps_by_region: {region: {tech: cap}} from prior year
        run_dir: output directory for this scenario
        demand_growth: {region: annual_cagr}
        carbon_price_override: override carbon price
        tech_costs: tech cost overrides from dashboard
        n_workers: number of parallel processes (default: min(3, cpu_count))

    Returns:
        list of result dicts from run_single_region
    """
    n_workers = min(n_workers, mp.cpu_count(), len(regions))
    tasks = []
    for region in regions:
        dgrowth = demand_growth.get(region) if demand_growth else None
        tasks.append({
            "region": region,
            "year": year,
            "scenario": scenario,
            "prev_caps": prev_caps_by_region.get(region, {}),
            "run_dir": run_dir,
            "demand_growth": dgrowth,
            "carbon_price_override": carbon_price_override,
            "tech_costs": tech_costs,
        })

    t0 = time.time()
    log.info(f"Launching {len(tasks)} parallel workers for year {year} ...")

    # Use spawn context for safety across platforms
    ctx = mp.get_context("spawn")
    with ctx.Pool(processes=n_workers) as pool:
        results = pool.map(_worker, tasks)

    elapsed = time.time() - t0
    log.info(f"Year {year} complete in {elapsed:.1f}s")
    return results


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        stream=sys.stdout,
    )

    parser = argparse.ArgumentParser(description="Run Malaysia energy model (all years, parallel regions)")
    parser.add_argument("--scenario", choices=["baseline", "netr_target", "accelerated_re"],
                        default="baseline")
    parser.add_argument("--regions", nargs="+",
                        default=["peninsular", "sabah", "sarawak"])
    parser.add_argument("--no-parallel", action="store_true",
                        help="Run regions sequentially (for debugging)")
    parser.add_argument("--demand-growth-peninsular", type=float, default=None)
    parser.add_argument("--demand-growth-sabah", type=float, default=None)
    parser.add_argument("--demand-growth-sarawak", type=float, default=None)
    args = parser.parse_args()

    demand_growth = {}
    if args.demand_growth_peninsular:
        demand_growth["peninsular"] = args.demand_growth_peninsular
    if args.demand_growth_sabah:
        demand_growth["sabah"] = args.demand_growth_sabah
    if args.demand_growth_sarawak:
        demand_growth["sarawak"] = args.demand_growth_sarawak

    from scripts.myopic_chain import run_myopic_chain
    t_start = time.time()
    run_myopic_chain(
        scenario=args.scenario,
        regions=args.regions,
        demand_growth=demand_growth or None,
        parallel=not args.no_parallel,
    )
    total = time.time() - t_start
    log.info(f"\nTotal runtime: {total/60:.1f} minutes")


if __name__ == "__main__":
    main()
