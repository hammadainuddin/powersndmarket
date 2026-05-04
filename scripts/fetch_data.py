"""
fetch_data.py
-------------
Fetch and process input time series data for all three Malaysia regions:
  1. Solar irradiance (GHI -> capacity factor) from NASA POWER API
  2. Synthesize hourly demand profiles from monthly load statistics
  3. Hydro capacity factors from seasonal inflow assumptions

Usage:
    python scripts/fetch_data.py --region all

Outputs (raw CSVs):
    model/timeseries/solar_cf_{region}_raw.csv
    model/timeseries/demand_{region}_raw.csv
    model/timeseries/hydro_cf_{region}_raw.csv
    model/timeseries/wind_cf_{region}_raw.csv   (placeholder, low values)
"""

import argparse
import time
import numpy as np
import pandas as pd
import requests
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
TIMESERIES_DIR = PROJECT_ROOT / "model" / "timeseries"
TIMESERIES_DIR.mkdir(parents=True, exist_ok=True)

NASA_POWER_URL = "https://power.larc.nasa.gov/api/temporal/hourly/point"

# Representative coordinates per region
REGION_COORDS = {
    "peninsular": {"lat": 3.8, "lon": 101.7},   # Kuala Lumpur area
    "sabah": {"lat": 5.9, "lon": 116.1},          # Kota Kinabalu area
    "sarawak": {"lat": 1.55, "lon": 110.35},      # Kuching area
}

# Annual peak demand (MW) and energy (TWh) per region (2024 estimates)
REGION_DEMAND = {
    "peninsular": {"peak_mw": 20500, "annual_twh": 120.0},
    "sabah": {"peak_mw": 1200, "annual_twh": 7.5},
    "sarawak": {"peak_mw": 3200, "annual_twh": 20.0},
}

# Hydro seasonal profile (monthly capacity factors, 0-1)
# Peninsular: influenced by two monsoon seasons (Northeast Oct-Mar, Southwest May-Sep)
# Sarawak: Bakun/Murum — wetter Nov-Feb
HYDRO_MONTHLY_CF = {
    "peninsular": [0.55, 0.50, 0.45, 0.40, 0.45, 0.50, 0.55, 0.55, 0.60, 0.65, 0.70, 0.65],
    "sabah": [0.60, 0.55, 0.50, 0.45, 0.40, 0.38, 0.40, 0.42, 0.48, 0.55, 0.62, 0.65],
    "sarawak": [0.75, 0.70, 0.65, 0.60, 0.55, 0.55, 0.55, 0.55, 0.58, 0.62, 0.70, 0.78],
}


def fetch_nasa_solar(lat: float, lon: float, year: int = 2022) -> pd.Series:
    """Fetch hourly GHI from NASA POWER and convert to PV capacity factor."""
    params = {
        "parameters": "ALLSKY_SFC_SW_DWN",
        "community": "RE",
        "longitude": lon,
        "latitude": lat,
        "start": f"{year}0101",
        "end": f"{year}1231",
        "format": "JSON",
        "time-standard": "UTC",
        "header": "true",
    }
    print(f"  Fetching NASA POWER solar for lat={lat}, lon={lon}, year={year} ...")
    resp = requests.get(NASA_POWER_URL, params=params, timeout=60)
    resp.raise_for_status()
    data = resp.json()

    # NASA POWER returns data keyed as YYYYMMDDHH
    raw = data["properties"]["parameter"]["ALLSKY_SFC_SW_DWN"]
    values = list(raw.values())

    # Convert GHI (W/m2) -> PV capacity factor using simple linear model
    # Standard test conditions: 1000 W/m2 = 1.0 CF; apply performance ratio 0.80
    ghi = np.array(values, dtype=float)
    ghi = np.where(ghi < 0, 0, ghi)  # Remove fill values
    cf = (ghi / 1000.0) * 0.80       # Performance ratio 0.80
    cf = np.clip(cf, 0, 1)

    # Build hourly DatetimeIndex
    idx = pd.date_range(f"{year}-01-01", periods=len(cf), freq="h")
    series = pd.Series(cf, index=idx, name="solar_cf")

    # Trim/pad to exactly 8760 hours
    series = series.iloc[:8760]
    return series


def synthesize_demand(region: str, year: int = 2025) -> pd.Series:
    """
    Synthesize hourly demand profile from:
    - Annual energy target (TWh)
    - Typical Malaysian load shape: weekday/weekend pattern, seasonal AC peak
    """
    annual_twh = REGION_DEMAND[region]["annual_twh"]
    peak_mw = REGION_DEMAND[region]["peak_mw"]

    idx = pd.date_range(f"{year}-01-01", periods=8760, freq="h")
    hours = np.arange(8760)
    hour_of_day = hours % 24
    day_of_year = hours // 24

    # Base daily load shape (normalized, peak=1.0)
    # Malaysia: morning peak ~9-11am, evening peak ~19-21h
    daily_shape = (
        0.60
        + 0.20 * np.exp(-((hour_of_day - 10) ** 2) / 8)   # morning peak
        + 0.30 * np.exp(-((hour_of_day - 20) ** 2) / 6)   # evening peak
        - 0.10 * np.exp(-((hour_of_day - 4) ** 2) / 4)    # night trough
    )
    daily_shape = np.clip(daily_shape, 0.45, 1.0)

    # Seasonal factor: higher demand in hot dry months (March-May, Sept-Oct)
    day_in_year = day_of_year % 365
    seasonal = 1.0 + 0.08 * np.sin(2 * np.pi * (day_in_year - 60) / 365)

    # Weekday factor (Mon-Fri higher, Sat-Sun ~10% lower)
    weekday = idx.dayofweek  # 0=Mon, 6=Sun
    dow_factor = np.where(weekday >= 5, 0.88, 1.0)

    profile = daily_shape * seasonal * dow_factor
    profile = profile / profile.max()  # Normalize to 1.0 peak

    # Scale to actual MW
    # Calliope expects the demand as energy_cap_equals=1 with resource in MW
    demand_mw = profile * peak_mw

    # Normalize so that the series integrates to annual_twh
    actual_twh = demand_mw.sum() / 1e6  # MWh -> TWh
    demand_mw = demand_mw * (annual_twh / actual_twh)

    # Calliope demand tech: resource should be negative (demand sink)
    series = pd.Series(-demand_mw, index=idx, name="demand")
    return series


def synthesize_hydro_cf(region: str, year: int = 2025) -> pd.Series:
    """Build hourly hydro capacity factor from monthly seasonality + small noise."""
    idx = pd.date_range(f"{year}-01-01", periods=8760, freq="h")
    monthly_cf = HYDRO_MONTHLY_CF[region]

    cf_hourly = np.zeros(8760)
    for i, ts in enumerate(idx):
        month = ts.month - 1  # 0-indexed
        base_cf = monthly_cf[month]
        # Add small random variation (±3%)
        cf_hourly[i] = base_cf + np.random.normal(0, 0.03)

    cf_hourly = np.clip(cf_hourly, 0.1, 0.95)
    return pd.Series(cf_hourly, index=idx, name="hydro_cf")


def synthesize_wind_cf(region: str, year: int = 2025) -> pd.Series:
    """
    Placeholder wind CF for Malaysia.
    Malaysia has very low wind speeds (annual mean ~3-4 m/s).
    Only Sabah coastal and some Peninsular sites are viable.
    """
    idx = pd.date_range(f"{year}-01-01", periods=8760, freq="h")
    # Low wind CF: ~15% annual mean with diurnal variation
    if region == "sarawak":
        mean_cf = 0.05  # Very low inland Sarawak
    elif region == "sabah":
        mean_cf = 0.18  # Coastal Sabah
    else:
        mean_cf = 0.12  # Peninsular coastal sites

    hour_of_day = np.arange(8760) % 24
    # Wind slightly higher at night/early morning
    diurnal = 1.0 - 0.3 * np.sin(2 * np.pi * (hour_of_day - 6) / 24)
    cf = mean_cf * diurnal
    cf += np.random.normal(0, 0.02, 8760)
    cf = np.clip(cf, 0, 0.6)
    return pd.Series(cf, index=idx, name="wind_cf")


def fetch_and_save_region(region: str, year: int = 2025, use_nasa: bool = True) -> None:
    print(f"\nProcessing region: {region} (year={year})")
    lat = REGION_COORDS[region]["lat"]
    lon = REGION_COORDS[region]["lon"]

    # Solar
    solar_path = TIMESERIES_DIR / f"solar_cf_{region}_raw.csv"
    if not solar_path.exists():
        if use_nasa:
            try:
                solar = fetch_nasa_solar(lat, lon, year=year - 3)  # Use recent historical year
                solar.to_frame().to_csv(solar_path)
                print(f"  Saved: {solar_path}")
                time.sleep(1)  # Be polite to NASA API
            except Exception as e:
                print(f"  NASA API failed ({e}), synthesizing solar CF instead")
                use_nasa = False
        if not use_nasa:
            # Fallback: synthesize from Malaysia typical irradiance
            idx = pd.date_range(f"{year}-01-01", periods=8760, freq="h")
            hour = np.arange(8760) % 24
            day = np.arange(8760) // 24
            # Typical Malaysia: GHI ~5-6 kWh/m2/day, peak around noon
            ghi = np.where(
                (hour >= 6) & (hour <= 18),
                4.5 * np.sin(np.pi * (hour - 6) / 12) * (1 + 0.05 * np.sin(2 * np.pi * day / 365)),
                0,
            ) * 0.80 / 1.0  # CF
            ghi = np.clip(ghi, 0, 0.85)
            pd.Series(ghi, index=idx, name="solar_cf").to_frame().to_csv(solar_path)
            print(f"  Saved synthetic solar: {solar_path}")
    else:
        print(f"  Solar already exists: {solar_path}")

    # Demand
    demand_path = TIMESERIES_DIR / f"demand_{region}_raw.csv"
    if not demand_path.exists():
        demand = synthesize_demand(region, year=year)
        demand.to_frame().to_csv(demand_path)
        print(f"  Saved: {demand_path}")
    else:
        print(f"  Demand already exists: {demand_path}")

    # Hydro CF
    hydro_path = TIMESERIES_DIR / f"hydro_cf_{region}_raw.csv"
    if not hydro_path.exists():
        hydro = synthesize_hydro_cf(region, year=year)
        hydro.to_frame().to_csv(hydro_path)
        print(f"  Saved: {hydro_path}")
    else:
        print(f"  Hydro already exists: {hydro_path}")

    # Wind CF
    wind_path = TIMESERIES_DIR / f"wind_cf_{region}_raw.csv"
    if not wind_path.exists():
        wind = synthesize_wind_cf(region, year=year)
        wind.to_frame().to_csv(wind_path)
        print(f"  Saved: {wind_path}")
    else:
        print(f"  Wind already exists: {wind_path}")


def main():
    parser = argparse.ArgumentParser(description="Fetch/synthesize input time series data")
    parser.add_argument("--region", choices=["peninsular", "sabah", "sarawak", "all"], default="all")
    parser.add_argument("--year", type=int, default=2025)
    parser.add_argument("--no-nasa", action="store_true", help="Skip NASA API, use synthetic solar")
    args = parser.parse_args()

    regions = ["peninsular", "sabah", "sarawak"] if args.region == "all" else [args.region]
    for r in regions:
        fetch_and_save_region(r, year=args.year, use_nasa=not args.no_nasa)

    print("\nAll data fetched. Run time_cluster.py next.")


if __name__ == "__main__":
    main()
