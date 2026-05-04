"""
time_cluster.py
---------------
Generate 24 representative days from full-year hourly time series using k-means clustering.

Usage:
    python scripts/time_cluster.py --region peninsular --year 2025

Outputs:
    model/timeseries/demand_{region}_clustered.csv
    model/timeseries/solar_cf_{region}_clustered.csv
    model/timeseries/wind_cf_{region}_clustered.csv
    model/timeseries/hydro_cf_{region}_clustered.csv
    model/timeseries/cluster_weights_{region}.csv
"""

import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans

PROJECT_ROOT = Path(__file__).parent.parent
TIMESERIES_DIR = PROJECT_ROOT / "model" / "timeseries"
N_CLUSTERS = 24


def load_raw_timeseries(region: str) -> pd.DataFrame:
    """Load all raw hourly time series for a region into a single DataFrame."""
    files = {
        "demand": TIMESERIES_DIR / f"demand_{region}_raw.csv",
        "solar_cf": TIMESERIES_DIR / f"solar_cf_{region}_raw.csv",
        "wind_cf": TIMESERIES_DIR / f"wind_cf_{region}_raw.csv",
        "hydro_cf": TIMESERIES_DIR / f"hydro_cf_{region}_raw.csv",
    }
    dfs = {}
    for key, path in files.items():
        if path.exists():
            dfs[key] = pd.read_csv(path, index_col=0, parse_dates=True).squeeze()
        else:
            print(f"  WARNING: {path} not found — skipping {key}")
    return pd.DataFrame(dfs)


def reshape_to_daily(df: pd.DataFrame) -> np.ndarray:
    """Reshape hourly DataFrame (8760 x n_vars) to daily matrix (365 x 24*n_vars)."""
    n_hours = len(df)
    n_days = n_hours // 24
    n_vars = df.shape[1]
    # Reshape: (365, 24, n_vars) -> (365, 24*n_vars)
    return df.values[:n_days * 24].reshape(n_days, 24 * n_vars)


def run_kmeans(daily_matrix: np.ndarray, n_clusters: int = N_CLUSTERS, random_state: int = 42):
    """Cluster daily profiles and return labels + cluster centres."""
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(daily_matrix)
    km = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10, max_iter=300)
    labels = km.fit_predict(scaled)
    return labels, scaler.inverse_transform(km.cluster_centers_)


def build_clustered_timeseries(
    labels: np.ndarray,
    centres: np.ndarray,
    columns: list,
    year: int = 2025,
) -> "tuple[pd.DataFrame, pd.DataFrame]":
    """
    Build a clustered time series DataFrame from cluster centres and weights.

    Returns:
        ts: DataFrame with representative days (n_clusters * 24 rows)
        weights: DataFrame with day weights (how many actual days each rep day represents)
    """
    n_clusters, n_cols_flat = centres.shape
    n_vars = len(columns)
    # centres shape: (n_clusters, 24 * n_vars)
    centres_3d = centres.reshape(n_clusters, 24, n_vars)

    # Build timestamp index: use Jan 1 as base, stack n_clusters days
    base = pd.Timestamp(f"{year}-01-01")
    dfs = []
    for i in range(n_clusters):
        day_start = base + pd.Timedelta(days=i)
        idx = pd.date_range(day_start, periods=24, freq="h")
        day_df = pd.DataFrame(centres_3d[i], index=idx, columns=columns)
        dfs.append(day_df)
    ts = pd.concat(dfs)

    # Compute weights (number of original days mapped to each cluster)
    unique, counts = np.unique(labels, return_counts=True)
    weight_map = dict(zip(unique, counts))
    weights_list = [weight_map.get(i, 0) for i in range(n_clusters)]
    weights = pd.DataFrame({
        "cluster": range(n_clusters),
        "weight": weights_list,
        "start_date": [base + pd.Timedelta(days=i) for i in range(n_clusters)],
    })
    return ts, weights


def split_and_save(ts: pd.DataFrame, weights: pd.DataFrame, region: str) -> None:
    """Split clustered time series back into per-variable files."""
    for col in ts.columns:
        out_path = TIMESERIES_DIR / f"{col}_{region}.csv"
        ts[[col]].to_csv(out_path)
        print(f"  Saved: {out_path}")

    weights_path = TIMESERIES_DIR / f"cluster_weights_{region}.csv"
    weights.to_csv(weights_path, index=False)
    print(f"  Saved weights: {weights_path}")


def cluster_region(region: str, year: int = 2025) -> None:
    print(f"\nClustering region: {region} (year={year}, n_clusters={N_CLUSTERS})")
    df = load_raw_timeseries(region)
    if df.empty:
        print(f"  ERROR: No data found for {region}. Run fetch_data.py first.")
        return

    daily = reshape_to_daily(df)
    print(f"  Daily matrix shape: {daily.shape}")

    labels, centres = run_kmeans(daily)
    print(f"  K-means complete. Unique clusters: {len(np.unique(labels))}")

    ts, weights = build_clustered_timeseries(labels, centres, list(df.columns), year=year)
    split_and_save(ts, weights, region)
    print(f"  Clustering done for {region}.")


def main():
    parser = argparse.ArgumentParser(description="Cluster hourly time series to representative days")
    parser.add_argument("--region", choices=["peninsular", "sabah", "sarawak", "all"], default="all")
    parser.add_argument("--year", type=int, default=2025, help="Base year for timestep index")
    args = parser.parse_args()

    regions = ["peninsular", "sabah", "sarawak"] if args.region == "all" else [args.region]
    for r in regions:
        cluster_region(r, year=args.year)


if __name__ == "__main__":
    main()
