"""
Page 5 — Data
View and edit annual data schedules and upload 8760-hour time series profiles.
"""

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="Data | Malaysia Energy Model", layout="wide")
st.title("Input Data")
st.markdown("View and edit annual schedules and hourly time series profiles used by the model.")

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

DATA_DIR = PROJECT_ROOT / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)
TIMESERIES_DIR = PROJECT_ROOT / "model" / "timeseries"
ANNUAL_INPUTS_PATH = DATA_DIR / "annual_inputs.json"

REGIONS = ["peninsular", "sabah", "sarawak"]
REGION_LABELS = {
    "peninsular": "Peninsular Malaysia (TNB)",
    "sabah": "Sabah (SESB)",
    "sarawak": "Sarawak (SEB)",
}
MILESTONE_YEARS = [2025, 2030, 2035, 2040, 2045, 2050]

TECH_LABELS = {
    "coal_existing":        "Coal existing (MW)",
    "gas_ccgt":             "Gas CCGT existing floor (MW)",
    "gas_ocgt":             "Gas OCGT existing floor (MW)",
    "hydro_large_existing": "Large hydro existing (MW)",
    "hydro_ror_min":        "Run-of-river hydro min (MW)",
    "solar_utility_max":    "Solar utility max (MW)",
    "solar_rooftop_max":    "Solar rooftop max (MW)",
    "wind_onshore_max":     "Wind onshore max (MW)",
    "biomass_max":          "Biomass max (MW)",
    "battery_max":          "Battery storage max (MW)",
    "diesel_genset":        "Diesel genset (MW)",
}
REVERSE_TECH_LABELS = {v: k for k, v in TECH_LABELS.items()}

# ---------------------------------------------------------------------------
# Defaults derived from location YAMLs and build_model.py
# ---------------------------------------------------------------------------
DEFAULTS = {
    "demand_peak_mw": {
        "peninsular": {2025: 20208, 2030: 24001, 2035: 28506, 2040: 33856, 2045: 40210, 2050: 47757},
        "sabah":      {2025: 1262,  2030: 1573,  2035: 1960,  2040: 2443,  2045: 3044,  2050: 3793},
        "sarawak":    {2025: 3357,  2030: 4084,  2035: 4969,  2040: 6046,  2045: 7356,  2050: 8949},
    },
    "capacity": {
        "peninsular": {
            "coal_existing":        [12800, 10000, 7000,  3000,  0,     0],
            "gas_ccgt":             [10980, 10980, 10980, 10980, 10980, 10980],
            "gas_ocgt":             [1000,  1000,  1000,  1000,  1000,  1000],
            "hydro_large_existing": [1300,  1300,  1300,  1300,  1300,  1300],
            "hydro_ror_min":        [0,     0,     0,     0,     0,     0],
            "solar_utility_max":    [7000,  15000, 22000, 30000, 37000, 40000],
            "solar_rooftop_max":    [1050,  2250,  3300,  4500,  5550,  6000],
            "wind_onshore_max":     [500,   500,   500,   500,   500,   500],
            "biomass_max":          [2000,  2000,  2000,  2000,  2000,  2000],
            "battery_max":          [10000, 10000, 10000, 10000, 10000, 10000],
            "diesel_genset":        [0,     0,     0,     0,     0,     0],
        },
        "sabah": {
            "coal_existing":        [0,    0,    0,    0,    0,    0],
            "gas_ccgt":             [900,  900,  900,  900,  900,  900],
            "gas_ocgt":             [300,  300,  300,  300,  300,  300],
            "hydro_large_existing": [140,  140,  140,  140,  140,  140],
            "hydro_ror_min":        [0,    0,    0,    0,    0,    0],
            "solar_utility_max":    [800,  1500, 2000, 2500, 3000, 3000],
            "solar_rooftop_max":    [120,  225,  300,  375,  450,  450],
            "wind_onshore_max":     [300,  300,  300,  300,  300,  300],
            "biomass_max":          [300,  300,  300,  300,  300,  300],
            "battery_max":          [1500, 1500, 1500, 1500, 1500, 1500],
            "diesel_genset":        [150,  150,  150,  100,  50,   0],
        },
        "sarawak": {
            "coal_existing":        [920,  920,  500,  0,    0,    0],
            "gas_ccgt":             [1200, 1200, 1200, 1200, 1200, 1200],
            "gas_ocgt":             [0,    0,    0,    0,    0,    0],
            "hydro_large_existing": [3452, 3452, 3452, 3452, 3452, 3452],
            "hydro_ror_min":        [1285, 1285, 1285, 1285, 1285, 1285],
            "solar_utility_max":    [500,  2000, 4000, 6000, 8000, 10000],
            "solar_rooftop_max":    [75,   300,  600,  900,  1200, 1500],
            "wind_onshore_max":     [0,    0,    0,    0,    0,    0],
            "biomass_max":          [0,    0,    0,    0,    0,    0],
            "battery_max":          [5000, 5000, 5000, 5000, 5000, 5000],
            "diesel_genset":        [100,  100,  100,  50,   0,    0],
        },
    },
    "fuel_prices": {
        "gas_usd_gj":     [10.0, 10.5, 11.0, 11.5, 12.0, 12.5],
        "coal_usd_tonne": [100,  95,   90,   85,   80,   75],
    },
    "hvdc_mw": [0, 0, 500, 1000, 1000, 1000],
}


def _defaults_as_json() -> dict:
    return {
        "demand_peak_mw": {
            r: {str(yr): DEFAULTS["demand_peak_mw"][r][yr] for yr in MILESTONE_YEARS}
            for r in REGIONS
        },
        "capacity": {
            r: {
                tech: {str(yr): DEFAULTS["capacity"][r][tech][i] for i, yr in enumerate(MILESTONE_YEARS)}
                for tech in DEFAULTS["capacity"][r]
            }
            for r in REGIONS
        },
        "fuel_prices": {
            str(yr): {
                "gas_usd_gj":     DEFAULTS["fuel_prices"]["gas_usd_gj"][i],
                "coal_usd_tonne": DEFAULTS["fuel_prices"]["coal_usd_tonne"][i],
            }
            for i, yr in enumerate(MILESTONE_YEARS)
        },
        "hvdc_mw": {str(yr): DEFAULTS["hvdc_mw"][i] for i, yr in enumerate(MILESTONE_YEARS)},
    }


def load_annual_inputs() -> dict:
    if ANNUAL_INPUTS_PATH.exists():
        with open(ANNUAL_INPUTS_PATH) as f:
            return json.load(f)
    return _defaults_as_json()


def save_annual_inputs(data: dict):
    with open(ANNUAL_INPUTS_PATH, "w") as f:
        json.dump(data, f, indent=2)


# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------
annual_data = load_annual_inputs()

tab_annual, tab_hourly = st.tabs(["Annual Data", "Hourly Profiles"])

# ===========================================================================
# TAB 1 — ANNUAL DATA
# ===========================================================================
with tab_annual:
    st.markdown("Edit annual schedules then click **Save** to apply to the next model run.")

    # ---- Peak Demand -------------------------------------------------------
    st.subheader("Peak Demand (MW)")
    demand_df = pd.DataFrame(
        {REGION_LABELS[r]: [annual_data["demand_peak_mw"][r][str(yr)] for yr in MILESTONE_YEARS]
         for r in REGIONS},
        index=MILESTONE_YEARS,
    )
    demand_df.index.name = "Year"
    edited_demand = st.data_editor(
        demand_df, use_container_width=True,
        column_config={c: st.column_config.NumberColumn(c, min_value=0, step=100, format="%d MW")
                       for c in demand_df.columns},
        key="de_demand",
    )

    # ---- Installed / Max Capacity ------------------------------------------
    st.subheader("Installed & Maximum Capacity (MW)")
    st.caption(
        "Existing fleet rows set the minimum (locked) capacity. "
        "Max rows cap new investment for that milestone year."
    )
    region_tabs = st.tabs([REGION_LABELS[r] for r in REGIONS])
    edited_caps = {}
    for i, region in enumerate(REGIONS):
        with region_tabs[i]:
            cap_data = annual_data["capacity"][region]
            cap_df = pd.DataFrame(
                {TECH_LABELS.get(tech, tech): [cap_data[tech][str(yr)] for yr in MILESTONE_YEARS]
                 for tech in cap_data},
                index=MILESTONE_YEARS,
            )
            cap_df.index.name = "Year"
            edited_caps[region] = st.data_editor(
                cap_df, use_container_width=True,
                column_config={c: st.column_config.NumberColumn(c, min_value=0, step=50, format="%d MW")
                               for c in cap_df.columns},
                key=f"de_cap_{region}",
            )

    # ---- Fuel Prices -------------------------------------------------------
    st.subheader("Fuel Prices")
    fuel_df = pd.DataFrame(
        {
            "Gas (USD/GJ)":     [annual_data["fuel_prices"][str(yr)]["gas_usd_gj"] for yr in MILESTONE_YEARS],
            "Coal (USD/tonne)": [annual_data["fuel_prices"][str(yr)]["coal_usd_tonne"] for yr in MILESTONE_YEARS],
        },
        index=MILESTONE_YEARS,
    )
    fuel_df.index.name = "Year"
    edited_fuel = st.data_editor(
        fuel_df, use_container_width=True,
        column_config={
            "Gas (USD/GJ)":     st.column_config.NumberColumn("Gas (USD/GJ)", min_value=0.0, step=0.5, format="%.1f"),
            "Coal (USD/tonne)": st.column_config.NumberColumn("Coal (USD/tonne)", min_value=0, step=5, format="%d"),
        },
        key="de_fuel",
    )

    # ---- HVDC Transmission -------------------------------------------------
    st.subheader("HVDC Sabah–Sarawak Transmission Capacity (MW)")
    hvdc_df = pd.DataFrame(
        {"HVDC Capacity (MW)": [annual_data["hvdc_mw"][str(yr)] for yr in MILESTONE_YEARS]},
        index=MILESTONE_YEARS,
    )
    hvdc_df.index.name = "Year"
    edited_hvdc = st.data_editor(
        hvdc_df, use_container_width=True,
        column_config={"HVDC Capacity (MW)": st.column_config.NumberColumn(
            "HVDC Capacity (MW)", min_value=0, step=100, format="%d MW")},
        key="de_hvdc",
    )

    # ---- Save --------------------------------------------------------------
    if st.button("Save Annual Data", type="primary"):
        new_data: dict = {
            "demand_peak_mw": {
                r: {str(yr): int(edited_demand.loc[yr, REGION_LABELS[r]]) for yr in MILESTONE_YEARS}
                for r in REGIONS
            },
            "capacity": {},
            "fuel_prices": {
                str(yr): {
                    "gas_usd_gj":     float(edited_fuel.loc[yr, "Gas (USD/GJ)"]),
                    "coal_usd_tonne": float(edited_fuel.loc[yr, "Coal (USD/tonne)"]),
                }
                for yr in MILESTONE_YEARS
            },
            "hvdc_mw": {
                str(yr): int(edited_hvdc.loc[yr, "HVDC Capacity (MW)"]) for yr in MILESTONE_YEARS
            },
        }
        for region in REGIONS:
            new_data["capacity"][region] = {}
            for col in edited_caps[region].columns:
                tech_key = REVERSE_TECH_LABELS.get(col, col)
                new_data["capacity"][region][tech_key] = {
                    str(yr): int(edited_caps[region].loc[yr, col]) for yr in MILESTONE_YEARS
                }
        save_annual_inputs(new_data)
        st.success("Annual data saved. Changes apply to the next model run.")
        st.rerun()


# ===========================================================================
# TAB 2 — HOURLY PROFILES
# ===========================================================================
with tab_hourly:
    st.markdown(
        "View and replace the 8,760-hour raw profiles (demand and renewable capacity factors). "
        "After uploading, click **Re-cluster** to regenerate the representative days used by Calliope."
    )

    col_r, col_p = st.columns(2)
    with col_r:
        sel_region = st.selectbox(
            "Region", options=REGIONS, format_func=lambda x: REGION_LABELS[x], key="hp_region"
        )
    with col_p:
        profile_type = st.selectbox(
            "Profile type",
            options=["demand", "solar_cf", "wind_cf", "hydro_cf"],
            format_func=lambda x: {
                "demand":   "Demand (MW, negative = consumption)",
                "solar_cf": "Solar capacity factor (0–1)",
                "wind_cf":  "Wind capacity factor (0–1)",
                "hydro_cf": "Hydro capacity factor (0–1)",
            }[x],
            key="hp_profile",
        )

    raw_path = TIMESERIES_DIR / f"{profile_type}_{sel_region}_raw.csv"

    # ---- Current profile ---------------------------------------------------
    if raw_path.exists():
        df_raw = pd.read_csv(raw_path, index_col=0, parse_dates=True)
        values = df_raw.iloc[:, 0].values
        col_name = df_raw.columns[0]

        st.markdown(f"**Current file:** `{raw_path.name}` — {len(values):,} rows")

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Min",  f"{values.min():.3f}")
        m2.metric("Max",  f"{values.max():.3f}")
        m3.metric("Mean", f"{values.mean():.3f}")
        m4.metric("Rows", f"{len(values):,}")

        _PLOT_LAYOUT = dict(
            plot_bgcolor="#1e2130", paper_bgcolor="#1e2130",
            font=dict(color="#e0e0e0"), margin=dict(t=30, b=40),
            height=280,
        )
        _GRID = dict(gridcolor="#2e3450", zerolinecolor="#2e3450")

        ptab_full, ptab_week = st.tabs(["Full Year", "First Week (168 h)"])
        with ptab_full:
            fig = go.Figure(go.Scatter(
                x=list(range(len(values))), y=values,
                mode="lines", line=dict(width=0.7, color="#f5c518"),
            ))
            fig.update_layout(xaxis_title="Hour of year", yaxis_title=col_name, **_PLOT_LAYOUT)
            fig.update_xaxes(**_GRID)
            fig.update_yaxes(**_GRID)
            st.plotly_chart(fig, use_container_width=True)

        with ptab_week:
            fig2 = go.Figure(go.Scatter(
                x=list(range(168)), y=values[:168],
                mode="lines", line=dict(width=1.5, color="#f5c518"),
            ))
            fig2.update_layout(xaxis_title="Hour", yaxis_title=col_name, **_PLOT_LAYOUT)
            fig2.update_xaxes(**_GRID)
            fig2.update_yaxes(**_GRID)
            st.plotly_chart(fig2, use_container_width=True)

        # Download current
        st.download_button(
            "Download current CSV",
            data=df_raw.to_csv().encode(),
            file_name=raw_path.name,
            mime="text/csv",
        )
    else:
        st.warning(f"No existing file at `{raw_path.name}`.")

    # ---- Upload new profile ------------------------------------------------
    st.markdown("---")
    st.subheader("Upload New Profile")
    st.caption(
        "CSV: first column = datetime index, second column = values. "
        "Exactly **8,760 rows** required. "
        "Demand: negative MW. Capacity factors: 0–1."
    )

    uploaded = st.file_uploader(
        f"Upload replacement for `{profile_type}_{sel_region}_raw.csv`",
        type=["csv"],
        key=f"up_{sel_region}_{profile_type}",
    )

    if uploaded is not None:
        try:
            df_up = pd.read_csv(uploaded, index_col=0, parse_dates=True)
            vals_up = df_up.iloc[:, 0].values
            n_rows = len(vals_up)
            col_up = df_up.columns[0]

            errors = []
            if n_rows != 8760:
                errors.append(f"Expected 8,760 rows, got {n_rows:,}.")
            if profile_type in ("solar_cf", "wind_cf", "hydro_cf"):
                if vals_up.min() < -0.001 or vals_up.max() > 1.001:
                    errors.append(
                        f"Capacity factors must be 0–1 (got min={vals_up.min():.3f}, max={vals_up.max():.3f})."
                    )
            if profile_type == "demand" and vals_up.max() > 0:
                st.warning(
                    "Demand values should be negative (consumption). "
                    "Detected positive values — confirm this is intentional."
                )

            if errors:
                for e in errors:
                    st.error(e)
            else:
                st.success(
                    f"Valid: {n_rows:,} rows | min={vals_up.min():.3f} | "
                    f"max={vals_up.max():.3f} | mean={vals_up.mean():.3f}"
                )

                fig_up = go.Figure(go.Scatter(
                    x=list(range(n_rows)), y=vals_up,
                    mode="lines", line=dict(width=0.7, color="#8bc34a"),
                ))
                fig_up.update_layout(
                    title="Uploaded profile preview",
                    xaxis_title="Hour of year", yaxis_title=col_up,
                    plot_bgcolor="#1e2130", paper_bgcolor="#1e2130",
                    font=dict(color="#e0e0e0"), margin=dict(t=40, b=40), height=250,
                )
                fig_up.update_xaxes(gridcolor="#2e3450", zerolinecolor="#2e3450")
                fig_up.update_yaxes(gridcolor="#2e3450", zerolinecolor="#2e3450")
                st.plotly_chart(fig_up, use_container_width=True)

                col_sv, col_rc = st.columns(2)
                with col_sv:
                    if st.button("Save to model", type="primary", key="btn_save_profile"):
                        df_up.to_csv(raw_path)
                        st.success(f"Saved `{raw_path.name}`. Click Re-cluster to apply.")

                with col_rc:
                    if st.button("Re-cluster time series", key="btn_recluster"):
                        with st.spinner(f"Re-clustering {sel_region} ({profile_type})..."):
                            result = subprocess.run(
                                [sys.executable,
                                 str(PROJECT_ROOT / "scripts" / "time_cluster.py"),
                                 "--region", sel_region],
                                capture_output=True, text=True, cwd=str(PROJECT_ROOT),
                            )
                        if result.returncode == 0:
                            st.success("Re-clustering complete. Run the model to use updated profiles.")
                        else:
                            st.error(f"Re-clustering failed:\n```\n{result.stderr[-2000:]}\n```")

        except Exception as e:
            st.error(f"Could not parse CSV: {e}")
