# Malaysia Energy System Model 2025–2050

A **Calliope**-based long-term capacity expansion model covering Malaysia's three power grids:
**Peninsular Malaysia** (TNB), **Sabah** (SESB), and **Sarawak** (Sarawak Energy) — from 2025 to 2050.

Aligned with Malaysia's **National Energy Transition Roadmap (NETR)**:
31% RE by 2025 → 40% by 2035 → **70% RE by 2050**.

---

## Features

- Multi-region myopic rolling optimisation across 6 milestone years (2025–2050)
- Three physically separate grids modelled independently, with optional Sabah–Sarawak HVDC interconnector
- Streamlit dashboard for interactive scenario configuration, simulation launch, and results exploration
- Time-series clustering (24 representative days from 8,760 hourly timesteps) for tractable LP solves
- Technology cost learning curves sourced from IRENA 2023

---

## Project Structure

```
malaysia_calliope/
├── model/
│   ├── techs/           # Technology definitions (supply, storage, demand, transmission)
│   ├── locations/       # Regional node definitions (peninsular, sabah, sarawak)
│   ├── timeseries/      # Clustered hourly time-series CSVs
│   ├── scenarios/       # Scenario YAML overrides
│   └── overrides/       # Optional policy & cost overrides
├── scripts/
│   ├── fetch_data.py    # NASA POWER API + synthetic profile generation
│   ├── time_cluster.py  # K-means time clustering (24 representative days)
│   ├── build_model.py   # Assemble Calliope model per region + milestone year
│   ├── myopic_chain.py  # Myopic rolling optimisation chain
│   ├── run_parallel.py  # Parallel regional runs
│   ├── postprocess.py   # Post-processing: NetCDF → summary DataFrames
│   └── time_cluster.py  # Temporal aggregation utility
├── dashboard/
│   ├── app.py           # Streamlit entry point
│   ├── pages/           # Dashboard pages: Inputs, Scenarios, Run, Results, Data
│   └── utils/           # Chart helpers and config writers
├── data/
│   ├── raw/             # Raw input data (capacity, fuel prices, demand)
│   └── processed/       # Cleaned inputs ready for model ingestion
├── results/
│   ├── baseline/        # Baseline scenario outputs (NetCDF + JSON summaries)
│   └── netr_target/     # NETR-aligned scenario outputs
└── requirements.txt
```

---

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Install a solver

The model requires a linear programming solver.

**GLPK** (free, works out-of-the-box):
```bash
conda install -c conda-forge glpk
```

**HiGHS** (faster for large models — recommended):
```bash
pip install highspy
```

> **macOS note:** After installing GLPK via conda, remove quarantine with:
> ```bash
> xattr -d com.apple.quarantine $(which glpsol)
> ```

### 3. Fetch and prepare input data

```bash
# Use synthetic solar profiles (no NASA API key needed)
python scripts/fetch_data.py --no-nasa

# Generate 24 representative days via k-means clustering
python scripts/time_cluster.py
```

### 4. Run a scenario

```bash
# Baseline scenario — all three regions in parallel
python scripts/run_parallel.py --scenario baseline

# NETR target scenario
python scripts/run_parallel.py --scenario netr_target
```

### 5. Launch the dashboard

```bash
streamlit run dashboard/app.py
```

Open [http://localhost:8501](http://localhost:8501) in your browser.

---

## The Three Grids

| Region | Operator | 2024 Capacity | Generation Mix |
|--------|----------|--------------|----------------|
| Peninsular Malaysia | TNB | 26,152 MW | Coal 49%, Gas 42%, Hydro 5%, Solar 5% |
| Sabah | SESB | ~1,500 MW | Gas >80%, Hydro 4%, Solar growing |
| Sarawak | Sarawak Energy | 5,745 MW | Hydro 62%, Gas 21%, Coal 16% |

The three grids are **physically separate** with no existing interconnection. The model includes a **Sabah–Sarawak HVDC cable** as an investable option from 2035.

---

## Scenarios

| Scenario | Description |
|----------|-------------|
| `baseline` | Business as usual. No new coal post-2030. Moderate RE growth. |
| `netr_target` | NETR-aligned. 70% RE by 2050. Coal phased out by 2040. Carbon price trajectory applied. |
| `accelerated_re` | Aggressive decarbonisation. Coal exits by 2035. 80%+ RE by 2050. |

---

## Dashboard Pages

| Page | Description |
|------|-------------|
| **Inputs** | Set demand growth, technology CAPEX/OPEX, fuel prices, carbon price, and RE targets |
| **Scenarios** | Choose a base scenario and configure optional overrides |
| **Run** | Launch simulation and monitor progress |
| **Results** | Explore capacity mix, generation dispatch, system costs, emissions, and battery KPIs |
| **Data** | Browse and download raw model inputs and outputs |

---

## Runtime

| Setting | Time |
|---------|------|
| Temporal resolution | 24 representative days (k-means from 8,760 hourly timesteps) |
| Regions per run | 3 (solved in parallel) |
| Milestone years | 6 (2025, 2030, 2035, 2040, 2045, 2050) |
| Estimated runtime (GLPK, 8-core laptop) | 60–90 min per scenario |
| Estimated runtime (HiGHS/Gurobi) | 20–40 min per scenario |

---

## Data Sources

| Data | Source |
|------|--------|
| Installed capacity | Energy Commission Malaysia Statistics Handbook |
| Solar irradiance | NASA POWER API (power.larc.nasa.gov) |
| Technology costs | IRENA Renewable Power Generation Costs 2023 |
| Fuel prices | World Bank Commodity Markets |
| Policy targets | Malaysia NETR (Ministry of Economy, 2023) |

---

## Requirements

- Python 3.8+
- `calliope >= 0.6.10`
- `streamlit >= 1.30`
- `plotly`, `xarray`, `pandas`, `scikit-learn`, `numpy`, `scipy`, `netCDF4`, `pyyaml`, `requests`
- LP solver: GLPK (via conda) or HiGHS (`pip install highspy`)

---

## License

MIT License. See [LICENSE](LICENSE) for details.
