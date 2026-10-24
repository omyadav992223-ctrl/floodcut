"""State and data loaders for FloodCut dashboard.

Provides cached access to the drainage network, rainfall scenarios,
observed rainfall datasets, and geospatial layers.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import streamlit as st

from floodcut.geo_loader import (
    GeoLayer,
    load_imd_grid_points,
    load_waterlogging_hotspots,
)
from floodcut.io import load_network, load_scenarios
from floodcut.models import DrainageNetwork, RainfallScenario
from floodcut.provenance import OPENCITY_IMD_MONTHLY
from floodcut.rainfall_loader import (
    RainfallDataset,
    load_rainfall_csv,
    load_synthetic_fixture,
)

BASE_DIR = Path(__file__).resolve().parents[3]
DATA_SIM_DIR = BASE_DIR / "data" / "simulated"
DATA_REAL_DIR = BASE_DIR / "data" / "real"


@st.cache_data
def get_drainage_network(path: Optional[str] = None) -> DrainageNetwork:
    """Load and cache the drainage network."""
    p = Path(path) if path else (DATA_SIM_DIR / "mumbai_sim_network.json")
    return load_network(p)


@st.cache_data
def get_scenarios(path: Optional[str] = None) -> dict[str, RainfallScenario]:
    """Load and cache standard rainfall scenarios."""
    p = Path(path) if path else (DATA_SIM_DIR / "rainfall_scenarios.json")
    return load_scenarios(p)


@st.cache_data
def get_rainfall_dataset() -> Tuple[RainfallDataset, bool]:
    """Load observed IMD rainfall if available; fallback to synthetic fixture.

    Returns (dataset, is_real_dataset).
    """
    real_csv = DATA_REAL_DIR / "imd_mumbai_monthly_rainfall_1901_2021.csv"
    if real_csv.exists():
        try:
            ds = load_rainfall_csv(
                real_csv,
                dataset_record=OPENCITY_IMD_MONTHLY,
                station_id="IMD_GRID_19.00N_72.75E",
                provenance="REAL:IMD_GRID",
            )
            return ds, True
        except Exception:
            pass
    # Fallback to offline synthetic fixture
    return load_synthetic_fixture(), False


@st.cache_data
def get_imd_grid_layer() -> GeoLayer:
    """Return the IMD grid point geospatial layer."""
    return load_imd_grid_points()


@st.cache_data
def get_hotspot_layer() -> GeoLayer:
    """Return the waterlogging hotspots geospatial layer."""
    return load_waterlogging_hotspots()
