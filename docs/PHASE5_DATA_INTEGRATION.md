# Phase 5: Real Data Integration

## Overview

Phase 5 adds a documented real-data integration layer to FloodCut. It:

1. Integrates a real IMD/OpenCity monthly rainfall dataset for Mumbai (1901–2021).
2. Introduces a consistent provenance labelling system to distinguish **real**, **derived**, **assumed**, and **synthetic** data.
3. Implements robust data-loading and preprocessing pipelines with full validation and transparent error handling.
4. Adds a georeferenced layer of IMD grid points and approximate waterlogging hotspots.
5. Documents exactly what data is available, what could not be obtained, and what assumptions were made.
6. Adds 60+ offline automated tests covering all new functionality and Phase 3–4 regression.

> **This project is a prototype simulation, NOT a validated flood prediction system.** See [Known Limitations](#known-limitations).

---

## Data sources

### 1. IMD Monthly Rainfall — OpenCity India

| Field | Value |
|---|---|
| **Provenance label** | `REAL:IMD_GRID` |
| **Dataset name** | Mumbai Monthly Rainfall 1901–2021 (IMD Grid Point) |
| **Source organisation** | India Meteorological Department (IMD), via OpenCity India |
| **Dataset landing page** | https://data.opencity.in/dataset/mumbai-rainfall-data |
| **Direct CSV URL** | https://data.opencity.in/dataset/c0e2f1ba-848a-4716-982b-d42636c44bf5/resource/3b1fcae7-b24f-431f-a259-64b56077acd5/download/6b7b0aed-e31d-433b-ba6d-b17b08a000df.csv |
| **Retrieval date** | 2026-10-23 |
| **Geographic coverage** | Single IMD 0.25° grid cell centred at 19.00° N, 72.75° E (Greater Mumbai region) |
| **Observation period** | 1901–2021 (121 years) |
| **Format** | CSV (one row per year, monthly columns in mm) |
| **License** | OpenCity India; IMD data for non-commercial research. Verify current terms at the dataset URL before redistribution. |

**Bundled file location:** `data/real/imd_mumbai_monthly_rainfall_1901_2021.csv`

### 2. IMD Grid Point Location

| Field | Value |
|---|---|
| **Provenance label** | `REAL:IMD_GRID` |
| **Description** | Grid-cell centre coordinate derived from the OpenCity dataset metadata |
| **Coordinates** | 19.00° N, 72.75° E (WGS84) |
| **Note** | This is a 0.25° × 0.25° grid-cell centre, NOT an AWS or raingauge station |

### 3. Mumbai Waterlogging Hotspots (approximate)

| Field | Value |
|---|---|
| **Provenance label** | `DERIVED:GEOCODED` |
| **Description** | Small curated list of well-known waterlogging junctions compiled from MCGM civic reports and media sources, geocoded to approximate ward coordinates |
| **Source URL** | https://portal.mcgm.gov.in/ (publicly accessible) |
| **Accuracy** | ± ~500 m (named junctions / ward centroids, not precise surveys) |
| **Note** | NOT flood-depth measurements. Presence on this list indicates chronic reporting, not a confirmed flood record for any specific event. |

---

## Data not obtained and why

| Data type | Reason | What to do if you have it |
|---|---|---|
| **IMD raingauge station data** (Colaba / Santacruz daily) | Requires formal request via [IMD Data Service Portal](https://dsp.imdpune.gov.in/). Not publicly downloadable without registration. | Place CSV in `data/real/` and call `load_rainfall_csv(path, provenance="REAL:IMD_STATION", ...)` |
| **BMC/MCGM drain survey GIS** | Not published in open machine-readable format at the time of implementation. | If obtained, load with `geo_loader.load_geo_points_csv()` or extend `io.load_network()` |
| **Ward boundary polygons** (GeoJSON) | Polygon geometry requires geopandas (not a current dependency). Candidate sources: [BharatAtlas](https://bharatlas.com/view/wards_mumbai), [DataMeet](https://github.com/datameet/Municipal_Spatial_Data). | Add geopandas dependency and load separately; keep provenance `REAL:COMMUNITY_GIS` |
| **Real-time rainfall API** | Requires institutional login (IITM Mesonet, IMD API). | Use `load_rainfall_csv()` after downloading; label `REAL:IMD_REALTIME` |
| **Flood depth/inundation observations** | No accessible public dataset with georeferenced depth measurements for Mumbai found. | Required for model validation; DO NOT substitute model overflow as measured depth. |

---

## Provenance system

Every dataset and record is tagged with a provenance label defined in `src/floodcut/provenance.py`.

### Canonical labels

| Label | Meaning |
|---|---|
| `REAL` | Obtained from a verifiable external source with documented provenance |
| `REAL:IMD_GRID` | Specifically the IMD gridded dataset via OpenCity |
| `REAL:IMD_STATION` | Direct IMD raingauge / AWS station observation (not yet loaded) |
| `DERIVED` | Computed from a REAL dataset via a documented method |
| `DERIVED:TEMPORAL_DISAGGREGATE` | Monthly total converted to hourly intensity via documented assumption |
| `DERIVED:GEOCODED` | Named location geocoded to approximate coordinates |
| `ASSUMED` | Value decided by modeller without empirical source |
| `SYNTHETIC` | Artificial data for testing or algorithm demonstration |

### How to check provenance in code

```python
from floodcut.provenance import base_category, is_valid, REAL

# Check a record
print(record.provenance)          # e.g. "REAL:IMD_GRID"
print(base_category(record.provenance))  # "REAL"
print(dataset.dataset_record.is_real())  # True / False
```

---

## Data formats and schemas

### Monthly rainfall CSV (IMD/OpenCity format)

```
Year,Jan,Feb,Mar,April,May,June,July,Aug,Sept,Oct,Nov,Dec,Total
1901,13.1,0,0,3.9,...
```

- Month column names: case-insensitive; `April`, `June`, `July`, `Sept` accepted.
- Values: monthly accumulated rainfall in **millimetres (mm)**.
- Missing values: empty cell, `NA`, `N/A`, `nan` → treated as missing (`None`), **not** filled with zeros.
- Negative values: treated as missing with a warning.
- `Total` column: optional; silently retained.

### Geospatial CSV format (points layer)

```
id,name,lat,lon,<extra_columns>
```

- `lat` / `lon`: decimal degrees, WGS84 (EPSG:4326).
- Valid range: lat ∈ [−90, 90], lon ∈ [−180, 180].
- Points outside the Greater Mumbai bounding box (18.85–19.30° N, 72.75–73.05° E) trigger a `GeospatialDataWarning`.
- Extra columns preserved as `attributes` dict.

---

## Preprocessing and unit conversions

### Monthly mm → hourly intensity (mm/h)

The rational-method engine requires intensity in **mm/h**. Monthly totals cannot be used directly as mm/h — they are dimensionally and numerically different.

**Conversion used (DERIVED:TEMPORAL_DISAGGREGATE):**

```
intensity_mm_h = total_mm / hours_in_month
```

where `hours_in_month` is the standard hours per calendar month (e.g. July = 744 h).

**Critical caveats:**
1. The mean hourly intensity for even a heavy Mumbai monsoon month (e.g. 1000 mm/Jul) is ~1.3 mm/h — orders of magnitude below a design storm (50 mm/h).
2. This mean intensity MUST NOT be used as a design storm intensity for engineering purposes.
3. The function `rainfall_loader.monthly_mm_to_intensity_mm_h()` issues an explicit warning when called via `make_scenario_from_record()`.
4. For proper flood risk analysis, use IDF (Intensity-Duration-Frequency) curves from IMD.

See `rainfall_loader.py` for the documented implementation.

---

## Missing data and error handling

| Situation | Behaviour |
|---|---|
| File not found | `RainfallDataError` raised immediately |
| Missing required columns | `RainfallDataError` raised immediately |
| Missing or empty month cell | `None` stored; row retained; warning NOT issued for plain empty cells |
| Negative rainfall value | `None` stored; `RainfallDataWarning` issued |
| Non-numeric cell | `None` stored; `RainfallDataWarning` issued |
| Duplicate year | Second row dropped; `RainfallDataWarning` issued; skipped row recorded |
| Invalid year format | Row skipped; description added to `skipped_rows` |
| All monsoon months missing | `monsoon_total_mm()` returns `None` |
| Point outside Mumbai bbox | `GeospatialDataWarning` issued; point still loaded |
| Invalid coordinate range | Row skipped; description added to `skipped_rows` |

**Missing values are NEVER silently replaced with fabricated values.**

---

## CRS and spatial decisions

- All coordinates are in **WGS84 (EPSG:4326)**, decimal degrees.
- No coordinate transformations are applied in Phase 5.
- The IMD grid point is a cell centre, not a surveyed station location.
- Waterlogging hotspot coordinates are approximate (± ~500 m).
- The synthetic drainage-network junction coordinates are illustrative (not surveyed).

---

## Running the integration

### Load the real rainfall data

```python
from floodcut.rainfall_loader import load_rainfall_csv
from floodcut.provenance import OPENCITY_IMD_MONTHLY

ds = load_rainfall_csv(
    "data/real/imd_mumbai_monthly_rainfall_1901_2021.csv",
    dataset_record=OPENCITY_IMD_MONTHLY,
    station_id="IMD_GRID_19.00N_72.75E",
    provenance="REAL:IMD_GRID",
)
print(ds.summary())
# [REAL:IMD_GRID] Mumbai Monthly Rainfall 1901–2021 | ...
# records=121  years=1901–2021  skipped_rows=0  missing_monthly_values=0
```

### Derive an illustrative intensity from real data

```python
import warnings
from floodcut.rainfall_loader import make_scenario_from_record
from floodcut.models import RainfallScenario

rec = ds.record_for_year(2005)
with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always")
    intensity, label = make_scenario_from_record(rec, "Jul", "july_2005")
    # WARNING: This is a TIME-AVERAGED mean, NOT a design storm.

sc = RainfallScenario(
    id="july_2005_derived",
    name="July 2005 mean intensity (ILLUSTRATIVE ONLY)",
    intensity_mm_h=intensity,
    description=label,  # carries full provenance chain
)
print(f"Derived intensity: {intensity:.4f} mm/h (DERIVED:TEMPORAL_DISAGGREGATE)")
```

### Run the model with a derived scenario

```python
from floodcut.io import load_network
from floodcut.graph import build_flow_network
from floodcut.maxflow import analyze

net = load_network("data/simulated/mumbai_sim_network.json")
# Net is SYNTHETIC; scenario is DERIVED. Results are model outputs, not flood measurements.

fn = build_flow_network(net, sc)
result = analyze(net, fn)
print(f"Max flow (SYNTHETIC network, DERIVED intensity): {result.max_flow_m3s:.3f} m³/s")
print("NOTE: This is a prototype model output, NOT a real-world flood prediction.")
```

### Load geospatial layers

```python
from floodcut.geo_loader import load_imd_grid_points, load_waterlogging_hotspots

grid = load_imd_grid_points()
print(grid.points[0].provenance)  # REAL:IMD_GRID

hotspots = load_waterlogging_hotspots()
print(hotspots.points[0].provenance)  # DERIVED:GEOCODED
```

### Run tests

```bash
pytest tests/test_phase5_data_integration.py -v
pytest  # full suite (163+ tests expected to pass)
```

---

## What happens when a dataset is unavailable

- If `data/real/imd_mumbai_monthly_rainfall_1901_2021.csv` is deleted, the real-data tests are skipped (not failed) via `pytest.skip()`.
- If internet access is unavailable, use `load_synthetic_fixture()` which returns a bundled synthetic dataset requiring no network access.
- If a required column is missing, `RainfallDataError` is raised immediately with a clear message — the loader never silently substitutes synthetic data.
- If individual month values are missing, they are stored as `None` and reported via `record.missing_months()`.

---

## Known limitations

> Read before trusting any output.

1. **Prototype simulation, not a validated flood predictor.** The max-flow model uses a synthetic drainage network with invented capacities, the steady-state rational method, and best-case flow routing. None of these match real hydraulic conditions.

2. **No real drain capacity data.** No BMC/MCGM survey of actual drain capacities or conditions was available. All drain capacities are `SYNTHETIC`.

3. **Monthly totals ≠ design storms.** Converting monthly rainfall to an hourly intensity (DERIVED:TEMPORAL_DISAGGREGATE) produces a time-averaged mean that is physically different from a design storm. Do not use for engineering decisions.

4. **Single grid point.** The IMD dataset represents one 0.25° grid cell, not individual raingauge stations or sub-ward rainfall variability.

5. **Waterlogging hotspot coordinates are approximate.** Named junctions geocoded to ± ~500 m are not precise flood-extent polygons. No flood depth or return period data is encoded.

6. **Overflow ≠ measured flood.** Model-computed overflow values (m³/s) represent the amount of water that cannot be routed through the drainage network under the max-flow assumption. They are NOT measured real-world flood depths, volumes, or inundation extents.

7. **Steady-state model.** No storage, ponding, tide-surge interaction, or time variation is modelled.

See `docs/ASSUMPTIONS.md` for the full list of modelling assumptions (A1–A10).
