# Real data — Phase 5

This directory contains **real** datasets obtained from verifiable external sources.

> **All files here are clearly distinguished from the synthetic development data in `../simulated/`.**

## Files

### `imd_mumbai_monthly_rainfall_1901_2021.csv`

| Field | Value |
|---|---|
| **Provenance** | `REAL:IMD_GRID` |
| **Dataset name** | Mumbai Monthly Rainfall 1901–2021 (IMD Grid Point) |
| **Source organisation** | India Meteorological Department (IMD), published via OpenCity India |
| **Dataset page** | https://data.opencity.in/dataset/mumbai-rainfall-data |
| **Direct CSV URL** | https://data.opencity.in/dataset/c0e2f1ba-848a-4716-982b-d42636c44bf5/resource/3b1fcae7-b24f-431f-a259-64b56077acd5/download/6b7b0aed-e31d-433b-ba6d-b17b08a000df.csv |
| **Retrieval date** | 2026-10-23 |
| **Geographic coverage** | Single IMD 0.25° grid point centred at approximately 19.00° N, 72.75° E (covers Greater Mumbai and vicinity) |
| **Observation period** | 1901–2021 (121 years, 1 row per year) |
| **Spatial resolution** | 0.25° × 0.25° grid cell |
| **Units** | Monthly accumulated rainfall in millimetres (mm) |
| **CRS** | WGS84 (EPSG:4326) — grid-cell centre coordinates |
| **License** | Published on OpenCity India (data.opencity.in). IMD data is generally available for non-commercial research use; formal licensing terms are not stated on the dataset page. Verify current terms at the dataset URL before redistribution. |

#### Column schema

| Column | Type | Description |
|---|---|---|
| `Year` | int | Calendar year |
| `Jan` … `Dec` | float | Monthly accumulated rainfall in mm |
| `Total` | float | Annual total (sum of monthly values; may differ slightly due to rounding) |

#### Important limitations

1. **Grid point, not a raingauge station.** This is an IMD gridded dataset (derived from interpolated or reanalysis data), not a direct AWS/raingauge observation. Values represent the areal mean over the 0.25° × 0.25° grid cell.
2. **Monthly totals ≠ hourly intensities.** The rational-method engine requires rainfall intensity in mm/h. Monthly totals must be temporally disaggregated before use (see `rainfall_loader.monthly_mm_to_intensity_mm_h`). The disaggregated mean intensity is far lower than a design-storm intensity.
3. **`REAL` ≠ validated flood prediction.** Using real rainfall observations as input to the model does not validate the model's overflow predictions against real-world flood observations. The simulation uses a synthetic drainage network (SYNTHETIC provenance) and steady-state max-flow assumptions.

---

## What is NOT included here

| Data type | Reason not included | Provenance if obtained |
|---|---|---|
| IMD station observations (Colaba / Santacruz daily) | Require formal data request via IMD Data Service Portal (dsp.imdpune.gov.in); not publicly downloadable | Would be `REAL:IMD_STATION` |
| BMC/MCGM drain survey or GIS layer | Not published in machine-readable open format | Would be `REAL:BMC_OFFICIAL` |
| Ward boundary polygons (GeoJSON) | Polygon geometry requires geopandas (not a current dependency); license unclear | Would be `REAL:COMMUNITY_GIS` or `REAL:BMC_OFFICIAL` |
| Real-time rainfall API (IMD / IITM Mesonet) | Requires institutional login or API key; not scripted in Phase 5 | Would be `REAL:IMD_REALTIME` |

---

## Provenance labels used in Phase 5

| Label | Meaning |
|---|---|
| `REAL:IMD_GRID` | Obtained from IMD gridded dataset via verifiable public source |
| `REAL:IMD_STATION` | Direct raingauge or AWS station observation (not yet available) |
| `DERIVED:TEMPORAL_DISAGGREGATE` | Monthly total converted to hourly intensity via documented assumption |
| `DERIVED:GEOCODED` | Named location converted to coordinates via manual geocoding |
| `ASSUMED` | Value decided by modeller without empirical source |
| `SYNTHETIC` | Artificial data for testing or algorithm demonstration |
