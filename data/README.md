# Data provenance

`simulated/` holds **synthetic** data created for development. Nothing here comes from BMC/MCGM, IMD or any
survey. Every file declares `"is_simulated": true`, and the CLI prints a banner whenever such data is loaded.

`real/` holds **real** data obtained from verifiable external sources. Each file and its provenance are
documented in `real/README.md`. Provenance labels distinguish these clearly from the synthetic data.

| File | Provenance | Contents |
|---|---|---|
| `simulated/mumbai_sim_network.json` | `SYNTHETIC` | 14 junctions, 14 drains, 6 catchments, 2 outfalls, 4 critical assets |
| `simulated/rainfall_scenarios.json` | `SYNTHETIC` | 4 rainfall + tide scenarios (assumptions, not forecasts) |
| `real/imd_mumbai_monthly_rainfall_1901_2021.csv` | `REAL:IMD_GRID` | IMD monthly rainfall for Mumbai grid point 1901–2021 (OpenCity India) |

## Provenance labels (Phase 5)

| Label | Meaning |
|---|---|
| `SYNTHETIC` | Artificial data for algorithm development; no real source |
| `REAL:IMD_GRID` | Obtained from IMD gridded dataset, published via OpenCity India |
| `DERIVED:TEMPORAL_DISAGGREGATE` | Monthly total converted to hourly intensity via documented assumption |
| `DERIVED:GEOCODED` | Named location geocoded to approximate coordinates |
| `ASSUMED` | Modeller decision; no empirical source |

## Rules

- Real and synthetic data are kept in **separate directories** and carry **distinct provenance labels**.
- Coordinates in `simulated/` are illustrative positions inside Greater Mumbai, not surveyed locations.
- Do not mix real and synthetic records in a single file without explicit provenance labelling.
- See `docs/PHASE5_DATA_INTEGRATION.md` for full data source documentation.
