"""Phase 5: Geospatial data loading and validation.

Loads and validates geographic datasets relevant to Mumbai drainage modelling.
Currently supported layers:

1. **Rainfall station / grid-point coordinates** — CSV with lat/lon.
2. **Waterlogging hotspot locations** — CSV with lat/lon and optional
   attribute columns (ward, description).

All real geospatial layers are labelled with provenance using ``provenance.py``.
Clearly documented limitations are applied where data is unavailable.

Coordinate Reference System (CRS)
----------------------------------
All loaded coordinates are assumed to be in **WGS84 (EPSG:4326)** unless
stated otherwise. Degrees are decimal (not DMS). Latitude positive = North,
Longitude positive = East.

Mumbai bounding box used for validation (from ``validation.py``):
    Latitude:  18.85° N to 19.30° N
    Longitude: 72.75° E to 73.05° E

Known limitations
-----------------
* No real drain-network GIS layer is available. Roads or waterways from OSM
  are NOT used as drain proxy — doing so would misrepresent the data.
* No BMC/MCGM published drain survey in machine-readable GIS format was
  accessible at the time of implementation (October 2026). The synthetic
  drain network from Phases 1–4 is retained with ``SYNTHETIC`` label.
* Waterlogging hotspots from civic data are available in table form but without
  precise surveyed geometries; any coordinates derived from named locations are
  labelled ``DERIVED:GEOCODED`` with the geocoding method documented.
"""

from __future__ import annotations

import csv
import io
import math
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from .provenance import DatasetRecord, assert_valid, REAL, SYNTHETIC, DERIVED

# ---------------------------------------------------------------------------
# Mumbai geographic bounds (mirrors validation.py for consistency)
# ---------------------------------------------------------------------------

MUMBAI_LAT_RANGE = (18.85, 19.30)
MUMBAI_LON_RANGE = (72.75, 73.05)


# ---------------------------------------------------------------------------
# Errors and warnings
# ---------------------------------------------------------------------------

class GeospatialDataError(ValueError):
    """Raised when a geospatial data file is malformed or missing."""


class GeospatialDataWarning(UserWarning):
    """Issued for non-fatal geospatial issues (out-of-bounds, missing coords)."""


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

@dataclass
class GeoPoint:
    """A single georeferenced point with provenance.

    Attributes
    ----------
    id : str
        Unique identifier for this point.
    name : str
        Human-readable label.
    lat : float
        Latitude in decimal degrees (WGS84).
    lon : float
        Longitude in decimal degrees (WGS84).
    provenance : str
        Provenance label (see ``provenance.py``).
    attributes : dict
        Additional key-value attributes (ward, description, etc.).
    """
    id: str
    name: str
    lat: float
    lon: float
    provenance: str
    attributes: dict = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        assert_valid(self.provenance)
        if self.attributes is None:
            self.attributes = {}
        if not math.isfinite(self.lat) or not (-90 <= self.lat <= 90):
            raise GeospatialDataError(
                f"Point {self.id!r}: invalid latitude {self.lat!r}"
            )
        if not math.isfinite(self.lon) or not (-180 <= self.lon <= 180):
            raise GeospatialDataError(
                f"Point {self.id!r}: invalid longitude {self.lon!r}"
            )

    def in_mumbai_bbox(self) -> bool:
        """True if the point lies within the Greater Mumbai bounding box."""
        lat_ok = MUMBAI_LAT_RANGE[0] <= self.lat <= MUMBAI_LAT_RANGE[1]
        lon_ok = MUMBAI_LON_RANGE[0] <= self.lon <= MUMBAI_LON_RANGE[1]
        return lat_ok and lon_ok


@dataclass
class GeoLayer:
    """A collection of georeferenced points with shared provenance metadata.

    Attributes
    ----------
    points : list[GeoPoint]
        All loaded points.
    dataset_record : DatasetRecord
        Provenance metadata for this layer.
    skipped_rows : list[str]
        Descriptions of rows that were skipped during loading.
    crs : str
        Coordinate Reference System (default WGS84/EPSG:4326).
    """
    points: list[GeoPoint]
    dataset_record: DatasetRecord
    skipped_rows: list[str] = None  # type: ignore[assignment]
    crs: str = "WGS84 (EPSG:4326)"

    def __post_init__(self) -> None:
        if self.skipped_rows is None:
            self.skipped_rows = []

    def outside_bbox(self) -> list[GeoPoint]:
        """Points that fall outside the Greater Mumbai bounding box."""
        return [p for p in self.points if not p.in_mumbai_bbox()]

    def summary(self) -> str:
        outside = len(self.outside_bbox())
        return (
            f"{self.dataset_record.summary()}\n"
            f"  points={len(self.points)}  "
            f"outside_mumbai_bbox={outside}  "
            f"skipped_rows={len(self.skipped_rows)}"
        )


# ---------------------------------------------------------------------------
# Generic CSV point loader
# ---------------------------------------------------------------------------

def _parse_coord(value: str, col: str, row_idx: int) -> Optional[float]:
    """Parse a coordinate cell; warn and return None on failure."""
    stripped = value.strip()
    if stripped in ("", "NA", "N/A", "nan", "null", "-"):
        warnings.warn(
            f"Row {row_idx}, column {col!r}: missing coordinate — row skipped.",
            GeospatialDataWarning,
            stacklevel=4,
        )
        return None
    try:
        return float(stripped)
    except ValueError:
        warnings.warn(
            f"Row {row_idx}, column {col!r}: cannot parse {stripped!r} as float "
            f"— row skipped.",
            GeospatialDataWarning,
            stacklevel=4,
        )
        return None


def load_geo_points_csv(
    source: str | Path,
    *,
    dataset_record: DatasetRecord,
    id_col: str = "id",
    name_col: str = "name",
    lat_col: str = "lat",
    lon_col: str = "lon",
    provenance: str,
    warn_outside_bbox: bool = True,
) -> GeoLayer:
    """Load a CSV of georeferenced points.

    The CSV must have columns for id, name, latitude, and longitude (column
    names are configurable). All other columns become ``attributes``.

    Parameters
    ----------
    source:
        Path to the CSV file OR a raw CSV string (detected automatically).
    dataset_record:
        Provenance metadata for the layer.
    id_col, name_col, lat_col, lon_col:
        Column names for id, name, latitude, longitude.
    provenance:
        Provenance label for individual points.
    warn_outside_bbox:
        If True, emit a ``GeospatialDataWarning`` for points outside the
        Greater Mumbai bounding box.

    Returns
    -------
    GeoLayer
    """
    assert_valid(provenance)

    if isinstance(source, str) and "\n" in source:
        text = source
    else:
        p = Path(source)
        if not p.exists():
            raise GeospatialDataError(f"GeoCSV not found: {p}")
        try:
            text = p.read_text(encoding="utf-8")
        except OSError as exc:
            raise GeospatialDataError(f"Cannot read {p}: {exc}") from exc

    try:
        reader = csv.DictReader(io.StringIO(text))
        fieldnames = reader.fieldnames or []
    except csv.Error as exc:
        raise GeospatialDataError(f"CSV parse error: {exc}") from exc

    # Check required columns exist.
    required = {id_col, lat_col, lon_col}
    missing = required - set(f.strip() for f in fieldnames)
    if missing:
        raise GeospatialDataError(
            f"CSV is missing required columns: {sorted(missing)}. "
            f"Found: {list(fieldnames)}"
        )

    points: list[GeoPoint] = []
    skipped: list[str] = []
    seen_ids: set[str] = set()

    for row_idx, row in enumerate(reader, start=2):
        pid = row.get(id_col, "").strip()
        if not pid:
            skipped.append(f"Row {row_idx}: empty id — skipped.")
            continue
        if pid in seen_ids:
            warnings.warn(
                f"Row {row_idx}: duplicate id {pid!r} — later row dropped.",
                GeospatialDataWarning,
                stacklevel=2,
            )
            skipped.append(f"Row {row_idx}: duplicate id {pid!r} — dropped.")
            continue
        seen_ids.add(pid)

        pname = row.get(name_col, pid).strip() if name_col in row else pid

        lat_raw = row.get(lat_col, "")
        lon_raw = row.get(lon_col, "")
        lat = _parse_coord(lat_raw, lat_col, row_idx)
        lon = _parse_coord(lon_raw, lon_col, row_idx)
        if lat is None or lon is None:
            skipped.append(
                f"Row {row_idx} (id={pid!r}): missing or invalid coordinate — skipped."
            )
            continue

        attrs = {
            k.strip(): v.strip()
            for k, v in row.items()
            if k.strip() not in {id_col, name_col, lat_col, lon_col}
        }

        try:
            pt = GeoPoint(
                id=pid, name=pname, lat=lat, lon=lon,
                provenance=provenance, attributes=attrs,
            )
        except GeospatialDataError as exc:
            skipped.append(f"Row {row_idx} (id={pid!r}): {exc} — skipped.")
            continue

        if warn_outside_bbox and not pt.in_mumbai_bbox():
            warnings.warn(
                f"Point {pid!r} at ({lat}, {lon}) is outside the Greater Mumbai "
                f"bounding box ({MUMBAI_LAT_RANGE}, {MUMBAI_LON_RANGE}).",
                GeospatialDataWarning,
                stacklevel=2,
            )
        points.append(pt)

    return GeoLayer(
        points=points,
        dataset_record=dataset_record,
        skipped_rows=skipped,
    )


# ---------------------------------------------------------------------------
# Known IMD grid point(s) for Mumbai
# ---------------------------------------------------------------------------

#: The single IMD 0.25° grid point that covers the Mumbai region.
#: Source: OpenCity dataset metadata and IMD documentation.
#: Provenance: REAL:IMD_GRID (location published by IMD / OpenCity).
#: CRS: WGS84 (EPSG:4326), grid-cell centre.
IMD_GRID_POINTS_CSV = """\
id,name,lat,lon,grid_resolution_deg,data_source
IMD_GRID_19.00N_72.75E,IMD Mumbai grid point,19.00,72.75,0.25,IMD via OpenCity (data.opencity.in)
"""

from .provenance import DatasetRecord, OPENCITY_IMD_MONTHLY  # noqa: E402 (re-import for default)

_IMD_GRID_LAYER_RECORD = DatasetRecord(
    provenance="REAL:IMD_GRID",
    name="IMD Rainfall Grid Points — Mumbai region",
    source_organisation="India Meteorological Department (IMD), via OpenCity India",
    source_url="https://data.opencity.in/dataset/mumbai-rainfall-data",
    retrieval_date=OPENCITY_IMD_MONTHLY.retrieval_date,
    license_notes=OPENCITY_IMD_MONTHLY.license_notes,
    geographic_coverage="Single 0.25° grid cell centred at 19.00° N, 72.75° E",
    observation_period="1901–2021",
    crs="WGS84 (EPSG:4326) — grid-cell centre",
    notes=(
        "Grid-point location derived from the dataset metadata published by IMD "
        "and documented on the OpenCity dataset landing page. This is a single "
        "point representing the grid-cell centre; it is NOT an AWS or raingauge "
        "station location."
    ),
)


def load_imd_grid_points() -> GeoLayer:
    """Return the known IMD grid point(s) for the Mumbai region.

    This uses a hardcoded small CSV derived from the OpenCity dataset
    metadata. No internet connection is required.

    Returns
    -------
    GeoLayer with provenance ``REAL:IMD_GRID``.
    """
    return load_geo_points_csv(
        IMD_GRID_POINTS_CSV,
        dataset_record=_IMD_GRID_LAYER_RECORD,
        provenance="REAL:IMD_GRID",
    )


# ---------------------------------------------------------------------------
# Mumbai ward boundaries — availability note
# ---------------------------------------------------------------------------

WARD_BOUNDARY_UNAVAILABILITY_NOTE = """
Mumbai ward boundary data (BMC/MCGM administrative ward polygons) is
referenced by multiple open-data sources (BharatAtlas, GitHub community
repos, DataMeet) but was not downloaded and bundled in this phase because:

1. The data is polygon geometry (GeoJSON/Shapefile) — FloodCut's current
   models use only point locations. Polygon processing requires geopandas,
   which is not a current dependency.
2. No single authoritative machine-readable source with a clearly stated
   open license was confirmed at the time of implementation (October 2026).

IMPORT MECHANISM FOR USER-SUPPLIED DATA:
  If you have a ward boundary GeoJSON file, you can load it with:
    import json; data = json.load(open("mcgm_wards.geojson"))
  The provenance label for data from the BharatAtlas community source
  would be: REAL:COMMUNITY_GIS
  For officially published BMC data: REAL:BMC_OFFICIAL

  Verify the license before use. Candidate public sources:
    - https://bharatlas.com/view/wards_mumbai  (BharatAtlas — community)
    - https://github.com/datameet/Municipal_Spatial_Data  (DataMeet)
"""


# ---------------------------------------------------------------------------
# Waterlogging hotspot layer — derived from civic reports
# ---------------------------------------------------------------------------

#: A small curated list of well-known Mumbai waterlogging hotspots.
#: PROVENANCE: DERIVED:GEOCODED
#: These locations are compiled from publicly reported civic information
#: (MCGM press releases, media reports) and geocoded to approximate ward
#: centroids. They are NOT precise surveyed flood depths or flood extent
#: polygons. Coordinates are approximate (± ~500 m).
#:
#: Source references (not machine-readable APIs — manually compiled):
#:   MCGM Disaster Management Cell annual waterlogging reports (various years)
#:   Mumbai Mirror / Times of India civic reporting
#:   https://portal.mcgm.gov.in/ (publicly accessible civic portal)
WATERLOGGING_HOTSPOTS_CSV = """\
id,name,lat,lon,ward,notes
WL_HINDMATA,Hindmata Junction,19.0182,72.8310,G-South,Chronic low-lying underpass; frequently reported flooded
WL_MILAN_SUBWAY,Milan Subway,19.0555,72.8382,H-West,Below-grade subway; floods in monsoon
WL_KING_CIRCLE,King Circle Junction,19.0218,72.8467,F-South,Known waterlogging point reported multiple years
WL_ANDHERI_SUBWAY,Andheri Subway,19.1197,72.8426,K-West,Underpass repeatedly closed during monsoon
WL_SION_CIRCLE,Sion Circle,19.0434,72.8597,F-North,At-grade junction; medium flooding severity
WL_KURLA_STATION,Kurla Station area,19.0728,72.8784,L,Near railway; reported flooding
WL_MATUNGA,Matunga Road Station,19.0283,72.8479,F-South,Below-grade road; floods during heavy rain
"""

_HOTSPOT_LAYER_RECORD = DatasetRecord(
    provenance="DERIVED:GEOCODED",
    name="Mumbai Waterlogging Hotspots (approximate geocoded locations)",
    source_organisation=(
        "Compiled from MCGM Disaster Management Cell reports and media "
        "sources; geocoded by FloodCut team"
    ),
    source_url="https://portal.mcgm.gov.in/",
    retrieval_date=None,  # compilation date varies
    license_notes=(
        "Derived from publicly available civic data. No specific license stated. "
        "Coordinates are approximate (± ~500 m) and represent named junctions "
        "or localities, NOT precise flood-extent polygons."
    ),
    geographic_coverage="Greater Mumbai — selected frequently reported locations",
    observation_period="Compiled from reports across multiple years (≈2015–2025)",
    crs="WGS84 (EPSG:4326) — approximate geocoded coordinates",
    notes=(
        "IMPORTANT LIMITATIONS:\n"
        "1. These are NAMED LOCATIONS reported in civic sources, not precise "
        "surveyed flood boundaries. Coordinates are geocoded to ward centroids "
        "or known junction positions (accuracy ± ~500 m).\n"
        "2. Flood depth, duration, and frequency are NOT encoded — only location.\n"
        "3. This list is incomplete; many waterlogging events are not documented "
        "in accessible public sources.\n"
        "4. Do NOT present model overflow values at these locations as measured "
        "real-world flood depths.\n"
        "5. Provenance label: DERIVED:GEOCODED (not REAL:OBSERVED)."
    ),
)


def load_waterlogging_hotspots() -> GeoLayer:
    """Return the bundled approximate waterlogging hotspot layer.

    No internet connection required. Uses the bundled CSV above.

    Returns
    -------
    GeoLayer with provenance ``DERIVED:GEOCODED``.
    """
    return load_geo_points_csv(
        WATERLOGGING_HOTSPOTS_CSV,
        dataset_record=_HOTSPOT_LAYER_RECORD,
        provenance="DERIVED:GEOCODED",
    )
