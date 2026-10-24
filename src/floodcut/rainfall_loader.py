"""Phase 5: Rainfall data loading and preprocessing.

Loads, validates, and preprocesses real (or synthetic test) rainfall
observations. The primary supported format is a **CSV with one row per year
and one column per calendar month**, which matches the OpenCity / IMD monthly
grid-point dataset.

Provenance contract
-------------------
* Every ``RainfallRecord`` carries a ``provenance`` label (see
  ``provenance.py``).
* Records loaded from a verified external file are labelled ``REAL:IMD_GRID``.
* Records in the bundled test fixture are labelled ``SYNTHETIC``.
* Derived intensity values produced from monthly totals are labelled
  ``DERIVED:TEMPORAL_DISAGGREGATE``.

Unit convention
---------------
The raw OpenCity CSV reports monthly accumulated rainfall in **millimetres
(mm)**. The rational-method engine expects rainfall *intensity* in **mm/h**.
Converting monthly totals to an instantaneous intensity requires a documented
temporal disaggregation assumption (see ``monthly_mm_to_intensity_mm_h``).
Never pass raw monthly mm totals directly as mm/h — that would be
dimensionally incorrect by several orders of magnitude.

Column contract for the IMD monthly CSV
----------------------------------------
Required columns (case-insensitive, leading/trailing spaces stripped):
    ``Year``, ``Jan``, ``Feb``, ``Mar``, ``April`` (or ``Apr``),
    ``May``, ``June`` (or ``Jun``), ``July`` (or ``Jul``), ``Aug``,
    ``Sept`` (or ``Sep``), ``Oct``, ``Nov``, ``Dec``.

Optional columns silently retained but not required:
    ``Total``.

Any column not in this list is silently dropped.
"""

from __future__ import annotations

import csv
import io
import warnings
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Iterable, Iterator, Optional

from .provenance import (
    DERIVED,
    REAL,
    SYNTHETIC,
    DatasetRecord,
    OPENCITY_IMD_MONTHLY,
    assert_valid,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Canonical month column names in order Jan–Dec.
MONTH_NAMES = (
    "Jan", "Feb", "Mar", "Apr", "May", "Jun",
    "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
)

# Accepted alternative spellings from the OpenCity CSV header.
_MONTH_ALIASES: dict[str, str] = {
    "january": "Jan", "jan": "Jan",
    "february": "Feb", "feb": "Feb",
    "march": "Mar", "mar": "Mar",
    "april": "Apr", "apr": "Apr",
    "may": "May",
    "june": "Jun", "jun": "Jun",
    "july": "Jul", "jul": "Jul",
    "august": "Aug", "aug": "Aug",
    "september": "Sep", "sept": "Sep", "sep": "Sep",
    "october": "Oct", "oct": "Oct",
    "november": "Nov", "nov": "Nov",
    "december": "Dec", "dec": "Dec",
}

#: Mumbai monsoon months (June–September). These carry the bulk of annual
#: rainfall and are the focus of flood-risk analysis.
MONSOON_MONTHS = ("Jun", "Jul", "Aug", "Sep")

#: IMD grid point for the dataset (latitude, longitude).
IMD_GRID_LAT = 19.00
IMD_GRID_LON = 72.75

#: WGS84 bounding box for Greater Mumbai.
MUMBAI_LAT_RANGE = (18.85, 19.30)
MUMBAI_LON_RANGE = (72.75, 73.05)


# ---------------------------------------------------------------------------
# Rainfall data errors
# ---------------------------------------------------------------------------

class RainfallDataError(ValueError):
    """Raised when a rainfall data file or record is malformed."""


class RainfallDataWarning(UserWarning):
    """Issued for non-fatal problems (missing values, duplicates, etc.)."""


# ---------------------------------------------------------------------------
# Record types
# ---------------------------------------------------------------------------

@dataclass
class MonthlyRainfallRecord:
    """Monthly rainfall totals for one calendar year.

    Attributes
    ----------
    year : int
        Calendar year (e.g. 2005).
    monthly_mm : dict[str, float | None]
        Mapping from three-letter month abbreviation to accumulated rainfall
        in millimetres, or ``None`` if the value was missing or invalid.
    station_id : str
        Station or grid-point identifier (e.g. ``"IMD_GRID_19.00N_72.75E"``).
    provenance : str
        Provenance label (see ``provenance.py``).
    source_row : int | None
        1-based row index in the source file (useful for error messages).
    """
    year: int
    monthly_mm: dict[str, Optional[float]]  # key = 3-letter month abbrev
    station_id: str
    provenance: str
    source_row: Optional[int] = None

    def __post_init__(self) -> None:
        assert_valid(self.provenance)
        if not isinstance(self.year, int) or self.year < 1800 or self.year > 2100:
            raise RainfallDataError(
                f"year must be an integer in [1800, 2100], got {self.year!r}"
            )

    def annual_total_mm(self) -> Optional[float]:
        """Sum of all non-missing monthly values; None if ALL are missing."""
        vals = [v for v in self.monthly_mm.values() if v is not None]
        return sum(vals) if vals else None

    def monsoon_total_mm(self) -> Optional[float]:
        """Sum of Jun+Jul+Aug+Sep; None if ALL monsoon values are missing."""
        vals = [
            self.monthly_mm.get(m)
            for m in MONSOON_MONTHS
            if self.monthly_mm.get(m) is not None
        ]
        return sum(vals) if vals else None  # type: ignore[arg-type]

    def missing_months(self) -> list[str]:
        """Month abbreviations for which the value is None."""
        return [m for m in MONTH_NAMES if self.monthly_mm.get(m) is None]


@dataclass
class RainfallDataset:
    """Container for a loaded set of monthly rainfall records.

    Attributes
    ----------
    records : list[MonthlyRainfallRecord]
        One record per year, sorted by year ascending.
    dataset_record : DatasetRecord
        Provenance metadata for this dataset.
    skipped_rows : list[str]
        Human-readable descriptions of rows that were skipped (malformed,
        duplicate year, etc.).
    """
    records: list[MonthlyRainfallRecord]
    dataset_record: DatasetRecord
    skipped_rows: list[str] = field(default_factory=list)

    def years(self) -> list[int]:
        return [r.year for r in self.records]

    def year_range(self) -> Optional[tuple[int, int]]:
        if not self.records:
            return None
        return (self.records[0].year, self.records[-1].year)

    def record_for_year(self, year: int) -> Optional[MonthlyRainfallRecord]:
        for r in self.records:
            if r.year == year:
                return r
        return None

    def monsoon_totals(self) -> dict[int, Optional[float]]:
        """Map year → monsoon total mm for all records."""
        return {r.year: r.monsoon_total_mm() for r in self.records}

    def summary(self) -> str:
        yr = self.year_range()
        yr_str = f"{yr[0]}–{yr[1]}" if yr else "none"
        missing_ct = sum(
            len(r.missing_months()) for r in self.records
        )
        return (
            f"{self.dataset_record.summary()}\n"
            f"  records={len(self.records)}  years={yr_str}  "
            f"skipped_rows={len(self.skipped_rows)}  "
            f"missing_monthly_values={missing_ct}"
        )


# ---------------------------------------------------------------------------
# Column normalisation
# ---------------------------------------------------------------------------

def _normalise_header(raw: list[str]) -> dict[str, str]:
    """Return {raw_column_name: canonical_name} for recognised columns.

    Unrecognised columns are omitted from the result.
    """
    mapping: dict[str, str] = {}
    for col in raw:
        stripped = col.strip()
        lower = stripped.lower()
        if lower == "year":
            mapping[col] = "Year"
        elif lower in _MONTH_ALIASES:
            mapping[col] = _MONTH_ALIASES[lower]
        # Silently skip unrecognised columns (e.g. "Total").
    return mapping


# ---------------------------------------------------------------------------
# Numeric parsing
# ---------------------------------------------------------------------------

def _parse_float(value: str, col: str, row_idx: int) -> Optional[float]:
    """Parse a cell value to float, returning None for missing/NaN values.

    Logs a warning for values that are non-empty but unparseable.
    """
    stripped = value.strip()
    if stripped in ("", "NA", "N/A", "nan", "NaN", "null", "NULL", "-"):
        return None
    try:
        f = float(stripped)
    except ValueError:
        warnings.warn(
            f"Row {row_idx}, column {col!r}: cannot parse {stripped!r} as float "
            f"— treating as missing.",
            RainfallDataWarning,
            stacklevel=4,
        )
        return None
    if f < 0:
        warnings.warn(
            f"Row {row_idx}, column {col!r}: negative rainfall {f} mm "
            f"— treating as missing.",
            RainfallDataWarning,
            stacklevel=4,
        )
        return None
    return f


# ---------------------------------------------------------------------------
# Core loader
# ---------------------------------------------------------------------------

def _load_from_reader(
    reader: Iterable[dict[str, str]],
    header: list[str],
    *,
    dataset_record: DatasetRecord,
    station_id: str,
    provenance: str,
) -> RainfallDataset:
    """Internal: load records from a csv.DictReader-compatible iterable."""
    col_map = _normalise_header(header)

    missing_months = [m for m in MONTH_NAMES if m not in col_map.values()]
    if missing_months:
        raise RainfallDataError(
            f"CSV is missing required month columns: {missing_months}. "
            f"Recognised columns: {list(col_map.values())}"
        )
    if "Year" not in col_map.values():
        raise RainfallDataError("CSV is missing required 'Year' column.")

    # Reverse map: raw column name → canonical name
    # We need: canonical name → raw column name for DictReader lookups
    canonical_to_raw: dict[str, str] = {v: k for k, v in col_map.items()}

    records: list[MonthlyRainfallRecord] = []
    skipped: list[str] = []
    seen_years: set[int] = set()

    for row_idx, row in enumerate(reader, start=2):  # row 1 = header
        raw_year = row.get(canonical_to_raw["Year"], "").strip()
        if not raw_year:
            skipped.append(f"Row {row_idx}: empty year — skipped.")
            continue
        try:
            year = int(float(raw_year))
        except ValueError:
            skipped.append(
                f"Row {row_idx}: cannot parse year {raw_year!r} — skipped."
            )
            continue

        if year in seen_years:
            warnings.warn(
                f"Row {row_idx}: duplicate year {year} — later row dropped.",
                RainfallDataWarning,
                stacklevel=3,
            )
            skipped.append(f"Row {row_idx}: duplicate year {year} — dropped.")
            continue
        seen_years.add(year)

        monthly: dict[str, Optional[float]] = {}
        for month in MONTH_NAMES:
            raw_col = canonical_to_raw.get(month)
            if raw_col is None:
                monthly[month] = None
            else:
                cell = row.get(raw_col, "")
                monthly[month] = _parse_float(cell, month, row_idx)

        try:
            rec = MonthlyRainfallRecord(
                year=year,
                monthly_mm=monthly,
                station_id=station_id,
                provenance=provenance,
                source_row=row_idx,
            )
        except RainfallDataError as exc:
            skipped.append(f"Row {row_idx}: {exc} — skipped.")
            continue
        records.append(rec)

    records.sort(key=lambda r: r.year)
    return RainfallDataset(
        records=records,
        dataset_record=dataset_record,
        skipped_rows=skipped,
    )


def load_rainfall_csv(
    path: str | Path,
    *,
    dataset_record: Optional[DatasetRecord] = None,
    station_id: str = "UNKNOWN",
    provenance: str = "REAL:IMD_GRID",
) -> RainfallDataset:
    """Load a monthly rainfall CSV (one row per year, one column per month).

    Parameters
    ----------
    path:
        Path to the CSV file. Must be UTF-8 encoded.
    dataset_record:
        Provenance metadata. Defaults to ``OPENCITY_IMD_MONTHLY`` if
        ``provenance`` starts with ``"REAL"``, else a generic record.
    station_id:
        Identifier for the station or grid point.
    provenance:
        Provenance label for individual records.

    Returns
    -------
    RainfallDataset

    Raises
    ------
    RainfallDataError
        If the file does not exist, is not valid CSV, or is missing required
        columns.
    """
    assert_valid(provenance)
    p = Path(path)
    if not p.exists():
        raise RainfallDataError(f"Rainfall CSV not found: {p}")

    try:
        text = p.read_text(encoding="utf-8")
    except OSError as exc:
        raise RainfallDataError(f"Cannot read {p}: {exc}") from exc

    return load_rainfall_csv_string(
        text,
        dataset_record=dataset_record,
        station_id=station_id,
        provenance=provenance,
    )


def load_rainfall_csv_string(
    text: str,
    *,
    dataset_record: Optional[DatasetRecord] = None,
    station_id: str = "UNKNOWN",
    provenance: str = "REAL:IMD_GRID",
) -> RainfallDataset:
    """Load a monthly rainfall CSV from a string (useful for tests).

    Accepts the same format as ``load_rainfall_csv``.
    """
    assert_valid(provenance)
    if dataset_record is None:
        from .provenance import DatasetRecord  # avoid circular import at top
        dataset_record = OPENCITY_IMD_MONTHLY

    try:
        reader = csv.DictReader(io.StringIO(text))
        # Force reading the header now so we can catch errors immediately.
        header = reader.fieldnames
        if header is None:
            raise RainfallDataError("CSV appears to be empty.")
        header = list(header)
        return _load_from_reader(
            reader,
            header,
            dataset_record=dataset_record,
            station_id=station_id,
            provenance=provenance,
        )
    except csv.Error as exc:
        raise RainfallDataError(f"CSV parse error: {exc}") from exc


# ---------------------------------------------------------------------------
# Unit conversion / temporal disaggregation
# ---------------------------------------------------------------------------

#: Number of hours in each month (based on 365-day year average per month).
#: Used as the denominator when converting monthly mm totals to mean hourly
#: intensity. This is a deliberately conservative, documented approximation.
_HOURS_PER_MONTH = {
    "Jan": 744.0, "Feb": 672.0, "Mar": 744.0, "Apr": 720.0,
    "May": 744.0, "Jun": 720.0, "Jul": 744.0, "Aug": 744.0,
    "Sep": 720.0, "Oct": 744.0, "Nov": 720.0, "Dec": 744.0,
}


def monthly_mm_to_intensity_mm_h(
    total_mm: float,
    month: str,
) -> float:
    """Convert a monthly rainfall total (mm) to a mean hourly intensity (mm/h).

    This is a temporal disaggregation: it divides the monthly total by the
    number of hours in that calendar month to get a *time-averaged* intensity.

    IMPORTANT CAVEATS (documented as DERIVED:TEMPORAL_DISAGGREGATE):
    1. Real rainfall is intermittent. The mean hourly intensity is far lower
       than a design storm or peak intensity. It should NEVER be used directly
       as the rational-method design intensity without explicit acknowledgement.
    2. A 50 mm/h design storm for a 1-hour event is not the same as
       50 mm accumulated over a month.
    3. This function is provided for illustrative integration only. For flood
       risk screening, use scenario-based intensities (e.g. 50 mm/h design
       storm) informed by IDF curves, not monthly averages.

    Parameters
    ----------
    total_mm:
        Monthly accumulated rainfall in millimetres.
    month:
        Three-letter month abbreviation (e.g. ``"Jun"``).

    Returns
    -------
    float
        Mean hourly intensity in mm/h. Rounded to 6 significant figures.
    """
    hours = _HOURS_PER_MONTH.get(month)
    if hours is None:
        raise ValueError(
            f"Unknown month abbreviation {month!r}. "
            f"Use one of {list(_HOURS_PER_MONTH)}"
        )
    if total_mm < 0:
        raise ValueError(f"total_mm must be >= 0, got {total_mm!r}")
    return round(total_mm / hours, 6)


def monsoon_season_intensity_mm_h(
    record: MonthlyRainfallRecord,
    *,
    hours_per_event: float = 6.0,
) -> Optional[float]:
    """Estimate a design-storm-like intensity from a monsoon season total.

    This converts the June–September total rainfall into an approximate mean
    *event* intensity by dividing by an assumed ``hours_per_event`` duration.
    The result is labelled ``DERIVED:TEMPORAL_DISAGGREGATE`` in all outputs.

    This is a rough proxy, NOT an IDF-derived design intensity. It is provided
    only to allow illustrative integration with the max-flow engine. For any
    real design application, use proper IDF curves from IMD.

    Parameters
    ----------
    record:
        A ``MonthlyRainfallRecord``.
    hours_per_event:
        Assumed storm event duration in hours (default 6 h). This must be
        documented wherever the result is used.

    Returns
    -------
    Optional[float]
        Intensity in mm/h, or ``None`` if all monsoon months are missing.
    """
    total = record.monsoon_total_mm()
    if total is None:
        return None
    # Total monsoon rainfall / (number of monsoon hours × fraction assumed to
    # fall in peak events). We keep it simple: assume one canonical 6-hour
    # event per year contributes the "design" value.
    # NOTE: This is purely illustrative. The result is ~3-4 orders of magnitude
    # smaller than a real design storm.
    return round(total / hours_per_event, 4)


# ---------------------------------------------------------------------------
# Integration adapter
# ---------------------------------------------------------------------------

def make_scenario_from_record(
    record: MonthlyRainfallRecord,
    month: str,
    scenario_id: str,
    *,
    hours_per_event: float = 6.0,
    tide_factor: float = 1.0,
) -> tuple[float, str]:
    """Derive a (intensity_mm_h, provenance_label) pair from a monthly record.

    The returned intensity is labelled ``DERIVED:TEMPORAL_DISAGGREGATE`` and
    is suitable ONLY for illustrative model runs, not for design decisions.

    Parameters
    ----------
    record:
        Source ``MonthlyRainfallRecord``.
    month:
        Three-letter month abbreviation to extract (e.g. ``"Jul"``).
    scenario_id:
        Identifier for logging; not returned.
    hours_per_event:
        Assumed storm duration (default 6 h).
    tide_factor:
        Tide factor [0, 1] for outfall capacity scaling.

    Returns
    -------
    (intensity_mm_h, provenance_label)

    Raises
    ------
    RainfallDataError
        If the requested month is missing in the record.
    """
    mm = record.monthly_mm.get(month)
    if mm is None:
        raise RainfallDataError(
            f"Month {month!r} is missing in record for year {record.year}. "
            f"Cannot derive intensity."
        )
    intensity = monthly_mm_to_intensity_mm_h(mm, month)
    # Warn loudly that this is a mean intensity, not a design storm.
    warnings.warn(
        f"Intensity {intensity:.4f} mm/h derived from {record.year}-{month} "
        f"monthly total ({mm:.2f} mm) is a TIME-AVERAGED mean, NOT a design "
        f"storm. Do not use for engineering decisions.",
        RainfallDataWarning,
        stacklevel=2,
    )
    label = f"DERIVED:TEMPORAL_DISAGGREGATE (source: {record.provenance}, year={record.year}, month={month})"
    return intensity, label


# ---------------------------------------------------------------------------
# Bundled test fixture (offline — no internet required)
# ---------------------------------------------------------------------------

#: Minimal synthetic CSV for tests. Labelled SYNTHETIC — not real data.
SYNTHETIC_FIXTURE_CSV = """\
Year,Jan,Feb,Mar,Apr,May,Jun,Jul,Aug,Sep,Oct,Nov,Dec,Total
2005,0.0,0.0,0.0,0.0,12.5,641.2,1034.0,585.3,280.1,45.2,0.0,0.0,2598.3
2006,0.0,0.0,0.0,0.0,18.7,320.4,892.5,601.8,195.4,30.1,0.0,0.0,2058.9
2007,1.2,0.0,0.0,0.0,0.0,450.3,1011.2,723.4,310.2,12.4,0.0,0.0,2508.7
"""

#: DatasetRecord for the bundled synthetic fixture.
SYNTHETIC_FIXTURE_RECORD = DatasetRecord(
    provenance="SYNTHETIC",
    name="FloodCut Phase 5 synthetic rainfall fixture (test only)",
    source_organisation="FloodCut project team",
    source_url="",
    retrieval_date=None,
    license_notes="Internal test fixture. NOT real data.",
    geographic_coverage="N/A",
    observation_period="2005–2007 (synthetic)",
    crs="N/A",
    notes=(
        "Three rows of synthetic monthly rainfall data created for unit tests. "
        "Values are plausible for Mumbai monsoon conditions but are NOT "
        "observations. Do not use for any real analysis."
    ),
)


def load_synthetic_fixture() -> RainfallDataset:
    """Load the bundled synthetic test fixture (no internet required).

    Returns a ``RainfallDataset`` labelled ``SYNTHETIC`` for use in offline
    tests and demonstrations.
    """
    return load_rainfall_csv_string(
        SYNTHETIC_FIXTURE_CSV,
        dataset_record=SYNTHETIC_FIXTURE_RECORD,
        station_id="SYNTHETIC_FIXTURE",
        provenance="SYNTHETIC",
    )
