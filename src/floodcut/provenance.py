"""Phase 5: Data provenance labels and metadata.

Every dataset, record, or derived value in FloodCut is tagged with one of the
provenance categories below. This module defines the canonical labels, the
DatasetRecord container, and helper utilities so that downstream code can
always distinguish real observations from model assumptions or synthetic test
fixtures.

Provenance hierarchy
--------------------
REAL        Obtained from a verifiable external source. Provenance (source
            organisation, URL, retrieval date) is recorded. Do NOT use for
            any value that has been transformed or estimated — use DERIVED.

DERIVED     Computed or estimated from a REAL dataset via a documented method
            (e.g. spatial assignment, unit conversion, temporal aggregation,
            climatological normal). The method and its assumptions are stored
            in ``notes``.

ASSUMED     A value decided by the modeller without an empirical source; used
            when measurements are unavailable (e.g. synthetic drain capacities,
            invented runoff coefficients). The rationale is in ``notes``.

SYNTHETIC   Artificial data created for testing or demonstration of the
            algorithm (e.g. a hand-crafted network or a fixture CSV). Never
            claim to represent real-world observations.

Sub-categories (optional)
-------------------------
Callers can extend the label with a colon-separated qualifier, e.g.
``"REAL:IMD_GRID"`` or ``"DERIVED:TEMPORAL_AGGREGATE"``.  The base category
is always the prefix before the colon.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Optional

# ---------------------------------------------------------------------------
# Canonical labels
# ---------------------------------------------------------------------------

REAL = "REAL"
DERIVED = "DERIVED"
ASSUMED = "ASSUMED"
SYNTHETIC = "SYNTHETIC"

_VALID_BASES = {REAL, DERIVED, ASSUMED, SYNTHETIC}


def base_category(label: str) -> str:
    """Return the base provenance category ('REAL', 'DERIVED', etc.).

    >>> base_category('REAL:IMD_GRID')
    'REAL'
    >>> base_category('SYNTHETIC')
    'SYNTHETIC'
    """
    return label.split(":")[0].upper()


def is_valid(label: str) -> bool:
    """True iff the label's base is one of the four canonical categories."""
    return base_category(label) in _VALID_BASES


def assert_valid(label: str) -> None:
    """Raise ValueError if label is not a recognised provenance label."""
    if not is_valid(label):
        raise ValueError(
            f"Unknown provenance label {label!r}. "
            f"Base must be one of {sorted(_VALID_BASES)}"
        )


# ---------------------------------------------------------------------------
# Dataset-level record
# ---------------------------------------------------------------------------

@dataclass
class DatasetRecord:
    """Provenance metadata for an entire dataset.

    Attributes
    ----------
    provenance : str
        A provenance label such as ``REAL``, ``REAL:IMD_GRID``,
        ``SYNTHETIC``, etc.
    name : str
        Human-readable name of the dataset.
    source_organisation : str
        Organisation that produced or published the data (e.g. ``"India
        Meteorological Department (IMD)"``). Empty string if unknown.
    source_url : str
        URL where the data was obtained, or ``""`` if not applicable.
    retrieval_date : date | None
        Date on which the data was accessed.  ``None`` for synthetic data.
    license_notes : str
        Usage restrictions or license information, if known.  Empty string
        if not stated.
    geographic_coverage : str
        Textual description of the geographic area covered.
    observation_period : str
        Time span of the observations (e.g. ``"1901–2021"``), or ``""``
        for synthetic data.
    crs : str
        Coordinate Reference System, e.g. ``"WGS84 (EPSG:4326)"``.
        Empty string if not applicable.
    notes : str
        Free-text notes: assumptions, known limitations, transformation
        methods, or any caveat the user should be aware of.
    """
    provenance: str
    name: str
    source_organisation: str = ""
    source_url: str = ""
    retrieval_date: Optional[date] = None
    license_notes: str = ""
    geographic_coverage: str = ""
    observation_period: str = ""
    crs: str = ""
    notes: str = ""

    def __post_init__(self) -> None:
        assert_valid(self.provenance)

    def is_real(self) -> bool:
        """True if this dataset was sourced from an external verifiable source."""
        return base_category(self.provenance) == REAL

    def is_synthetic(self) -> bool:
        return base_category(self.provenance) == SYNTHETIC

    def summary(self) -> str:
        """Single-line human-readable summary."""
        retrieval = (
            self.retrieval_date.isoformat() if self.retrieval_date else "N/A"
        )
        return (
            f"[{self.provenance}] {self.name} | "
            f"source={self.source_organisation or 'unknown'} | "
            f"retrieved={retrieval} | "
            f"period={self.observation_period or 'N/A'}"
        )


# ---------------------------------------------------------------------------
# Pre-defined dataset records
# ---------------------------------------------------------------------------

#: OpenCity / IMD monthly grid rainfall for the Mumbai grid point.
OPENCITY_IMD_MONTHLY = DatasetRecord(
    provenance="REAL:IMD_GRID",
    name="Mumbai Monthly Rainfall 1901–2021 (IMD Grid Point)",
    source_organisation="India Meteorological Department (IMD), via OpenCity India",
    source_url=(
        "https://data.opencity.in/dataset/mumbai-rainfall-data"
    ),
    retrieval_date=date(2026, 10, 23),
    license_notes=(
        "Published on OpenCity India (data.opencity.in). IMD data is generally "
        "available for non-commercial research use; formal licensing terms are not "
        "stated on the dataset page. Users should verify current terms at "
        "https://data.opencity.in/dataset/mumbai-rainfall-data before redistribution."
    ),
    geographic_coverage=(
        "Single IMD 0.25° grid point approximately centred on Mumbai "
        "(19.00° N, 72.75° E); covers Greater Mumbai and vicinity."
    ),
    observation_period="1901–2021",
    crs="WGS84 (EPSG:4326) — grid cell centre coordinates",
    notes=(
        "Monthly accumulated rainfall in mm derived from IMD's gridded dataset "
        "(0.25° × 0.25° spatial resolution). This is a GRID point, not an "
        "individual AWS / raingauge station. Values represent the areal mean "
        "over the grid cell, not a point observation. "
        "LIMITATION: Monthly totals cannot be directly used as an instantaneous "
        "intensity input (mm/h) for the rational-method runoff calculation without "
        "a temporal disaggregation assumption — see rainfall_loader.py for the "
        "method used and its caveats."
    ),
)

#: Simulated synthetic network — carried over from Phases 1–4.
SIMULATED_NETWORK = DatasetRecord(
    provenance="SYNTHETIC",
    name="FloodCut synthetic drainage network (Mumbai-style)",
    source_organisation="FloodCut project team",
    source_url="",
    retrieval_date=None,
    license_notes="Internal project asset.",
    geographic_coverage="Illustrative positions within Greater Mumbai bounding box.",
    observation_period="",
    crs="WGS84 (EPSG:4326) — illustrative coordinates, NOT surveyed",
    notes=(
        "Hand-designed 14-junction, 14-drain, 6-catchment, 2-outfall network "
        "for algorithm development. All capacities, areas, runoff coefficients, "
        "repair costs, and coordinates are invented."
    ),
)
