"""Core domain models for FloodCut.

All models are immutable dataclasses that validate their own scalar fields at
construction time. Cross-record checks (e.g. a drain referencing a junction
that does not exist) live in `validation.py`.

Units
-----
capacity / flow : m^3/s        area : hectares        rainfall : mm/h
cost            : abstract "cost units" (not rupees; no real cost data used)
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Optional

from .hydrology import rational_method_m3s

# Ids are used to build graph node names such as "drain:D01", so they must not
# contain ':' or '*' (reserved for the super-source "S*" and super-sink "T*").
ID_PATTERN = re.compile(r"^[A-Za-z0-9_.\-]+$")


class ModelError(ValueError):
    """Raised when a model is constructed with invalid values."""


def _check_id(label: str, value: object) -> None:
    if not isinstance(value, str) or not ID_PATTERN.match(value):
        raise ModelError(f"{label} must match {ID_PATTERN.pattern!r}, got {value!r}")


def _check_num(label: str, value: object, lo: Optional[float] = None, hi: Optional[float] = None,
               lo_open: bool = False) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ModelError(f"{label} must be a finite number, got {value!r}")
    if lo is not None and (value <= lo if lo_open else value < lo):
        raise ModelError(f"{label} must be {'>' if lo_open else '>='} {lo}, got {value!r}")
    if hi is not None and value > hi:
        raise ModelError(f"{label} must be <= {hi}, got {value!r}")


@dataclass(frozen=True)
class Junction:
    """A node of the drainage network (inlet, manhole, confluence, outfall point)."""
    id: str
    name: str
    lat: float
    lon: float
    ground_elevation_m: Optional[float] = None  # reserved for later phases (DEM-derived direction)

    def __post_init__(self) -> None:
        _check_id("Junction.id", self.id)
        _check_num("Junction.lat", self.lat, -90, 90)
        _check_num("Junction.lon", self.lon, -180, 180)
        if self.ground_elevation_m is not None:
            _check_num("Junction.ground_elevation_m", self.ground_elevation_m)


@dataclass(frozen=True)
class Drain:
    """A directed drain (pipe, box drain, or nallah reach) from one junction to another.

    `design_capacity_m3s` is the healthy capacity. `capacity_factor` in [0, 1] is the
    current operating fraction (1.0 healthy, 0.0 fully blocked). Effective capacity is
    their product. Capacity is treated as a constant (see docs/ASSUMPTIONS.md, A4).
    """
    id: str
    from_junction: str
    to_junction: str
    design_capacity_m3s: float
    capacity_factor: float = 1.0
    repair_cost: float = 0.0
    length_m: Optional[float] = None
    kind: str = "closed_drain"

    def __post_init__(self) -> None:
        _check_id("Drain.id", self.id)
        _check_id("Drain.from_junction", self.from_junction)
        _check_id("Drain.to_junction", self.to_junction)
        if self.from_junction == self.to_junction:
            raise ModelError(f"Drain {self.id}: self-loop on junction {self.from_junction}")
        _check_num("Drain.design_capacity_m3s", self.design_capacity_m3s, 0)
        _check_num("Drain.capacity_factor", self.capacity_factor, 0, 1)
        _check_num("Drain.repair_cost", self.repair_cost, 0)
        if self.length_m is not None:
            _check_num("Drain.length_m", self.length_m, 0, lo_open=True)

    @property
    def effective_capacity_m3s(self) -> float:
        return self.design_capacity_m3s * self.capacity_factor


@dataclass(frozen=True)
class Catchment:
    """Area whose runoff enters the network at one junction (the rainfall demand source)."""
    id: str
    name: str
    junction_id: str
    area_ha: float
    runoff_coefficient: float

    def __post_init__(self) -> None:
        _check_id("Catchment.id", self.id)
        _check_id("Catchment.junction_id", self.junction_id)
        _check_num("Catchment.area_ha", self.area_ha, 0, lo_open=True)
        _check_num("Catchment.runoff_coefficient", self.runoff_coefficient, 0, 1, lo_open=True)

    def runoff_m3s(self, intensity_mm_h: float) -> float:
        """Rainfall demand of this catchment at a uniform intensity."""
        return rational_method_m3s(self.runoff_coefficient, intensity_mm_h, self.area_ha)


@dataclass(frozen=True)
class Outfall:
    """A junction where water leaves the network (sea, river). Keyed by its junction id.

    If `tidal`, the scenario tide factor scales its capacity (1.0 = low tide, 0.0 = fully
    backed up). This is a deliberate simplification (docs/ASSUMPTIONS.md, A5).
    """
    junction_id: str
    name: str
    base_capacity_m3s: float
    tidal: bool = True

    def __post_init__(self) -> None:
        _check_id("Outfall.junction_id", self.junction_id)
        _check_num("Outfall.base_capacity_m3s", self.base_capacity_m3s, 0)
        if not isinstance(self.tidal, bool):
            raise ModelError(f"Outfall.tidal must be a bool, got {self.tidal!r}")

    def effective_capacity_m3s(self, tide_factor: float) -> float:
        _check_num("tide_factor", tide_factor, 0, 1)
        return self.base_capacity_m3s * (tide_factor if self.tidal else 1.0)


@dataclass(frozen=True)
class CriticalAsset:
    """Hospital, underpass, station... linked to the catchment whose overflow may affect it.
    Defined now so the schema is stable; it is not analysed until Phase 5."""
    id: str
    name: str
    kind: str
    catchment_id: str

    def __post_init__(self) -> None:
        _check_id("CriticalAsset.id", self.id)
        _check_id("CriticalAsset.catchment_id", self.catchment_id)


@dataclass(frozen=True)
class RainfallScenario:
    """A rainfall + tide scenario. Scenarios here are assumptions, not forecasts."""
    id: str
    name: str
    intensity_mm_h: float
    tide_factor: float = 1.0
    description: str = ""

    def __post_init__(self) -> None:
        _check_id("RainfallScenario.id", self.id)
        _check_num("RainfallScenario.intensity_mm_h", self.intensity_mm_h, 0)
        _check_num("RainfallScenario.tide_factor", self.tide_factor, 0, 1)


@dataclass(frozen=True)
class NetworkMetadata:
    """Provenance of a dataset. `is_simulated` is mandatory and has no default on purpose."""
    name: str
    is_simulated: bool
    source: str
    description: str = ""
    disclaimer: str = ""
    schema_version: int = 1

    def __post_init__(self) -> None:
        if not isinstance(self.is_simulated, bool):
            raise ModelError("NetworkMetadata.is_simulated must be an explicit bool")


@dataclass(eq=False)
class DrainageNetwork:
    """Container for one dataset. Holds records keyed by id; contains no graph logic."""
    metadata: NetworkMetadata
    junctions: dict[str, Junction] = field(default_factory=dict)
    drains: dict[str, Drain] = field(default_factory=dict)
    catchments: dict[str, Catchment] = field(default_factory=dict)
    outfalls: dict[str, Outfall] = field(default_factory=dict)  # key = junction_id
    assets: dict[str, CriticalAsset] = field(default_factory=dict)

    def counts(self) -> dict[str, int]:
        return {"junctions": len(self.junctions), "drains": len(self.drains),
                "catchments": len(self.catchments), "outfalls": len(self.outfalls),
                "critical_assets": len(self.assets)}
