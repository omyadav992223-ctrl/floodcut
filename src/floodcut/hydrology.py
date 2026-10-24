"""Rainfall-to-runoff demand (Phase 1: steady-state rational method only)."""

from __future__ import annotations

import math


def rational_method_m3s(runoff_coefficient: float, intensity_mm_h: float, area_ha: float) -> float:
    """Peak runoff Q = C * i * A / 360, in m^3/s.

    Units: i in mm/h, A in hectares. The constant 360 converts
    (mm/h * ha) to m^3/s:  1 mm/h over 1 ha = 10 m^3/h = 1/360 m^3/s.

    Assumptions (see docs/ASSUMPTIONS.md): uniform intensity over the
    catchment, steady state, no time of concentration, no storage.
    """
    for name, v in (("runoff_coefficient", runoff_coefficient),
                    ("intensity_mm_h", intensity_mm_h),
                    ("area_ha", area_ha)):
        if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or v < 0:
            raise ValueError(f"{name} must be a finite number >= 0, got {v!r}")
    return runoff_coefficient * intensity_mm_h * area_ha / 360.0
