"""JSON loading/saving with explicit provenance and clear error messages."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from .models import (Catchment, CriticalAsset, DrainageNetwork, Drain, Junction, ModelError,
                     NetworkMetadata, Outfall, RainfallScenario)


class DataError(ValueError):
    """Raised when a data file is malformed (distinct from model value errors)."""


def _read_json(path: str | Path) -> dict[str, Any]:
    p = Path(path)
    try:
        with p.open(encoding="utf-8") as fh:
            data = json.load(fh)
    except FileNotFoundError as exc:
        raise DataError(f"File not found: {p}") from exc
    except json.JSONDecodeError as exc:
        raise DataError(f"Invalid JSON in {p}: {exc}") from exc
    if not isinstance(data, dict):
        raise DataError(f"{p}: top level must be a JSON object")
    return data


def _build(cls, rec: Any, where: str):
    if not isinstance(rec, dict):
        raise DataError(f"{where}: expected an object, got {type(rec).__name__}")
    try:
        return cls(**rec)
    except (TypeError, ModelError) as exc:
        raise DataError(f"{where}: {exc}") from exc


def _collect(cls, records: Any, key_of, where: str) -> dict[str, Any]:
    if not isinstance(records, list):
        raise DataError(f"'{where}' must be a list")
    out: dict[str, Any] = {}
    for idx, rec in enumerate(records):
        obj = _build(cls, rec, f"{where}[{idx}]")
        key = key_of(obj)
        if key in out:
            raise DataError(f"{where}[{idx}]: duplicate id {key!r}")
        out[key] = obj
    return out


def load_network(path: str | Path) -> DrainageNetwork:
    data = _read_json(path)
    meta_rec = data.get("metadata")
    if not isinstance(meta_rec, dict) or "is_simulated" not in meta_rec:
        raise DataError("metadata with an explicit 'is_simulated' flag is required")
    for required in ("junctions", "drains", "catchments", "outfalls"):
        if required not in data:
            raise DataError(f"Missing required section '{required}'")
    return DrainageNetwork(
        metadata=_build(NetworkMetadata, meta_rec, "metadata"),
        junctions=_collect(Junction, data["junctions"], lambda o: o.id, "junctions"),
        drains=_collect(Drain, data["drains"], lambda o: o.id, "drains"),
        catchments=_collect(Catchment, data["catchments"], lambda o: o.id, "catchments"),
        outfalls=_collect(Outfall, data["outfalls"], lambda o: o.junction_id, "outfalls"),
        assets=_collect(CriticalAsset, data.get("critical_assets", []), lambda o: o.id,
                        "critical_assets"),
    )


def save_network(network: DrainageNetwork, path: str | Path) -> None:
    payload = {
        "metadata": asdict(network.metadata),
        "junctions": [asdict(o) for o in network.junctions.values()],
        "drains": [asdict(o) for o in network.drains.values()],
        "catchments": [asdict(o) for o in network.catchments.values()],
        "outfalls": [asdict(o) for o in network.outfalls.values()],
        "critical_assets": [asdict(o) for o in network.assets.values()],
    }
    Path(path).write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def load_scenarios(path: str | Path) -> dict[str, RainfallScenario]:
    data = _read_json(path)
    if "scenarios" not in data:
        raise DataError("Missing required section 'scenarios'")
    return _collect(RainfallScenario, data["scenarios"], lambda o: o.id, "scenarios")
