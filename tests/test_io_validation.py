import json

import pytest

from floodcut.io import DataError, load_network, load_scenarios, save_network
from floodcut.validation import errors, validate_network

from conftest import make_network


def test_sample_data_is_flagged_simulated(sim_network):
    assert sim_network.metadata.is_simulated is True
    assert sim_network.counts() == {"junctions": 14, "drains": 14, "catchments": 6,
                                    "outfalls": 2, "critical_assets": 4}


def test_sample_data_validates_without_errors_or_warnings(sim_network):
    assert validate_network(sim_network) == []


def test_scenarios_load(sim_scenarios):
    assert set(sim_scenarios) == {"light", "design", "heavy", "extreme"}
    assert sim_scenarios["extreme"].tide_factor == 0.5


def test_round_trip(sim_network, tmp_path):
    out = tmp_path / "net.json"
    save_network(sim_network, out)
    again = load_network(out)
    assert again.counts() == sim_network.counts()
    assert again.drains["D05"] == sim_network.drains["D05"]
    assert again.metadata.is_simulated is True


def test_missing_is_simulated_flag_is_rejected(tmp_path):
    p = tmp_path / "n.json"
    p.write_text(json.dumps({"metadata": {"name": "x", "source": "s"}, "junctions": [],
                             "drains": [], "catchments": [], "outfalls": []}))
    with pytest.raises(DataError, match="is_simulated"):
        load_network(p)


def test_duplicate_ids_and_bad_values_are_rejected(tmp_path, sim_network):
    save_network(sim_network, tmp_path / "ok.json")
    data = json.loads((tmp_path / "ok.json").read_text())

    dup = json.loads(json.dumps(data)); dup["drains"].append(dup["drains"][0])
    (tmp_path / "dup.json").write_text(json.dumps(dup))
    with pytest.raises(DataError, match="duplicate id"):
        load_network(tmp_path / "dup.json")

    bad = json.loads(json.dumps(data)); bad["drains"][0]["design_capacity_m3s"] = -3
    (tmp_path / "bad.json").write_text(json.dumps(bad))
    with pytest.raises(DataError, match="design_capacity_m3s"):
        load_network(tmp_path / "bad.json")


def test_missing_file_and_invalid_json(tmp_path):
    with pytest.raises(DataError, match="not found"):
        load_network(tmp_path / "nope.json")
    (tmp_path / "x.json").write_text("{not json")
    with pytest.raises(DataError, match="Invalid JSON"):
        load_network(tmp_path / "x.json")


def test_validation_flags_referential_errors():
    net = make_network(["A", "B", "T"], [("D1", "A", "B", 1.0), ("D2", "B", "GHOST", 1.0)],
                       [("C1", "A", 10, 0.5), ("C2", "NOPE", 10, 0.5)], [("T", 5.0)])
    codes = {i.code for i in errors(validate_network(net))}
    assert {"E_DRAIN_ENDPOINT", "E_CATCHMENT_JUNCTION"} <= codes


def test_validation_flags_outfall_with_outflow_and_no_outfall():
    net = make_network(["A", "T"], [("D1", "T", "A", 1.0)], [("C1", "A", 10, 0.5)], [("T", 5.0)])
    assert "E_OUTFALL_HAS_OUTFLOW" in {i.code for i in errors(validate_network(net))}
    net2 = make_network(["A", "B"], [("D1", "A", "B", 1.0)], [("C1", "A", 10, 0.5)], [])
    assert "E_NO_OUTFALL" in {i.code for i in errors(validate_network(net2))}


def test_validation_warns_on_unreachable_catchment_isolated_junction_and_cycle():
    net = make_network(["A", "B", "T", "X"], [("D1", "A", "B", 1.0), ("D2", "B", "A", 1.0)],
                       [("C1", "A", 10, 0.5)], [("T", 5.0)])
    codes = {i.code for i in validate_network(net)}
    assert {"W_CATCHMENT_NO_PATH", "W_ISOLATED_JUNCTION", "W_CYCLE"} <= codes
    assert not errors(validate_network(net))
