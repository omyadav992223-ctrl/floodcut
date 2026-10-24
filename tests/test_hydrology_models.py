import pytest

from floodcut.hydrology import rational_method_m3s
from floodcut.models import (Catchment, Drain, Junction, ModelError, NetworkMetadata, Outfall,
                             RainfallScenario)


def test_rational_method_hand_calculation():
    # C=0.8, i=100 mm/h, A=50 ha -> 0.8*100*50/360 = 11.111...
    assert rational_method_m3s(0.8, 100, 50) == pytest.approx(11.1111, rel=1e-4)
    assert rational_method_m3s(0.8, 25, 50) == pytest.approx(2.7778, rel=1e-4)
    assert rational_method_m3s(0.8, 0, 50) == 0.0


@pytest.mark.parametrize("args", [(-0.1, 10, 1), (0.5, -1, 1), (0.5, 10, -1), (0.5, float("nan"), 1)])
def test_rational_method_rejects_bad_input(args):
    with pytest.raises(ValueError):
        rational_method_m3s(*args)


def test_drain_effective_capacity():
    assert Drain("D1", "A", "B", 4.0, 0.5).effective_capacity_m3s == pytest.approx(2.0)
    assert Drain("D1", "A", "B", 4.0, 0.0).effective_capacity_m3s == 0.0


@pytest.mark.parametrize("kwargs", [
    dict(id="D1", from_junction="A", to_junction="A", design_capacity_m3s=1),      # self-loop
    dict(id="D1", from_junction="A", to_junction="B", design_capacity_m3s=-1),     # negative cap
    dict(id="D1", from_junction="A", to_junction="B", design_capacity_m3s=1, capacity_factor=1.5),
    dict(id="D:1", from_junction="A", to_junction="B", design_capacity_m3s=1),     # reserved char
    dict(id="D1", from_junction="A", to_junction="B", design_capacity_m3s=True),   # bool is not a number
])
def test_drain_rejects_invalid(kwargs):
    with pytest.raises(ModelError):
        Drain(**kwargs)


def test_catchment_and_outfall_and_scenario_validation():
    with pytest.raises(ModelError):
        Catchment("C1", "c", "J1", area_ha=0, runoff_coefficient=0.5)
    with pytest.raises(ModelError):
        Catchment("C1", "c", "J1", area_ha=1, runoff_coefficient=1.2)
    with pytest.raises(ModelError):
        RainfallScenario("s", "s", 10, tide_factor=2)
    with pytest.raises(ModelError):
        Junction("J1", "j", lat=95, lon=72)
    with pytest.raises(ModelError):
        NetworkMetadata(name="x", is_simulated="yes", source="s")  # must be a real bool


def test_outfall_tide_scaling():
    tidal = Outfall("O", "o", 10.0, tidal=True)
    fixed = Outfall("O", "o", 10.0, tidal=False)
    assert tidal.effective_capacity_m3s(0.5) == pytest.approx(5.0)
    assert fixed.effective_capacity_m3s(0.5) == pytest.approx(10.0)
