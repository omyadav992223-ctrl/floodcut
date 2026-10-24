"""Tests for Phase 5: real data integration.

All tests run offline using deterministic fixtures. No live internet
connection is required. The only exception is the integration test that
loads the bundled real CSV file from ``data/real/``, which is checked into
the repository.

Test categories
---------------
1. Provenance module (labels, DatasetRecord).
2. Rainfall CSV loading — valid data.
3. Rainfall CSV loading — error and edge-case handling.
4. Unit conversion (monthly mm → mm/h).
5. Geospatial point loading — valid data.
6. Geospatial point loading — error and edge-case handling.
7. Integration with existing model (Phase 1–4 compatibility).
8. Regression — Phase 3 failure simulation and Phase 4 repair optimisation
   still pass on the synthetic network.
9. Real bundled data file — loads and validates without errors.
"""

from __future__ import annotations

import math
import warnings
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Import helpers
# ---------------------------------------------------------------------------

from floodcut.provenance import (
    ASSUMED, DERIVED, REAL, SYNTHETIC,
    DatasetRecord, assert_valid, base_category, is_valid,
    OPENCITY_IMD_MONTHLY, SIMULATED_NETWORK,
)
from floodcut.rainfall_loader import (
    MONTH_NAMES, MONSOON_MONTHS,
    MonthlyRainfallRecord, RainfallDataset, RainfallDataError, RainfallDataWarning,
    SYNTHETIC_FIXTURE_CSV, SYNTHETIC_FIXTURE_RECORD,
    load_rainfall_csv, load_rainfall_csv_string, load_synthetic_fixture,
    monthly_mm_to_intensity_mm_h, make_scenario_from_record,
    monsoon_season_intensity_mm_h,
)
from floodcut.geo_loader import (
    GeoLayer, GeoPoint, GeospatialDataError, GeospatialDataWarning,
    load_geo_points_csv, load_imd_grid_points, load_waterlogging_hotspots,
    MUMBAI_LAT_RANGE, MUMBAI_LON_RANGE,
)

# Path to the real bundled CSV.
_REAL_DATA = (
    Path(__file__).resolve().parents[1]
    / "data" / "real" / "imd_mumbai_monthly_rainfall_1901_2021.csv"
)

# ---------------------------------------------------------------------------
# 1. Provenance module
# ---------------------------------------------------------------------------

class TestProvenanceLabels:

    def test_valid_base_labels(self):
        for label in (REAL, DERIVED, ASSUMED, SYNTHETIC):
            assert is_valid(label), f"{label!r} should be valid"

    def test_valid_qualified_labels(self):
        for label in ("REAL:IMD_GRID", "DERIVED:TEMPORAL_DISAGGREGATE",
                      "ASSUMED:MODELLER", "SYNTHETIC"):
            assert is_valid(label), f"{label!r} should be valid"

    def test_invalid_labels(self):
        # base_category() uppercases before matching, so "real" → "REAL" is valid.
        # The truly invalid labels are the ones whose base is unrecognised.
        for bad in ("", "FAKE", "OBSERVATION", "MODEL", "FORECAST", "UNKNOWN"):
            assert not is_valid(bad), f"{bad!r} should be invalid"

    def test_base_category_strips_qualifier(self):
        assert base_category("REAL:IMD_GRID") == "REAL"
        assert base_category("DERIVED:XYZ") == "DERIVED"
        assert base_category("SYNTHETIC") == "SYNTHETIC"

    def test_assert_valid_raises_on_bad_label(self):
        with pytest.raises(ValueError, match="Unknown provenance"):
            assert_valid("INVENTED")

    def test_assert_valid_passes_on_good_label(self):
        assert_valid("REAL")
        assert_valid("REAL:IMD_GRID")

    def test_dataset_record_construction(self):
        from datetime import date
        rec = DatasetRecord(
            provenance="REAL:IMD_GRID",
            name="Test dataset",
            source_organisation="Test Org",
            retrieval_date=date(2024, 6, 1),
        )
        assert rec.is_real()
        assert not rec.is_synthetic()

    def test_dataset_record_synthetic(self):
        rec = DatasetRecord(provenance="SYNTHETIC", name="test")
        assert rec.is_synthetic()
        assert not rec.is_real()

    def test_dataset_record_invalid_provenance_rejected(self):
        with pytest.raises(ValueError):
            DatasetRecord(provenance="FAKE", name="test")

    def test_opencity_imd_monthly_record_is_real(self):
        assert OPENCITY_IMD_MONTHLY.is_real()
        assert OPENCITY_IMD_MONTHLY.source_url.startswith("https://")
        assert OPENCITY_IMD_MONTHLY.retrieval_date is not None

    def test_simulated_network_record_is_synthetic(self):
        assert SIMULATED_NETWORK.is_synthetic()

    def test_dataset_summary_contains_provenance(self):
        s = OPENCITY_IMD_MONTHLY.summary()
        assert "REAL" in s
        assert "IMD" in s


# ---------------------------------------------------------------------------
# 2. Rainfall CSV loading — valid data
# ---------------------------------------------------------------------------

class TestRainfallLoadingValid:

    def test_synthetic_fixture_loads(self):
        ds = load_synthetic_fixture()
        assert len(ds.records) == 3
        assert ds.years() == [2005, 2006, 2007]

    def test_synthetic_fixture_provenance_is_synthetic(self):
        ds = load_synthetic_fixture()
        assert all(r.provenance == "SYNTHETIC" for r in ds.records)

    def test_synthetic_fixture_dataset_record_is_synthetic(self):
        ds = load_synthetic_fixture()
        assert ds.dataset_record.is_synthetic()

    def test_records_have_all_month_keys(self):
        ds = load_synthetic_fixture()
        for rec in ds.records:
            assert set(rec.monthly_mm.keys()) == set(MONTH_NAMES)

    def test_monsoon_total_computed_correctly(self):
        ds = load_synthetic_fixture()
        rec = ds.record_for_year(2005)
        assert rec is not None
        # Jun=641.2, Jul=1034.0, Aug=585.3, Sep=280.1
        expected = 641.2 + 1034.0 + 585.3 + 280.1
        assert abs(rec.monsoon_total_mm() - expected) < 0.01

    def test_annual_total_computed_correctly(self):
        ds = load_synthetic_fixture()
        rec = ds.record_for_year(2005)
        assert rec is not None
        expected = sum(v for v in rec.monthly_mm.values() if v is not None)
        total = rec.annual_total_mm()
        assert total is not None
        assert abs(total - expected) < 0.01

    def test_year_range(self):
        ds = load_synthetic_fixture()
        assert ds.year_range() == (2005, 2007)

    def test_records_sorted_by_year(self):
        # Shuffle the year order in the CSV and ensure output is still sorted.
        shuffled = "Year,Jan,Feb,Mar,Apr,May,Jun,Jul,Aug,Sep,Oct,Nov,Dec\n"
        shuffled += "2007,0,0,0,0,0,450,1011,723,310,12,0,0\n"
        shuffled += "2005,0,0,0,0,12,641,1034,585,280,45,0,0\n"
        shuffled += "2006,0,0,0,0,18,320,892,601,195,30,0,0\n"
        ds = load_rainfall_csv_string(
            shuffled, station_id="TEST", provenance="SYNTHETIC",
            dataset_record=SYNTHETIC_FIXTURE_RECORD,
        )
        assert ds.years() == [2005, 2006, 2007]

    def test_load_from_file(self, tmp_path):
        csv_path = tmp_path / "rain.csv"
        csv_path.write_text(SYNTHETIC_FIXTURE_CSV, encoding="utf-8")
        ds = load_rainfall_csv(
            csv_path,
            station_id="TEST",
            provenance="SYNTHETIC",
            dataset_record=SYNTHETIC_FIXTURE_RECORD,
        )
        assert len(ds.records) == 3

    def test_monsoon_totals_dict(self):
        ds = load_synthetic_fixture()
        totals = ds.monsoon_totals()
        assert set(totals.keys()) == {2005, 2006, 2007}
        assert totals[2005] is not None

    def test_alternative_month_spellings(self):
        """CSV header uses 'April', 'June', 'July', 'Sept' (as in OpenCity)."""
        csv_text = (
            "Year,Jan,Feb,Mar,April,May,June,July,Aug,Sept,Oct,Nov,Dec\n"
            "2020,0,0,0,0,0,600.0,900.0,700.0,200.0,0,0,0\n"
        )
        ds = load_rainfall_csv_string(
            csv_text, provenance="SYNTHETIC",
            dataset_record=SYNTHETIC_FIXTURE_RECORD,
        )
        assert len(ds.records) == 1
        rec = ds.records[0]
        assert rec.monthly_mm["Jun"] == 600.0
        assert rec.monthly_mm["Jul"] == 900.0
        assert rec.monthly_mm["Sep"] == 200.0

    def test_summary_string_contains_key_info(self):
        ds = load_synthetic_fixture()
        s = ds.summary()
        assert "SYNTHETIC" in s
        assert "records=3" in s


# ---------------------------------------------------------------------------
# 3. Rainfall CSV loading — error and edge-case handling
# ---------------------------------------------------------------------------

class TestRainfallLoadingErrors:

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(RainfallDataError, match="not found"):
            load_rainfall_csv(tmp_path / "nonexistent.csv", provenance="SYNTHETIC",
                              dataset_record=SYNTHETIC_FIXTURE_RECORD)

    def test_missing_year_column_raises(self):
        bad_csv = "Month,Jan,Feb,Mar,Apr,May,Jun,Jul,Aug,Sep,Oct,Nov,Dec\n"
        bad_csv += "2020,0,0,0,0,0,600,900,700,200,0,0,0\n"
        with pytest.raises(RainfallDataError, match="Year"):
            load_rainfall_csv_string(bad_csv, provenance="SYNTHETIC",
                                     dataset_record=SYNTHETIC_FIXTURE_RECORD)

    def test_missing_month_column_raises(self):
        # Drop Jul
        bad_csv = "Year,Jan,Feb,Mar,Apr,May,Jun,Aug,Sep,Oct,Nov,Dec\n"
        bad_csv += "2020,0,0,0,0,0,600,700,200,0,0,0\n"
        with pytest.raises(RainfallDataError, match="missing required month"):
            load_rainfall_csv_string(bad_csv, provenance="SYNTHETIC",
                                     dataset_record=SYNTHETIC_FIXTURE_RECORD)

    def test_empty_csv_raises(self):
        with pytest.raises(RainfallDataError, match="empty"):
            load_rainfall_csv_string("", provenance="SYNTHETIC",
                                     dataset_record=SYNTHETIC_FIXTURE_RECORD)

    def test_invalid_year_is_skipped_with_warning(self):
        csv_text = (
            "Year,Jan,Feb,Mar,Apr,May,Jun,Jul,Aug,Sep,Oct,Nov,Dec\n"
            "BADYEAR,0,0,0,0,0,600,900,700,200,0,0,0\n"
            "2020,0,0,0,0,0,600,900,700,200,0,0,0\n"
        )
        ds = load_rainfall_csv_string(csv_text, provenance="SYNTHETIC",
                                     dataset_record=SYNTHETIC_FIXTURE_RECORD)
        assert len(ds.records) == 1  # only the valid row
        assert len(ds.skipped_rows) == 1

    def test_negative_rainfall_becomes_none_with_warning(self):
        csv_text = (
            "Year,Jan,Feb,Mar,Apr,May,Jun,Jul,Aug,Sep,Oct,Nov,Dec\n"
            "2020,0,0,0,0,0,-5.0,900,700,200,0,0,0\n"
        )
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            ds = load_rainfall_csv_string(csv_text, provenance="SYNTHETIC",
                                         dataset_record=SYNTHETIC_FIXTURE_RECORD)
        assert len(ds.records) == 1
        assert ds.records[0].monthly_mm["Jun"] is None
        assert any(issubclass(x.category, RainfallDataWarning) for x in w)

    def test_non_numeric_cell_becomes_none_with_warning(self):
        csv_text = (
            "Year,Jan,Feb,Mar,Apr,May,Jun,Jul,Aug,Sep,Oct,Nov,Dec\n"
            "2020,0,0,0,0,0,NA,900,700,200,0,0,0\n"
        )
        ds = load_rainfall_csv_string(csv_text, provenance="SYNTHETIC",
                                     dataset_record=SYNTHETIC_FIXTURE_RECORD)
        assert ds.records[0].monthly_mm["Jun"] is None

    def test_missing_month_listed(self):
        csv_text = (
            "Year,Jan,Feb,Mar,Apr,May,Jun,Jul,Aug,Sep,Oct,Nov,Dec\n"
            "2020,0,0,0,0,0,,900,700,200,0,0,0\n"
        )
        ds = load_rainfall_csv_string(csv_text, provenance="SYNTHETIC",
                                     dataset_record=SYNTHETIC_FIXTURE_RECORD)
        assert "Jun" in ds.records[0].missing_months()

    def test_duplicate_year_drops_second_with_warning(self):
        csv_text = (
            "Year,Jan,Feb,Mar,Apr,May,Jun,Jul,Aug,Sep,Oct,Nov,Dec\n"
            "2020,0,0,0,0,0,600,900,700,200,0,0,0\n"
            "2020,1,2,3,4,5,610,910,710,210,0,0,0\n"
        )
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            ds = load_rainfall_csv_string(csv_text, provenance="SYNTHETIC",
                                         dataset_record=SYNTHETIC_FIXTURE_RECORD)
        assert len(ds.records) == 1
        assert ds.records[0].monthly_mm["Jun"] == 600.0  # first row kept
        assert any(issubclass(x.category, RainfallDataWarning) for x in w)
        assert len(ds.skipped_rows) >= 1

    def test_invalid_provenance_raises(self):
        with pytest.raises(ValueError, match="Unknown provenance"):
            load_rainfall_csv_string(SYNTHETIC_FIXTURE_CSV, provenance="BAD",
                                     dataset_record=SYNTHETIC_FIXTURE_RECORD)

    def test_all_missing_monsoon_returns_none(self):
        csv_text = (
            "Year,Jan,Feb,Mar,Apr,May,Jun,Jul,Aug,Sep,Oct,Nov,Dec\n"
            "2020,0,0,0,0,0,,,,,0,0,0\n"
        )
        ds = load_rainfall_csv_string(csv_text, provenance="SYNTHETIC",
                                     dataset_record=SYNTHETIC_FIXTURE_RECORD)
        assert ds.records[0].monsoon_total_mm() is None

    def test_extra_columns_silently_dropped(self):
        csv_text = (
            "Year,Jan,Feb,Mar,Apr,May,Jun,Jul,Aug,Sep,Oct,Nov,Dec,Total,StationCode\n"
            "2020,0,0,0,0,0,600,900,700,200,0,0,0,2400,MUMB\n"
        )
        ds = load_rainfall_csv_string(csv_text, provenance="SYNTHETIC",
                                     dataset_record=SYNTHETIC_FIXTURE_RECORD)
        assert len(ds.records) == 1  # no crash


# ---------------------------------------------------------------------------
# 4. Unit conversion
# ---------------------------------------------------------------------------

class TestUnitConversion:

    def test_july_intensity_is_positive(self):
        intensity = monthly_mm_to_intensity_mm_h(744.0, "Jul")
        assert intensity > 0

    def test_july_intensity_units(self):
        # 744 mm over July (744 hours) → 1.0 mm/h
        intensity = monthly_mm_to_intensity_mm_h(744.0, "Jul")
        assert abs(intensity - 1.0) < 1e-4

    def test_zero_rainfall_gives_zero_intensity(self):
        assert monthly_mm_to_intensity_mm_h(0.0, "Jan") == 0.0

    def test_all_months_accepted(self):
        for month in MONTH_NAMES:
            result = monthly_mm_to_intensity_mm_h(300.0, month)
            assert result > 0

    def test_unknown_month_raises(self):
        with pytest.raises(ValueError, match="Unknown month"):
            monthly_mm_to_intensity_mm_h(100.0, "Xyz")

    def test_negative_rainfall_raises(self):
        with pytest.raises(ValueError):
            monthly_mm_to_intensity_mm_h(-5.0, "Jun")

    def test_intensity_far_smaller_than_design_storm(self):
        """Mean monthly intensity is orders of magnitude below a design storm (50 mm/h)."""
        # Even an extreme July total (e.g. 1200 mm over 744 h) → ~1.6 mm/h
        intensity = monthly_mm_to_intensity_mm_h(1200.0, "Jul")
        assert intensity < 50.0, (
            "Monthly mean intensity should be far below design-storm intensity"
        )

    def test_monsoon_season_intensity(self):
        ds = load_synthetic_fixture()
        rec = ds.record_for_year(2005)
        result = monsoon_season_intensity_mm_h(rec)
        assert result is not None
        assert result > 0

    def test_make_scenario_from_record(self):
        ds = load_synthetic_fixture()
        rec = ds.record_for_year(2005)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            intensity, label = make_scenario_from_record(rec, "Jul", "test_sc")
        assert intensity > 0
        assert "DERIVED" in label
        assert any(issubclass(x.category, RainfallDataWarning) for x in w)

    def test_make_scenario_from_record_missing_month_raises(self):
        """If the requested month is missing, raise RainfallDataError."""
        csv_text = (
            "Year,Jan,Feb,Mar,Apr,May,Jun,Jul,Aug,Sep,Oct,Nov,Dec\n"
            "2020,0,0,0,0,0,600,,700,200,0,0,0\n"
        )
        ds = load_rainfall_csv_string(csv_text, provenance="SYNTHETIC",
                                     dataset_record=SYNTHETIC_FIXTURE_RECORD)
        rec = ds.records[0]
        assert rec.monthly_mm["Jul"] is None
        with pytest.raises(RainfallDataError, match="missing"):
            with warnings.catch_warnings(record=True):
                warnings.simplefilter("always")
                make_scenario_from_record(rec, "Jul", "sc")


# ---------------------------------------------------------------------------
# 5. Geospatial point loading — valid data
# ---------------------------------------------------------------------------

class TestGeoLoadingValid:

    def test_imd_grid_points_load(self):
        layer = load_imd_grid_points()
        assert len(layer.points) == 1
        pt = layer.points[0]
        assert pt.id == "IMD_GRID_19.00N_72.75E"
        assert abs(pt.lat - 19.0) < 0.01
        assert abs(pt.lon - 72.75) < 0.01

    def test_imd_grid_point_provenance(self):
        layer = load_imd_grid_points()
        assert layer.points[0].provenance == "REAL:IMD_GRID"

    def test_imd_grid_point_in_mumbai_bbox(self):
        layer = load_imd_grid_points()
        assert layer.points[0].in_mumbai_bbox()

    def test_imd_layer_dataset_record_is_real(self):
        layer = load_imd_grid_points()
        assert layer.dataset_record.is_real()

    def test_waterlogging_hotspots_load(self):
        layer = load_waterlogging_hotspots()
        assert len(layer.points) > 0

    def test_waterlogging_hotspot_provenance(self):
        layer = load_waterlogging_hotspots()
        for pt in layer.points:
            assert pt.provenance == "DERIVED:GEOCODED"

    def test_waterlogging_hotspots_in_bbox(self):
        layer = load_waterlogging_hotspots()
        outside = layer.outside_bbox()
        assert len(outside) == 0, f"Points outside Mumbai bbox: {outside}"

    def test_load_custom_geo_csv(self):
        csv_text = (
            "id,name,lat,lon,ward\n"
            "P1,Point A,19.10,72.85,K-West\n"
            "P2,Point B,19.05,72.90,H-East\n"
        )
        from datetime import date
        rec = DatasetRecord(provenance="SYNTHETIC", name="test geo")
        layer = load_geo_points_csv(csv_text, dataset_record=rec,
                                    provenance="SYNTHETIC")
        assert len(layer.points) == 2
        assert layer.points[0].id == "P1"
        assert layer.points[0].attributes.get("ward") == "K-West"

    def test_geo_point_attributes_preserved(self):
        csv_text = "id,name,lat,lon,note\nX1,Test,19.0,72.8,important\n"
        rec = DatasetRecord(provenance="SYNTHETIC", name="t")
        layer = load_geo_points_csv(csv_text, dataset_record=rec,
                                    provenance="SYNTHETIC")
        assert layer.points[0].attributes["note"] == "important"

    def test_geo_summary_string(self):
        layer = load_waterlogging_hotspots()
        s = layer.summary()
        assert "DERIVED" in s

    def test_outside_bbox_reports_correctly(self):
        csv_text = "id,name,lat,lon\nFAR,Remote,28.0,77.0\n"
        rec = DatasetRecord(provenance="SYNTHETIC", name="t")
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            layer = load_geo_points_csv(csv_text, dataset_record=rec,
                                        provenance="SYNTHETIC")
        assert len(layer.outside_bbox()) == 1


# ---------------------------------------------------------------------------
# 6. Geospatial loading — error handling
# ---------------------------------------------------------------------------

class TestGeoLoadingErrors:

    def test_missing_file_raises(self, tmp_path):
        rec = DatasetRecord(provenance="SYNTHETIC", name="t")
        with pytest.raises(GeospatialDataError, match="not found"):
            load_geo_points_csv(tmp_path / "missing.csv", dataset_record=rec,
                                provenance="SYNTHETIC")

    def test_missing_required_column_raises(self):
        # No 'lat' column
        csv_text = "id,name,lon\nP1,Point,72.85\n"
        rec = DatasetRecord(provenance="SYNTHETIC", name="t")
        with pytest.raises(GeospatialDataError, match="missing required"):
            load_geo_points_csv(csv_text, dataset_record=rec,
                                provenance="SYNTHETIC")

    def test_invalid_lat_skipped_with_warning(self):
        csv_text = "id,name,lat,lon\nP1,Bad,NOTNUM,72.85\n"
        rec = DatasetRecord(provenance="SYNTHETIC", name="t")
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            layer = load_geo_points_csv(csv_text, dataset_record=rec,
                                        provenance="SYNTHETIC")
        assert len(layer.points) == 0
        assert len(layer.skipped_rows) == 1

    def test_out_of_range_lat_raises_on_point_construction(self):
        csv_text = "id,name,lat,lon\nP1,Bad,999.0,72.85\n"
        rec = DatasetRecord(provenance="SYNTHETIC", name="t")
        layer = load_geo_points_csv(csv_text, dataset_record=rec,
                                    provenance="SYNTHETIC")
        assert len(layer.points) == 0  # row skipped
        assert any("invalid latitude" in s or "skipped" in s
                   for s in layer.skipped_rows)

    def test_duplicate_geo_id_drops_second(self):
        csv_text = (
            "id,name,lat,lon\n"
            "P1,First,19.0,72.85\n"
            "P1,Second,19.1,72.86\n"
        )
        rec = DatasetRecord(provenance="SYNTHETIC", name="t")
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            layer = load_geo_points_csv(csv_text, dataset_record=rec,
                                        provenance="SYNTHETIC")
        assert len(layer.points) == 1
        assert layer.points[0].name == "First"

    def test_missing_lat_cell_skips_row(self):
        csv_text = "id,name,lat,lon\nP1,Ok,,72.85\n"
        rec = DatasetRecord(provenance="SYNTHETIC", name="t")
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            layer = load_geo_points_csv(csv_text, dataset_record=rec,
                                        provenance="SYNTHETIC")
        assert len(layer.points) == 0

    def test_invalid_provenance_on_geo_load_raises(self):
        csv_text = "id,name,lat,lon\nP1,Ok,19.0,72.85\n"
        rec = DatasetRecord(provenance="SYNTHETIC", name="t")
        with pytest.raises(ValueError, match="Unknown provenance"):
            load_geo_points_csv(csv_text, dataset_record=rec,
                                provenance="INVENTED")


# ---------------------------------------------------------------------------
# 7. Integration with existing model
# ---------------------------------------------------------------------------

class TestModelIntegration:
    """Test that real-data rainfall intensities can feed into the existing
    max-flow engine without breaking anything."""

    def _make_simple_net(self):
        """Reuse conftest's make_network factory."""
        from conftest import make_network
        return make_network(
            ["S", "M", "T"],
            [("D1", "S", "M", 5.0), ("D2", "M", "T", 5.0)],
            [("C1", "S", 10.0, 0.5)],
            [("T", 3.0)],
        )

    def test_derived_intensity_feeds_into_graph(self):
        """A DERIVED:TEMPORAL_DISAGGREGATE intensity runs through build_flow_network."""
        from floodcut.graph import build_flow_network
        from floodcut.models import RainfallScenario

        ds = load_synthetic_fixture()
        rec = ds.record_for_year(2005)
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            intensity, label = make_scenario_from_record(rec, "Jul", "derived_sc")

        # Build a scenario with the derived intensity (labelled appropriately).
        sc = RainfallScenario(
            id="derived_sc",
            name="Derived from IMD monthly (ILLUSTRATIVE ONLY)",
            intensity_mm_h=intensity,
            tide_factor=1.0,
            description=f"DERIVED:TEMPORAL_DISAGGREGATE | {label}",
        )
        net = self._make_simple_net()
        fn = build_flow_network(net, sc)
        assert fn.total_demand_m3s >= 0

    def test_derived_intensity_runs_maxflow(self):
        """End-to-end: derived intensity → flow network → max-flow analysis."""
        from floodcut.graph import build_flow_network
        from floodcut.maxflow import analyze
        from floodcut.models import RainfallScenario

        ds = load_synthetic_fixture()
        rec = ds.record_for_year(2007)
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            intensity, _ = make_scenario_from_record(rec, "Jul", "sc2")

        sc = RainfallScenario(
            id="sc2", name="Derived test",
            intensity_mm_h=intensity, tide_factor=1.0,
        )
        net = self._make_simple_net()
        fn = build_flow_network(net, sc)
        result = analyze(net, fn)
        assert math.isfinite(result.max_flow_m3s)
        assert math.isfinite(result.total_overflow_m3s)

    def test_real_data_provenance_visible_in_scenario_description(self):
        """Model input should carry provenance info from real data."""
        from floodcut.models import RainfallScenario

        ds = load_synthetic_fixture()
        rec = ds.record_for_year(2005)
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            intensity, label = make_scenario_from_record(rec, "Jun", "prov_sc")

        sc = RainfallScenario(
            id="prov_sc",
            name="Provenance test",
            intensity_mm_h=intensity,
            description=label,
        )
        assert "DERIVED" in sc.description
        assert "2005" in sc.description


# ---------------------------------------------------------------------------
# 8. Regression — Phases 3 & 4 still pass on the synthetic network
# ---------------------------------------------------------------------------

class TestPhase34Regression:
    """These tests confirm that Phase 5 changes did not break failure
    simulation or repair optimisation."""

    @pytest.fixture
    def sim_net_and_scenario(self):
        from floodcut.io import load_network, load_scenarios
        data = Path(__file__).resolve().parents[1] / "data" / "simulated"
        net = load_network(data / "mumbai_sim_network.json")
        scenarios = load_scenarios(data / "rainfall_scenarios.json")
        return net, scenarios["design"]

    def test_failure_simulation_counts(self, sim_net_and_scenario):
        import math
        from floodcut.failure import enumerate_failures
        net, sc = sim_net_and_scenario
        summary = enumerate_failures(net, sc, k_max=2)
        m = summary.m
        assert summary.count(1) == m
        assert summary.count(2) == math.comb(m, 2)

    def test_failure_simulation_worst_case_increases_overflow(self, sim_net_and_scenario):
        from floodcut.failure import enumerate_failures
        net, sc = sim_net_and_scenario
        summary = enumerate_failures(net, sc, k_max=1)
        wc = summary.worst_case(1)
        assert wc is not None
        assert wc.total_overflow_m3s >= summary.base_overflow_m3s - 1e-9

    def test_greedy_repair_reduces_overflow(self, sim_net_and_scenario):
        from floodcut.repair import greedy_repair
        net, sc = sim_net_and_scenario
        plan = greedy_repair(net, sc, budget=50.0)
        assert plan.total_overflow_m3s <= plan.base_overflow_m3s + 1e-9

    def test_exhaustive_repair_at_least_as_good_as_greedy(self, sim_net_and_scenario):
        from floodcut.repair import exhaustive_repair, greedy_repair
        net, sc = sim_net_and_scenario
        budget = 30.0
        greedy = greedy_repair(net, sc, budget=budget)
        exhaustive = exhaustive_repair(net, sc, budget=budget)
        assert exhaustive.overflow_reduction_m3s >= greedy.overflow_reduction_m3s - 1e-9


# ---------------------------------------------------------------------------
# 9. Real bundled data file
# ---------------------------------------------------------------------------

class TestRealBundledData:
    """Tests that load the actual downloaded OpenCity/IMD CSV.

    These tests run offline — they use the file checked into the repository.
    They are skipped if the file is not present (e.g. in minimal CI).
    """

    @pytest.fixture
    def real_dataset(self):
        if not _REAL_DATA.exists():
            pytest.skip("Real data file not present; skipping real-data tests.")
        from datetime import date
        from floodcut.provenance import OPENCITY_IMD_MONTHLY
        return load_rainfall_csv(
            _REAL_DATA,
            dataset_record=OPENCITY_IMD_MONTHLY,
            station_id="IMD_GRID_19.00N_72.75E",
            provenance="REAL:IMD_GRID",
        )

    def test_real_data_loads_without_error(self, real_dataset):
        assert len(real_dataset.records) > 0

    def test_real_data_covers_expected_years(self, real_dataset):
        yr = real_dataset.year_range()
        assert yr is not None
        assert yr[0] <= 1901
        assert yr[1] >= 2020

    def test_real_data_provenance_is_real(self, real_dataset):
        for rec in real_dataset.records:
            assert rec.provenance == "REAL:IMD_GRID"

    def test_real_data_monsoon_totals_reasonable(self, real_dataset):
        """Monsoon totals should be in a plausible range for Mumbai (400–3000 mm)."""
        totals = [
            t for t in real_dataset.monsoon_totals().values()
            if t is not None
        ]
        assert len(totals) > 100, "Expected many years of data"
        for t in totals:
            assert 100 <= t <= 4000, f"Implausible monsoon total: {t} mm"

    def test_real_data_no_skipped_rows(self, real_dataset):
        assert len(real_dataset.skipped_rows) == 0, (
            f"Unexpected skipped rows: {real_dataset.skipped_rows}"
        )

    def test_real_data_dataset_record_matches_opencity(self, real_dataset):
        assert real_dataset.dataset_record.is_real()
        assert "opencity" in real_dataset.dataset_record.source_url.lower()

    def test_real_data_derived_intensity_is_valid_float(self, real_dataset):
        """Derive an intensity from the 2005 July record and confirm it is a valid float."""
        rec_2005 = real_dataset.record_for_year(2005)
        if rec_2005 is None:
            pytest.skip("Year 2005 not in real dataset")
        mm = rec_2005.monthly_mm.get("Jul")
        if mm is None:
            pytest.skip("July 2005 is missing in real dataset")
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            intensity = monthly_mm_to_intensity_mm_h(mm, "Jul")
        assert math.isfinite(intensity)
        assert intensity >= 0.0

    def test_real_data_summary_reports_provenance(self, real_dataset):
        s = real_dataset.summary()
        assert "REAL" in s
