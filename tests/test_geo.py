"""Tests for the geographic risk computation engine."""
import pytest
from geo.risk_zone import GeoRiskEngine, GeoRiskResult


class TestGeoRiskEngine:

    def setup_method(self):
        self.engine = GeoRiskEngine()

    def test_returns_geo_risk_result(self):
        result = self.engine.compute_risk(34.05, -118.24)
        assert isinstance(result, GeoRiskResult)

    def test_composite_risk_score_between_zero_and_one(self):
        result = self.engine.compute_risk(34.05, -118.24)
        assert 0.0 <= result.composite_risk_score <= 1.0

    def test_vulnerability_score_between_zero_and_one(self):
        result = self.engine.compute_risk(34.05, -118.24)
        assert 0.0 <= result.vulnerability_score <= 1.0

    def test_affected_population_is_positive(self):
        result = self.engine.compute_risk(34.05, -118.24)
        assert result.estimated_affected_population > 0

    def test_coordinates_preserved_in_result(self):
        result = self.engine.compute_risk(37.77, -122.41)
        assert result.latitude == pytest.approx(37.77)
        assert result.longitude == pytest.approx(-122.41)

    @pytest.mark.parametrize("lat,lon", [
        (40.71, -74.00),   # New York
        (29.76, -95.36),   # Houston
        (21.30, -157.85),  # Honolulu
        (61.21, -149.86),  # Anchorage
        (25.77, -80.19),   # Miami (coastal)
    ])
    def test_various_us_cities(self, lat, lon):
        result = self.engine.compute_risk(lat, lon)
        assert result.composite_risk_score >= 0.0
        assert result.region_type in ["coastal", "urban_high_density", "suburban", "rural", "remote"]

    def test_hospital_count_is_positive(self):
        result = self.engine.compute_risk(34.05, -118.24)
        assert result.hospital_count_25km >= 1
