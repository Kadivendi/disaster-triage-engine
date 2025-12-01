"""Tests for the feature engineering pipeline."""
import pytest
from datetime import datetime
from ml.feature_engineer import FeatureEngineer, FeatureVector


class TestFeatureEngineer:

    def setup_method(self):
        self.fe = FeatureEngineer()

    def test_returns_feature_vector(self):
        result = self.fe.extract("wildfire", 7.5, 34.05, -118.24)
        assert isinstance(result, FeatureVector)

    def test_feature_vector_length(self):
        result = self.fe.extract("earthquake", 6.2, 37.77, -122.41)
        assert len(result.to_list()) == 10

    def test_all_features_between_zero_and_one(self):
        result = self.fe.extract("hurricane", 150.0, 25.77, -80.19)
        for val in result.to_list():
            assert 0.0 <= val <= 1.0, f"Feature out of range: {val}"

    def test_night_time_risk_factor_high(self):
        night = datetime(2025, 10, 15, 2, 30, 0)  # 2:30 AM
        result = self.fe.extract("wildfire", 5.0, 34.05, -118.24, timestamp=night)
        assert result.time_of_day_risk_factor >= 0.8

    def test_daytime_risk_factor_lower(self):
        day = datetime(2025, 10, 15, 14, 0, 0)  # 2:00 PM
        result = self.fe.extract("wildfire", 5.0, 34.05, -118.24, timestamp=day)
        assert result.time_of_day_risk_factor <= 0.5

    def test_earthquake_magnitude_normalized(self):
        result = self.fe.extract("earthquake", 10.0, 34.05, -118.24)
        assert result.magnitude_or_wind_speed == pytest.approx(1.0)

    def test_unknown_event_type_handled(self):
        """Unknown event type should not raise."""
        result = self.fe.extract("unusual_unknown_event", 3.0, 34.05, -118.24)
        assert isinstance(result, FeatureVector)
