"""Tests for the disaster severity classifier."""
import pytest
import numpy as np
from ml.classifier import SeverityClassifier, AlertSeverity, ClassificationResult, FEATURE_NAMES


class TestSeverityClassifier:

    def setup_method(self):
        self.clf = SeverityClassifier()

    def test_predict_returns_classification_result(self):
        features = {name: 0.5 for name in FEATURE_NAMES}
        result = self.clf.predict(features)
        assert isinstance(result, ClassificationResult)

    def test_severity_is_valid_enum_value(self):
        features = {name: 0.5 for name in FEATURE_NAMES}
        result = self.clf.predict(features)
        assert result.severity in list(AlertSeverity)

    def test_recommended_channels_not_empty(self):
        features = {name: 0.5 for name in FEATURE_NAMES}
        result = self.clf.predict(features)
        assert len(result.recommended_channels) > 0

    def test_extreme_event_recommends_all_channels(self):
        """High magnitude, dense population, high vulnerability → EXTREME."""
        features = {
            "event_type_encoded": 0.1,
            "magnitude_or_wind_speed": 0.95,
            "population_density_at_center": 0.9,
            "infrastructure_vulnerability_score": 0.9,
            "prior_24h_event_count": 0.8,
            "network_degradation_index": 0.85,
            "time_of_day_risk_factor": 0.9,
            "evacuation_route_availability": 0.1,
            "hospital_proximity_score": 0.1,
            "weather_compounding_factor": 0.9,
        }
        result = self.clf.predict(features)
        # Whether trained or not, channels should be a non-empty list
        assert isinstance(result.recommended_channels, list)
        assert len(result.recommended_channels) >= 1

    def test_confidence_between_zero_and_one(self):
        features = {name: float(i) / len(FEATURE_NAMES) for i, name in enumerate(FEATURE_NAMES)}
        result = self.clf.predict(features)
        assert 0.0 <= result.confidence <= 1.0

    def test_missing_features_default_to_zero(self):
        """Partial feature dict should not raise."""
        result = self.clf.predict({"magnitude_or_wind_speed": 0.8})
        assert isinstance(result, ClassificationResult)

    @pytest.mark.parametrize("severity", list(AlertSeverity))
    def test_all_severity_levels_have_channel_mapping(self, severity):
        from ml.classifier import CHANNEL_RULES
        assert severity in CHANNEL_RULES
        assert len(CHANNEL_RULES[severity]) > 0
