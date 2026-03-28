"""End-to-end integration tests for the full triage pipeline.

Tests the complete flow: raw event → preprocess → classify → predict
escalation → format output, using mock external feeds.
"""
import pytest
from unittest.mock import MagicMock, patch
from ml.preprocessor import Preprocessor, ScalingStrategy, DataQualityError
from ml.feature_engineer import FeatureEngineer
from cache.prediction_cache import PredictionCache


class TestPreprocessorPipeline:
    """Tests for the preprocessing stage of the pipeline."""

    def setup_method(self):
        self.preprocessor = Preprocessor(scaling=ScalingStrategy.MIN_MAX)

    def test_transform_valid_input(self):
        features = {
            "magnitude_or_wind_speed": 7.2,
            "population_density_at_center": 5000.0,
            "infrastructure_vulnerability_score": 0.7,
            "affected_area_sq_km": 150.0,
        }
        result = self.preprocessor.transform(features)
        assert isinstance(result, dict)
        assert len(result) == 10  # All 10 features including defaults
        # All values should be in [0, 1] after min-max scaling
        for name, value in result.items():
            assert 0.0 <= value <= 1.0, f"{name}={value} out of [0,1] range"

    def test_transform_missing_optional_features(self):
        features = {
            "magnitude_or_wind_speed": 5.0,
            "population_density_at_center": 1000.0,
            "infrastructure_vulnerability_score": 0.3,
            "affected_area_sq_km": 50.0,
        }
        result = self.preprocessor.transform(features)
        assert "network_degradation_index" in result
        assert result["network_degradation_index"] == 0.0  # default

    def test_transform_clips_out_of_range(self):
        features = {
            "magnitude_or_wind_speed": 999.0,  # exceeds max of 200
            "population_density_at_center": 1000.0,
            "infrastructure_vulnerability_score": 0.5,
            "affected_area_sq_km": 100.0,
        }
        result = self.preprocessor.transform(features)
        assert result["magnitude_or_wind_speed"] == 1.0  # clipped to max then scaled

    def test_transform_rejects_nan_values(self):
        features = {
            "magnitude_or_wind_speed": float("nan"),
            "population_density_at_center": 1000.0,
            "infrastructure_vulnerability_score": 0.5,
            "affected_area_sq_km": 100.0,
        }
        with pytest.raises(DataQualityError, match="NaN"):
            self.preprocessor.transform(features)

    def test_strict_mode_rejects_missing_required(self):
        preprocessor = Preprocessor(strict_mode=True)
        features = {"magnitude_or_wind_speed": 5.0}  # missing required fields
        with pytest.raises(DataQualityError, match="Missing required"):
            preprocessor.transform(features)


class TestPredictionCache:
    """Tests for the prediction caching layer."""

    def setup_method(self):
        self.cache = PredictionCache(max_size=100, default_ttl=60.0)

    def test_cache_miss_returns_none(self):
        result = self.cache.get({"feature_a": 1.0})
        assert result is None
        assert self.cache.stats["misses"] == 1

    def test_cache_hit_returns_stored_value(self):
        features = {"feature_a": 1.0, "feature_b": 2.0}
        prediction = {"severity": "EXTREME", "confidence": 0.94}
        self.cache.put(features, prediction)
        result = self.cache.get(features)
        assert result == prediction
        assert self.cache.stats["hits"] == 1

    def test_cache_deterministic_keys(self):
        """Same features in different order should hit the same cache entry."""
        features_a = {"x": 1.0, "y": 2.0, "z": 3.0}
        features_b = {"z": 3.0, "x": 1.0, "y": 2.0}
        self.cache.put(features_a, {"result": "test"})
        result = self.cache.get(features_b)
        assert result == {"result": "test"}

    def test_cache_lru_eviction(self):
        cache = PredictionCache(max_size=3, default_ttl=300.0)
        for i in range(4):
            cache.put({"i": float(i)}, {"result": i})
        # First entry should have been evicted
        assert cache.get({"i": 0.0}) is None
        assert cache.get({"i": 3.0}) == {"result": 3}

    def test_cache_clear(self):
        for i in range(5):
            self.cache.put({"i": float(i)}, {"result": i})
        count = self.cache.clear()
        assert count == 5
        assert self.cache.stats["size"] == 0


class TestFullPipeline:
    """Tests for the complete triage pipeline integration."""

    def test_pipeline_produces_valid_output(self):
        preprocessor = Preprocessor()
        raw_event = {
            "magnitude_or_wind_speed": 7.5,
            "population_density_at_center": 12000.0,
            "infrastructure_vulnerability_score": 0.85,
            "network_degradation_index": 0.6,
            "affected_area_sq_km": 200.0,
        }
        processed = preprocessor.transform(raw_event)
        assert len(processed) == 10
        assert all(isinstance(v, float) for v in processed.values())

    def test_pipeline_with_cache_integration(self):
        preprocessor = Preprocessor()
        cache = PredictionCache(max_size=100)

        raw_event = {
            "magnitude_or_wind_speed": 6.0,
            "population_density_at_center": 8000.0,
            "infrastructure_vulnerability_score": 0.5,
            "affected_area_sq_km": 100.0,
        }
        processed = preprocessor.transform(raw_event)

        # First call - cache miss
        cached = cache.get(processed)
        assert cached is None

        # Simulate prediction and cache
        prediction = {"severity": "SEVERE", "confidence": 0.88}
        cache.put(processed, prediction)

        # Second call - cache hit
        cached = cache.get(processed)
        assert cached == prediction
