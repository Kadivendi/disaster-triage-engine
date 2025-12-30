"""Robust data preprocessing pipeline with validation and normalization.

Handles the transformation of raw disaster event data into model-ready
feature vectors. Includes input validation, missing value imputation,
feature scaling, and data quality checks.
"""
import logging
import math
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List

logger = logging.getLogger(__name__)


class ScalingStrategy(Enum):
    """Feature scaling strategies."""
    MIN_MAX = "min_max"
    Z_SCORE = "z_score"
    ROBUST = "robust"
    NONE = "none"


@dataclass
class FeatureSpec:
    """Specification for a single input feature."""
    name: str
    min_value: float
    max_value: float
    required: bool = True
    default_value: float = 0.0
    scaling: ScalingStrategy = ScalingStrategy.MIN_MAX


# Feature specifications for the severity classifier
FEATURE_SPECS: List[FeatureSpec] = [
    FeatureSpec("magnitude_or_wind_speed", 0.0, 200.0, required=True),
    FeatureSpec("population_density_at_center", 0.0, 50000.0, required=True),
    FeatureSpec("infrastructure_vulnerability_score", 0.0, 1.0, required=True),
    FeatureSpec("network_degradation_index", 0.0, 1.0, required=False, default_value=0.0),
    FeatureSpec("prior_24h_event_count", 0, 100, required=False, default_value=0),
    FeatureSpec("distance_to_nearest_hospital_km", 0.0, 500.0, required=False, default_value=50.0),
    FeatureSpec("elevation_meters", -100.0, 9000.0, required=False, default_value=0.0),
    FeatureSpec("time_since_last_event_hours", 0.0, 8760.0, required=False, default_value=720.0),
    FeatureSpec("affected_area_sq_km", 0.0, 100000.0, required=True),
    FeatureSpec("historical_severity_avg", 0.0, 4.0, required=False, default_value=2.0),
]


class DataQualityError(Exception):
    """Raised when input data fails quality checks."""
    pass


class Preprocessor:
    """Feature preprocessing pipeline for disaster triage models.

    Pipeline stages:
    1. Validate input features against specifications
    2. Impute missing values with defaults
    3. Clip values to valid ranges
    4. Apply scaling transformation
    5. Return normalized feature vector
    """

    def __init__(self, scaling: ScalingStrategy = ScalingStrategy.MIN_MAX,
                 strict_mode: bool = False):
        self.scaling = scaling
        self.strict_mode = strict_mode
        self.specs = {s.name: s for s in FEATURE_SPECS}
        logger.info("Preprocessor initialized: scaling=%s, strict=%s", scaling.value, strict_mode)

    def transform(self, raw_features: Dict[str, Any]) -> Dict[str, float]:
        """Transform raw input features into model-ready format."""
        self._validate(raw_features)
        imputed = self._impute_missing(raw_features)
        clipped = self._clip_ranges(imputed)
        scaled = self._scale(clipped)
        return scaled

    def _validate(self, features: Dict[str, Any]) -> None:
        """Validate input features against specifications."""
        for spec in FEATURE_SPECS:
            if spec.required and spec.name not in features:
                if self.strict_mode:
                    raise DataQualityError(f"Missing required feature: {spec.name}")
                logger.warning("Missing required feature: %s (using default)", spec.name)
            value = features.get(spec.name)
            if value is not None and not isinstance(value, (int, float)):
                raise DataQualityError(
                    f"Feature {spec.name} must be numeric, got {type(value).__name__}"
                )
            if value is not None and math.isnan(value):
                raise DataQualityError(f"Feature {spec.name} contains NaN")

    def _impute_missing(self, features: Dict[str, Any]) -> Dict[str, float]:
        """Fill missing values with specification defaults."""
        result = {}
        for spec in FEATURE_SPECS:
            value = features.get(spec.name)
            if value is None:
                result[spec.name] = spec.default_value
            else:
                result[spec.name] = float(value)
        return result

    def _clip_ranges(self, features: Dict[str, float]) -> Dict[str, float]:
        """Clip feature values to valid ranges."""
        result = {}
        for name, value in features.items():
            spec = self.specs.get(name)
            if spec:
                clipped = max(spec.min_value, min(spec.max_value, value))
                if clipped != value:
                    logger.debug("Clipped %s: %.2f -> %.2f", name, value, clipped)
                result[name] = clipped
            else:
                result[name] = value
        return result

    def _scale(self, features: Dict[str, float]) -> Dict[str, float]:
        """Apply scaling transformation to features."""
        if self.scaling == ScalingStrategy.NONE:
            return features
        result = {}
        for name, value in features.items():
            spec = self.specs.get(name)
            if spec and spec.max_value != spec.min_value:
                if self.scaling == ScalingStrategy.MIN_MAX:
                    result[name] = (value - spec.min_value) / (spec.max_value - spec.min_value)
                else:
                    result[name] = value
            else:
                result[name] = value
        return result
