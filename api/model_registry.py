"""Model versioning and A/B testing support for prediction endpoints.

Supports loading multiple model versions simultaneously and splitting
traffic between them for canary deployments and A/B experiments.
"""
import logging
import random
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass
class ModelVersion:
    """A registered model version with metadata and traffic allocation."""
    version: str
    model_type: str  # "classifier" or "escalation"
    predict_fn: Callable
    traffic_weight: float = 0.0  # 0-100 percentage
    registered_at: float = field(default_factory=time.time)
    prediction_count: int = 0
    is_active: bool = True
    description: str = ""


class ModelRegistry:
    """Registry for managing model versions and traffic splitting.

    Supports:
    - Registering multiple model versions with metadata
    - Weighted traffic splitting for A/B testing
    - Promoting/demoting model versions
    - Prediction count tracking per version
    """

    def __init__(self):
        self._models: Dict[str, ModelVersion] = {}
        self._default_version: Optional[str] = None
        logger.info("ModelRegistry initialized")

    def register(self, version: str, model_type: str,
                 predict_fn: Callable, traffic_weight: float = 0.0,
                 description: str = "") -> None:
        """Register a new model version."""
        model = ModelVersion(
            version=version,
            model_type=model_type,
            predict_fn=predict_fn,
            traffic_weight=traffic_weight,
            description=description,
        )
        self._models[version] = model
        if self._default_version is None:
            self._default_version = version
            model.traffic_weight = 100.0
        logger.info("Model registered: v%s type=%s weight=%.0f%%",
                    version, model_type, traffic_weight)

    def predict(self, features: Dict[str, float],
                force_version: Optional[str] = None) -> Tuple[Dict[str, Any], str]:
        """Run prediction using traffic-weighted version selection."""
        if force_version:
            version = force_version
        else:
            version = self._select_version()

        model = self._models.get(version)
        if model is None:
            raise ValueError(f"Model version {version} not found")

        result = model.predict_fn(features)
        model.prediction_count += 1
        return result, version

    def promote(self, version: str, traffic_weight: float = 100.0) -> None:
        """Promote a model version to receive more traffic."""
        if version not in self._models:
            raise ValueError(f"Version {version} not found")
        self._models[version].traffic_weight = traffic_weight
        # Rebalance other weights
        remaining = 100.0 - traffic_weight
        others = [v for v in self._models.values() if v.version != version and v.is_active]
        if others:
            per_other = remaining / len(others)
            for other in others:
                other.traffic_weight = per_other
        self._default_version = version
        logger.info("Model promoted: v%s at %.0f%% traffic", version, traffic_weight)

    def list_versions(self) -> List[Dict[str, Any]]:
        """List all registered model versions with their metadata."""
        return [
            {
                "version": m.version,
                "model_type": m.model_type,
                "traffic_weight": m.traffic_weight,
                "prediction_count": m.prediction_count,
                "is_active": m.is_active,
                "registered_at": m.registered_at,
                "description": m.description,
            }
            for m in self._models.values()
        ]

    def _select_version(self) -> str:
        """Select a model version based on traffic weights."""
        active = [(v.version, v.traffic_weight) for v in self._models.values() if v.is_active]
        if not active:
            raise RuntimeError("No active model versions available")
        if len(active) == 1:
            return active[0][0]
        roll = random.uniform(0, 100)
        cumulative = 0.0
        for version, weight in active:
            cumulative += weight
            if roll <= cumulative:
                return version
        return active[-1][0]
