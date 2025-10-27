"""
Disaster Event Severity Classifier
Multi-class classifier that predicts alert severity (EXTREME/SEVERE/MODERATE/MINOR)
from incoming disaster event features. Trained on 5 years of NOAA/USGS historical data.
"""
import numpy as np
import logging
from dataclasses import dataclass
from enum import Enum
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import joblib
import os

logger = logging.getLogger(__name__)

MODEL_PATH = os.environ.get("MODEL_PATH", "models/saved/severity_classifier.pkl")


class AlertSeverity(str, Enum):
    EXTREME = "EXTREME"
    SEVERE = "SEVERE"
    MODERATE = "MODERATE"
    MINOR = "MINOR"


@dataclass
class ClassificationResult:
    severity: AlertSeverity
    confidence: float
    feature_importance: dict[str, float]
    recommended_channels: list[str]


FEATURE_NAMES = [
    "event_type_encoded",
    "magnitude_or_wind_speed",
    "population_density_at_center",
    "infrastructure_vulnerability_score",
    "prior_24h_event_count",
    "network_degradation_index",
    "time_of_day_risk_factor",
    "evacuation_route_availability",
    "hospital_proximity_score",
    "weather_compounding_factor",
]

CHANNEL_RULES = {
    AlertSeverity.EXTREME: ["rapid_alert_platform", "mesh_gateway", "sms", "push", "telegram"],
    AlertSeverity.SEVERE: ["rapid_alert_platform", "push", "telegram", "sms"],
    AlertSeverity.MODERATE: ["rapid_alert_platform", "push", "telegram"],
    AlertSeverity.MINOR: ["rapid_alert_platform", "push"],
}


class SeverityClassifier:
    """
    Gradient Boosting classifier for emergency alert severity prediction.
    Wrapped in sklearn Pipeline with StandardScaler for production inference.
    """

    def __init__(self):
        self._pipeline: Pipeline | None = None
        self._load_or_initialize()

    def _load_or_initialize(self):
        if os.path.exists(MODEL_PATH):
            logger.info(f"Loading pre-trained model from {MODEL_PATH}")
            self._pipeline = joblib.load(MODEL_PATH)
        else:
            logger.warning("No saved model found — initializing untrained classifier")
            self._pipeline = Pipeline([
                ("scaler", StandardScaler()),
                ("clf", GradientBoostingClassifier(
                    n_estimators=200,
                    learning_rate=0.05,
                    max_depth=5,
                    subsample=0.8,
                    random_state=42,
                )),
            ])

    def predict(self, features: dict[str, float]) -> ClassificationResult:
        """Classify a disaster event and recommend delivery channels."""
        feature_vector = np.array([
            features.get(name, 0.0) for name in FEATURE_NAMES
        ]).reshape(1, -1)

        if not hasattr(self._pipeline.named_steps["clf"], "classes_"):
            logger.warning("Model not trained — returning default MODERATE classification")
            return ClassificationResult(
                severity=AlertSeverity.MODERATE,
                confidence=0.0,
                feature_importance={},
                recommended_channels=CHANNEL_RULES[AlertSeverity.MODERATE],
            )

        proba = self._pipeline.predict_proba(feature_vector)[0]
        classes = self._pipeline.named_steps["clf"].classes_
        predicted_idx = proba.argmax()
        severity = AlertSeverity(classes[predicted_idx])

        importances = self._pipeline.named_steps["clf"].feature_importances_
        feature_importance = dict(zip(FEATURE_NAMES, importances.tolist()))

        return ClassificationResult(
            severity=severity,
            confidence=float(proba[predicted_idx]),
            feature_importance=feature_importance,
            recommended_channels=CHANNEL_RULES[severity],
        )

    def save(self):
        os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
        joblib.dump(self._pipeline, MODEL_PATH)
        logger.info(f"Model saved to {MODEL_PATH}")
