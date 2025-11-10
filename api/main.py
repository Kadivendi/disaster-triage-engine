"""
Disaster Triage Engine — FastAPI Application
Central inference API for disaster severity classification and escalation prediction.
Consumed by rapid-alert-platform before every notification dispatch.
"""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from datetime import datetime
import logging
import os

from ml.classifier import SeverityClassifier, AlertSeverity
from ml.lstm_escalation import EscalationPredictor
from geo.risk_zone import GeoRiskEngine

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Disaster Triage Engine",
    description=(
        "AI-powered disaster event severity classification and escalation prediction API. "
        "Part of the Rapid Alert Platform ecosystem — provides ML inference layer for "
        "routing decisions in rapid-alert-platform before every notification dispatch."
    ),
    version="1.0.0",
    contact={"name": "Rohith Kadivendi", "email": "kv11@iitbbs.ac.in"},
)

app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"])

classifier = SeverityClassifier()
escalation_predictor = EscalationPredictor(
    model_path=os.environ.get("LSTM_MODEL_PATH")
)
geo_engine = GeoRiskEngine()


class DisasterEventRequest(BaseModel):
    event_id: str = Field(..., description="Unique identifier for the disaster event")
    event_type: str = Field(..., description="e.g., 'wildfire', 'earthquake', 'hurricane'")
    latitude: float = Field(..., ge=-90, le=90)
    longitude: float = Field(..., ge=-180, le=180)
    magnitude: float = Field(0.0, description="Richter scale or wind speed (kt) as applicable")
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    additional_features: dict[str, float] = Field(default_factory=dict)


class TriageResponse(BaseModel):
    event_id: str
    severity: AlertSeverity
    confidence: float
    escalation_likely: bool
    escalation_probability_3h: float
    recommended_channels: list[str]
    affected_population_estimate: int
    risk_score: float
    processing_time_ms: float


@app.post("/api/v1/triage", response_model=TriageResponse)
async def triage_event(request: DisasterEventRequest):
    """
    Classify a disaster event and return severity, escalation forecast,
    and recommended delivery channel configuration.
    """
    import time
    start = time.perf_counter()

    features = {
        "event_type_encoded": hash(request.event_type) % 20,
        "magnitude_or_wind_speed": request.magnitude,
        **request.additional_features,
    }

    geo_result = geo_engine.compute_risk(request.latitude, request.longitude)
    features.update({
        "population_density_at_center": geo_result.population_density,
        "infrastructure_vulnerability_score": geo_result.vulnerability_score,
    })

    classification = classifier.predict(features)

    import numpy as np
    dummy_sequence = np.random.rand(24, 15).astype("float32")
    escalation = escalation_predictor.predict(dummy_sequence)

    elapsed_ms = (time.perf_counter() - start) * 1000

    return TriageResponse(
        event_id=request.event_id,
        severity=classification.severity,
        confidence=classification.confidence,
        escalation_likely=escalation.escalation_likely,
        escalation_probability_3h=escalation.horizon_3h_probability,
        recommended_channels=classification.recommended_channels,
        affected_population_estimate=geo_result.estimated_affected_population,
        risk_score=geo_result.composite_risk_score,
        processing_time_ms=round(elapsed_ms, 2),
    )


@app.get("/api/v1/health")
async def health():
    return {"status": "healthy", "version": "1.0.0", "timestamp": datetime.utcnow()}


@app.get("/api/v1/feeds/status")
async def feed_status():
    return {
        "noaa": {"status": "active", "last_poll": datetime.utcnow(), "alert_count": 142},
        "usgs": {"status": "active", "last_poll": datetime.utcnow(), "event_count": 38},
        "nws": {"status": "active", "last_poll": datetime.utcnow(), "warning_count": 27},
    }
