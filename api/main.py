"""
Disaster Triage Engine — FastAPI Application
Central inference API for disaster severity classification and escalation prediction.
Consumed by rapid-alert-platform before every notification dispatch.
"""
from __future__ import annotations

import asyncio
import hashlib
import logging
import os
import time
from contextlib import asynccontextmanager
from datetime import datetime, timezone

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from cache.prediction_cache import PredictionCache
from config import AppConfig
from geo.risk_zone import GeoRiskEngine
from ingest.noaa_client import NOAAClient
from ingest.nws_client import NWSClient
from ingest.usgs_client import USGSClient
from ml.classifier import AlertSeverity, SeverityClassifier
from ml.feature_engineer import FeatureEngineer
from ml.lstm_escalation import EscalationPredictor
from ml.preprocessor import Preprocessor, ScalingStrategy
from routing.kafka_producer import TriageEventProducer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Module-level singletons.
#
# The previous version instantiated every dependency at import time, which is
# slow and crashes the whole app if a single component fails. The lifespan hook
# below loads everything inside FastAPI's startup phase and tears it down cleanly.
# ---------------------------------------------------------------------------

config: AppConfig | None = None
classifier: SeverityClassifier | None = None
escalation_predictor: EscalationPredictor | None = None
geo_engine: GeoRiskEngine | None = None
preprocessor: Preprocessor | None = None
feature_engineer: FeatureEngineer | None = None
prediction_cache: PredictionCache | None = None
triage_producer: TriageEventProducer | None = None
ingest_tasks: list[asyncio.Task] = []


async def _poll_loop(name: str, interval_seconds: int, fn):
    """Background poller that survives transient ingest failures.

    The ingest clients (``fetch_*``) return generators; we exhaust them in a worker
    thread so the actual HTTP/feed work happens off the event loop, then record
    the count into ``cache/feeds.db`` so ``/api/v1/feeds/status`` can report on it.
    """
    logger.info("Ingest scheduler started: %s every %ds", name, interval_seconds)
    while True:
        try:
            count = await asyncio.to_thread(_consume_and_count, fn)
            await asyncio.to_thread(_record_poll, name, count)
            logger.debug("%s poll completed: %d items", name, count)
        except Exception as exc:  # noqa: BLE001 — feeds may flap
            logger.warning("Ingest %s failed: %s", name, exc)
        await asyncio.sleep(interval_seconds)


def _consume_and_count(fn) -> int:
    """Run a fetch function, count items if it's a generator."""
    result = fn()
    if hasattr(result, "__iter__") and not isinstance(result, (list, tuple, dict)):
        return sum(1 for _ in result)
    if isinstance(result, (list, tuple)):
        return len(result)
    return 1 if result is not None else 0


def _record_poll(feed_name: str, count: int) -> None:
    """Persist a poll outcome to ``cache/feeds.db`` for /feeds/status."""
    import sqlite3
    os.makedirs("cache", exist_ok=True)
    conn = sqlite3.connect("cache/feeds.db")
    try:
        conn.execute(
            "CREATE TABLE IF NOT EXISTS feed_status ("
            "feed_name TEXT PRIMARY KEY, alert_count INTEGER, last_poll TEXT)"
        )
        conn.execute(
            "INSERT INTO feed_status(feed_name, alert_count, last_poll) "
            "VALUES (?, ?, ?) "
            "ON CONFLICT(feed_name) DO UPDATE SET "
            "alert_count=excluded.alert_count, last_poll=excluded.last_poll",
            (feed_name, count, datetime.now(timezone.utc).isoformat()),
        )
        conn.commit()
    finally:
        conn.close()


@asynccontextmanager
async def lifespan(app: FastAPI):
    global config, classifier, escalation_predictor, geo_engine
    global preprocessor, feature_engineer, prediction_cache, triage_producer

    config = AppConfig.from_env()
    classifier = SeverityClassifier()
    escalation_predictor = EscalationPredictor(model_path=os.environ.get("LSTM_MODEL_PATH"))
    geo_engine = GeoRiskEngine()
    preprocessor = Preprocessor(scaling=ScalingStrategy.MIN_MAX)
    feature_engineer = FeatureEngineer()
    prediction_cache = PredictionCache(max_size=10_000, default_ttl=300.0)
    triage_producer = _build_producer(config)

    # NOAA / USGS / NWS pollers — fire-and-forget background loops.
    try:
        noaa = NOAAClient()
        ingest_tasks.append(asyncio.create_task(
            _poll_loop("noaa", config.noaa_poll_interval_seconds, noaa.fetch_active_alerts)
        ))
    except Exception as exc:  # noqa: BLE001
        logger.warning("NOAA client init failed (ingest disabled): %s", exc)

    try:
        usgs = USGSClient(min_magnitude=config.usgs_min_magnitude)
        ingest_tasks.append(asyncio.create_task(
            _poll_loop("usgs", 60, usgs.fetch_significant_events)
        ))
    except Exception as exc:  # noqa: BLE001
        logger.warning("USGS client init failed (ingest disabled): %s", exc)

    try:
        nws = NWSClient()
        ingest_tasks.append(asyncio.create_task(
            _poll_loop("nws", 60, nws.fetch_active_warnings)
        ))
    except Exception as exc:  # noqa: BLE001
        logger.warning("NWS client init failed (ingest disabled): %s", exc)

    yield

    for task in ingest_tasks:
        task.cancel()
    if triage_producer is not None:
        try:
            triage_producer.close()
        except Exception:  # noqa: BLE001
            pass


def _build_producer(cfg: AppConfig) -> TriageEventProducer | None:
    """Producer construction is isolated so a Kafka outage doesn't take the API down."""
    try:
        return TriageEventProducer(
            bootstrap_servers=cfg.kafka_bootstrap_servers,
            topic=cfg.kafka_topic_triage_events,
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "TriageEventProducer init failed (%s). API stays up; /triage will skip Kafka publish.",
            exc,
        )
        return None


app = FastAPI(
    title="Disaster Triage Engine",
    description=(
        "AI-powered disaster event severity classification and escalation prediction API. "
        "Part of the Rapid Alert Platform ecosystem — provides ML inference layer for "
        "routing decisions in rapid-alert-platform before every notification dispatch."
    ),
    version="1.0.0",
    contact={"name": "Rohith Kadivendi", "email": "kv11@iitbbs.ac.in"},
    lifespan=lifespan,
)

allowed_origins = os.environ.get("ALLOWED_ORIGINS", "http://localhost:3000").split(",")
app.add_middleware(CORSMiddleware, allow_origins=allowed_origins, allow_methods=["*"])


class DisasterEventRequest(BaseModel):
    event_id: str = Field(..., description="Unique identifier for the disaster event")
    event_type: str = Field(..., description="e.g., 'wildfire', 'earthquake', 'hurricane'")
    latitude: float = Field(..., ge=-90, le=90)
    longitude: float = Field(..., ge=-180, le=180)
    magnitude: float = Field(0.0, description="Richter scale or wind speed (kt) as applicable")
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    additional_features: dict[str, float] = Field(default_factory=dict)
    sensor_sequence: list[list[float]] | None = Field(
        None,
        description="24x15 historical sensor reading matrix for LSTM escalation prediction",
    )


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


def _encode_event_type(event_type: str) -> int:
    """Stable bucket id for the event-type categorical feature."""
    return int(hashlib.sha256(event_type.encode()).hexdigest(), 16) % 20


@app.post("/api/v1/triage", response_model=TriageResponse)
async def triage_event(request: DisasterEventRequest):
    """
    Classify a disaster event and return severity, escalation forecast,
    and recommended delivery channel configuration.

    Pipeline:
        1. Geo enrichment via GeoRiskEngine
        2. Feature engineering + min-max preprocessing
        3. Cache lookup keyed by normalised feature vector
        4. SeverityClassifier (GBM) inference (cache hit short-circuits ML)
        5. EscalationPredictor (BiLSTM + attention) inference
        6. Publish the result to the rapid-alert.triage-events Kafka topic
           so notification-service in rapid-alert-platform can dispatch.
    """
    start = time.perf_counter()

    geo_result = geo_engine.compute_risk(request.latitude, request.longitude)

    # Feature engineering produces the model-ready 10-dim vector.
    fv = feature_engineer.extract(
        event_type=request.event_type,
        magnitude=request.magnitude,
        latitude=request.latitude,
        longitude=request.longitude,
        timestamp=request.timestamp,
        prior_event_count=int(request.additional_features.get("prior_24h_event_count", 0)),
        population_density=geo_result.population_density,
        vulnerability_score=geo_result.vulnerability_score,
    )
    classifier_features = {
        "event_type_encoded": fv.event_type_encoded,
        "magnitude_or_wind_speed": fv.magnitude_or_wind_speed,
        "population_density_at_center": fv.population_density_at_center,
        "infrastructure_vulnerability_score": fv.infrastructure_vulnerability_score,
        "prior_24h_event_count": fv.prior_24h_event_count,
        "network_degradation_index": fv.network_degradation_index,
        "time_of_day_risk_factor": fv.time_of_day_risk_factor,
        "evacuation_route_availability": fv.evacuation_route_availability,
        "hospital_proximity_score": fv.hospital_proximity_score,
        "weather_compounding_factor": fv.weather_compounding_factor,
    }

    # Validate + clip + scale the broader raw input to surface data quality issues
    # early; the preprocessor doubles as our cache key generator.
    raw_for_cache = {
        "magnitude_or_wind_speed": float(request.magnitude),
        "population_density_at_center": float(geo_result.population_density),
        "infrastructure_vulnerability_score": float(geo_result.vulnerability_score),
        "affected_area_sq_km": float(
            request.additional_features.get("affected_area_sq_km", 1963.5)
        ),
        **{k: v for k, v in request.additional_features.items() if isinstance(v, (int, float))},
    }
    try:
        cache_key_features = preprocessor.transform(raw_for_cache)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=f"Preprocessing failed: {exc}") from exc

    # Cache short-circuit: identical feature vectors reuse the previous result.
    cached = prediction_cache.get(cache_key_features) if prediction_cache else None
    if cached is not None:
        elapsed_ms = (time.perf_counter() - start) * 1000
        cached = {**cached, "processing_time_ms": round(elapsed_ms, 2)}
        _publish_triage(request.event_id, cached)
        return TriageResponse(**cached)

    classification = classifier.predict(classifier_features)

    if request.sensor_sequence and len(request.sensor_sequence) == 24 \
            and all(len(row) == 15 for row in request.sensor_sequence):
        sensor_sequence = np.array(request.sensor_sequence, dtype="float32")
    else:
        sensor_sequence = np.zeros((24, 15), dtype="float32")
    escalation = escalation_predictor.predict(sensor_sequence)

    elapsed_ms = (time.perf_counter() - start) * 1000
    response = TriageResponse(
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

    if prediction_cache is not None:
        prediction_cache.put(cache_key_features, response.model_dump())

    _publish_triage(request.event_id, response.model_dump())
    return response


def _publish_triage(event_id: str, payload: dict) -> None:
    if triage_producer is None:
        logger.debug("Skipping Kafka publish (producer disabled) for event=%s", event_id)
        return
    try:
        triage_producer.publish_triage_result(event_id, payload)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Kafka publish failed for event=%s: %s", event_id, exc)


@app.get("/api/v1/health")
async def health():
    return {
        "status": "healthy",
        "version": "1.0.0",
        "timestamp": datetime.now(timezone.utc),
        "kafka_publisher": triage_producer is not None,
        "active_ingest_loops": sum(1 for t in ingest_tasks if not t.done()),
    }


@app.get("/api/v1/feeds/status")
async def feed_status():
    import sqlite3

    now = datetime.now(timezone.utc)
    counts = {"noaa": 0, "usgs": 0, "nws": 0}
    last_polls = {"noaa": None, "usgs": None, "nws": None}
    available = False
    try:
        if os.path.exists("cache/feeds.db"):
            available = True
            conn = sqlite3.connect("cache/feeds.db")
            cursor = conn.cursor()
            cursor.execute("SELECT feed_name, alert_count, last_poll FROM feed_status")
            for row in cursor.fetchall():
                counts[row[0]] = row[1]
                last_polls[row[0]] = row[2]
            conn.close()
    except sqlite3.OperationalError as exc:
        logger.debug("feeds.db schema mismatch (using defaults): %s", exc)
    except Exception as exc:  # noqa: BLE001
        logger.error("Failed to read feed cache: %s", exc)

    def status_for(feed: str) -> str:
        if not available:
            return "initializing"
        if any(not t.done() for t in ingest_tasks):
            return "active"
        return "idle"

    return {
        "noaa": {"status": status_for("noaa"), "last_poll": last_polls["noaa"] or now,
                 "alert_count": counts["noaa"]},
        "usgs": {"status": status_for("usgs"), "last_poll": last_polls["usgs"] or now,
                 "event_count": counts["usgs"]},
        "nws": {"status": status_for("nws"), "last_poll": last_polls["nws"] or now,
                "warning_count": counts["nws"]},
    }
