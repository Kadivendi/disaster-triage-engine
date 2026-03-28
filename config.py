"""Application configuration loaded from environment variables at startup."""
import os, sys, logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class AppConfig:
    kafka_bootstrap_servers: str = "localhost:9092"
    kafka_topic_triage_events: str = "rapid-alert.triage-events"
    kafka_group_id: str = "disaster-triage-engine"
    noaa_cap_feed_url: str = "https://api.weather.gov/alerts/active.atom"
    noaa_poll_interval_seconds: int = 60
    usgs_feed_url: str = "https://earthquake.usgs.gov/earthquakes/feed/v1.0/summary/significant_hour.geojson"
    usgs_min_magnitude: float = 2.5
    model_dir: str = "models/saved"
    escalation_threshold_3h: float = 0.65
    batch_window_ms: float = 50.0
    max_batch_size: int = 64
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    log_level: str = "INFO"

    @classmethod
    def from_env(cls) -> "AppConfig":
        cfg = cls(
            kafka_bootstrap_servers=os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092"),
            kafka_topic_triage_events=os.getenv("KAFKA_TOPIC_TRIAGE_EVENTS", "rapid-alert.triage-events"),
            noaa_poll_interval_seconds=int(os.getenv("NOAA_POLL_INTERVAL_SECONDS", "60")),
            usgs_min_magnitude=float(os.getenv("USGS_MIN_MAGNITUDE", "2.5")),
            model_dir=os.getenv("MODEL_DIR", "models/saved"),
            escalation_threshold_3h=float(os.getenv("ESCALATION_THRESHOLD_3H", "0.65")),
            batch_window_ms=float(os.getenv("BATCH_WINDOW_MS", "50.0")),
            api_port=int(os.getenv("API_PORT", "8000")),
            log_level=os.getenv("LOG_LEVEL", "INFO"),
        )
        cfg._validate()
        return cfg

    def _validate(self):
        errors = []
        if not self.kafka_bootstrap_servers:
            errors.append("KAFKA_BOOTSTRAP_SERVERS is required")
        if not (0.0 < self.escalation_threshold_3h < 1.0):
            errors.append(f"ESCALATION_THRESHOLD_3H must be 0-1, got {self.escalation_threshold_3h}")
        if self.noaa_poll_interval_seconds < 10:
            errors.append("NOAA_POLL_INTERVAL_SECONDS must be >= 10 to avoid rate limits")
        if errors:
            for e in errors: logger.error(f"Config: {e}")
            sys.exit(1)
