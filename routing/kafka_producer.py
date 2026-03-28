"""
Kafka Alert Event Producer

Publishes triaged disaster events to the `rapid-alert.triage-events` Kafka topic
consumed by `notification-service` in rapid-alert-platform.
"""
from __future__ import annotations

import json
import logging
import os

from kafka import KafkaProducer
from kafka.errors import KafkaError, NoBrokersAvailable

logger = logging.getLogger(__name__)

_DEFAULT_BOOTSTRAP = os.environ.get("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")
_DEFAULT_TOPIC = os.environ.get("ALERT_KAFKA_TOPIC", "rapid-alert.triage-events")


class TriageEventProducer:
    """
    Lightweight, eager-failure Kafka producer.

    Construction raises if no broker is reachable so that the caller can choose
    to degrade gracefully (the FastAPI app does this — it logs a warning and
    keeps serving requests, just without publishing).
    """

    def __init__(
        self,
        bootstrap_servers: str | list[str] = _DEFAULT_BOOTSTRAP,
        topic: str = _DEFAULT_TOPIC,
        request_timeout_ms: int = 5000,
    ) -> None:
        if isinstance(bootstrap_servers, str):
            servers = [s.strip() for s in bootstrap_servers.split(",") if s.strip()]
        else:
            servers = list(bootstrap_servers)

        try:
            self._producer = KafkaProducer(
                bootstrap_servers=servers,
                value_serializer=lambda v: json.dumps(v, default=str).encode("utf-8"),
                key_serializer=lambda k: k.encode("utf-8"),
                acks="all",
                retries=5,
                compression_type="gzip",
                request_timeout_ms=request_timeout_ms,
            )
        except NoBrokersAvailable as exc:
            raise RuntimeError(f"No Kafka brokers reachable at {servers}: {exc}") from exc

        self._topic = topic
        logger.info("Kafka producer ready: brokers=%s topic=%s", servers, topic)

    @property
    def topic(self) -> str:
        return self._topic

    def publish_triage_result(self, event_id: str, triage_result: dict) -> bool:
        """Publish a single triage result. Returns True if Kafka acknowledged."""
        try:
            future = self._producer.send(
                self._topic,
                key=event_id,
                value={"event_id": event_id, **triage_result},
            )
            future.get(timeout=10)
            logger.debug("Published triage result for event=%s", event_id)
            return True
        except KafkaError as exc:
            logger.error("Kafka publish failed for event=%s: %s", event_id, exc)
            return False

    def close(self) -> None:
        try:
            self._producer.flush(timeout=5)
        finally:
            self._producer.close(timeout=5)
