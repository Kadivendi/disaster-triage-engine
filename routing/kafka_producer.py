"""
Kafka Alert Event Producer
Publishes triaged disaster events to the rapid-alert-platform Kafka topic
for downstream notification dispatch.
"""
import json
import logging
from dataclasses import asdict
from kafka import KafkaProducer
from kafka.errors import KafkaError
import os

logger = logging.getLogger(__name__)

KAFKA_BOOTSTRAP = os.environ.get("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")
ALERT_TOPIC = os.environ.get("ALERT_KAFKA_TOPIC", "rapid-alert.triage-events")


class TriageEventProducer:
    def __init__(self):
        self._producer = KafkaProducer(
            bootstrap_servers=KAFKA_BOOTSTRAP.split(","),
            value_serializer=lambda v: json.dumps(v, default=str).encode("utf-8"),
            key_serializer=lambda k: k.encode("utf-8"),
            acks="all",
            retries=5,
            compression_type="gzip",
        )

    def publish_triage_result(self, event_id: str, triage_result: dict) -> bool:
        try:
            future = self._producer.send(
                ALERT_TOPIC,
                key=event_id,
                value={"event_id": event_id, "triage": triage_result},
            )
            future.get(timeout=10)
            logger.info(f"Published triage result for event {event_id}")
            return True
        except KafkaError as e:
            logger.error(f"Kafka publish failed for event {event_id}: {e}")
            return False

    def close(self):
        self._producer.flush()
        self._producer.close()
