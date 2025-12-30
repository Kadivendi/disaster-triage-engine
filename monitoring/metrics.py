"""Prometheus-compatible metrics for model performance and feed ingestion.

Tracks prediction accuracy, latency distributions, and real-time feed
ingestion rates across all data sources (NOAA, USGS, NWS).
"""
import time
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional
from collections import defaultdict
from threading import Lock

logger = logging.getLogger(__name__)


@dataclass
class LatencyHistogram:
    """Tracks latency distribution with configurable bucket boundaries."""
    buckets: List[float] = field(default_factory=lambda: [1, 5, 10, 25, 50, 100, 250, 500, 1000])
    counts: Dict[float, int] = field(default_factory=lambda: defaultdict(int))
    total: float = 0.0
    count: int = 0

    def observe(self, value_ms: float) -> None:
        """Record a latency observation."""
        self.total += value_ms
        self.count += 1
        for boundary in self.buckets:
            if value_ms <= boundary:
                self.counts[boundary] += 1
                return
        self.counts[float('inf')] += 1

    @property
    def avg(self) -> float:
        return self.total / self.count if self.count > 0 else 0.0


class TriageMetrics:
    """Central metrics registry for the disaster triage engine.

    Thread-safe metrics collection for:
    - Model prediction accuracy and latency
    - Feed ingestion rates and freshness
    - Event classification distribution
    """

    def __init__(self):
        self._lock = Lock()
        self._counters: Dict[str, int] = defaultdict(int)
        self._gauges: Dict[str, float] = {}
        self._histograms: Dict[str, LatencyHistogram] = defaultdict(LatencyHistogram)
        self._prediction_results: Dict[str, int] = defaultdict(int)
        logger.info("TriageMetrics initialized")

    def record_prediction(self, severity: str, latency_ms: float,
                          escalation_predicted: bool) -> None:
        """Record a completed prediction with its result and latency."""
        with self._lock:
            self._counters["predictions_total"] += 1
            self._prediction_results[severity] += 1
            self._histograms["prediction_latency"].observe(latency_ms)
            if escalation_predicted:
                self._counters["escalations_predicted"] += 1

    def record_feed_ingestion(self, source: str, event_count: int,
                               latency_ms: float) -> None:
        """Record a feed ingestion cycle with its results."""
        with self._lock:
            self._counters[f"feed_events_{source}"] += event_count
            self._counters[f"feed_polls_{source}"] += 1
            self._histograms[f"feed_latency_{source}"].observe(latency_ms)
            self._gauges[f"feed_last_poll_{source}"] = time.time()

    def record_classification_error(self, source: str, error_type: str) -> None:
        """Record a classification or ingestion error."""
        with self._lock:
            self._counters[f"errors_{source}_{error_type}"] += 1

    def set_model_accuracy(self, model_name: str, accuracy: float) -> None:
        """Update the current accuracy gauge for a model."""
        with self._lock:
            self._gauges[f"model_accuracy_{model_name}"] = accuracy

    def get_snapshot(self) -> Dict:
        """Return a point-in-time snapshot of all metrics."""
        with self._lock:
            return {
                "counters": dict(self._counters),
                "gauges": dict(self._gauges),
                "histograms": {
                    name: {"avg_ms": h.avg, "count": h.count, "total_ms": h.total}
                    for name, h in self._histograms.items()
                },
                "severity_distribution": dict(self._prediction_results),
                "timestamp": time.time(),
            }


# Module-level singleton for global access
_metrics_instance: Optional[TriageMetrics] = None


def get_metrics() -> TriageMetrics:
    """Get or create the global metrics instance."""
    global _metrics_instance
    if _metrics_instance is None:
        _metrics_instance = TriageMetrics()
    return _metrics_instance
