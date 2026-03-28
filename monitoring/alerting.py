"""Internal alerting for model drift detection and feed health monitoring.

Detects when the model's prediction distribution shifts significantly
from the expected baseline, indicating potential model degradation or
a genuine change in disaster patterns that requires retraining.
"""
import logging
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass
class DriftAlert:
    """Alert generated when model drift is detected."""
    metric_name: str
    current_value: float
    baseline_value: float
    drift_magnitude: float
    severity: str  # WARNING or CRITICAL
    timestamp: float = field(default_factory=time.time)
    message: str = ""


class ModelDriftDetector:
    """Detects distribution shifts in model predictions over time.

    Uses a sliding window approach to compare recent prediction
    distributions against a historical baseline. Alerts are generated
    when the distribution shift exceeds configurable thresholds.
    """

    def __init__(self, window_size: int = 1000, warning_threshold: float = 0.15,
                 critical_threshold: float = 0.30):
        self.window_size = window_size
        self.warning_threshold = warning_threshold
        self.critical_threshold = critical_threshold
        self._baseline: Dict[str, float] = {}
        self._recent: Deque[str] = deque(maxlen=window_size)
        self._alerts: List[DriftAlert] = []
        logger.info(
            "DriftDetector initialized: window=%d, warn=%.2f, crit=%.2f",
            window_size, warning_threshold, critical_threshold,
        )

    def set_baseline(self, distribution: Dict[str, float]) -> None:
        """Set the expected baseline distribution from training data."""
        total = sum(distribution.values())
        self._baseline = {k: v / total for k, v in distribution.items()}
        logger.info("Baseline distribution set: %s", self._baseline)

    def record_prediction(self, severity: str) -> Optional[DriftAlert]:
        """Record a prediction and check for drift."""
        self._recent.append(severity)
        if len(self._recent) < self.window_size:
            return None
        return self._check_drift()

    def _check_drift(self) -> Optional[DriftAlert]:
        """Compare recent distribution against baseline."""
        if not self._baseline:
            return None

        recent_dist = self._compute_distribution()
        max_drift = 0.0
        max_drift_class = ""

        for cls, baseline_pct in self._baseline.items():
            recent_pct = recent_dist.get(cls, 0.0)
            drift = abs(recent_pct - baseline_pct)
            if drift > max_drift:
                max_drift = drift
                max_drift_class = cls

        if max_drift >= self.critical_threshold:
            alert = DriftAlert(
                metric_name=f"severity_distribution_{max_drift_class}",
                current_value=recent_dist.get(max_drift_class, 0.0),
                baseline_value=self._baseline.get(max_drift_class, 0.0),
                drift_magnitude=max_drift,
                severity="CRITICAL",
                message=f"Critical drift in {max_drift_class}: {max_drift:.1%} from baseline",
            )
            self._alerts.append(alert)
            logger.critical("Model drift CRITICAL: %s", alert.message)
            return alert
        elif max_drift >= self.warning_threshold:
            alert = DriftAlert(
                metric_name=f"severity_distribution_{max_drift_class}",
                current_value=recent_dist.get(max_drift_class, 0.0),
                baseline_value=self._baseline.get(max_drift_class, 0.0),
                drift_magnitude=max_drift,
                severity="WARNING",
                message=f"Drift detected in {max_drift_class}: {max_drift:.1%} from baseline",
            )
            self._alerts.append(alert)
            logger.warning("Model drift WARNING: %s", alert.message)
            return alert
        return None

    def _compute_distribution(self) -> Dict[str, float]:
        """Compute the distribution of recent predictions."""
        counts: Dict[str, int] = {}
        total = len(self._recent)
        for severity in self._recent:
            counts[severity] = counts.get(severity, 0) + 1
        return {k: v / total for k, v in counts.items()}

    def get_recent_alerts(self, limit: int = 20) -> List[DriftAlert]:
        """Return the most recent drift alerts."""
        return self._alerts[-limit:]


class FeedHealthMonitor:
    """Monitors feed freshness and alerts on staleness."""

    def __init__(self, staleness_threshold_seconds: float = 300.0):
        self.staleness_threshold = staleness_threshold_seconds
        self._last_poll: Dict[str, float] = {}

    def record_poll(self, source: str) -> None:
        """Record a successful feed poll."""
        self._last_poll[source] = time.time()

    def check_health(self) -> Dict[str, Dict]:
        """Check all feeds for staleness."""
        now = time.time()
        results = {}
        for source, last in self._last_poll.items():
            age = now - last
            results[source] = {
                "last_poll_seconds_ago": round(age, 1),
                "status": "STALE" if age > self.staleness_threshold else "HEALTHY",
            }
        return results
