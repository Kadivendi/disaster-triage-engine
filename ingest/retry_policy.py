"""
Retry & Rate-Limit Policy
Shared exponential backoff and jitter for all external API clients.
Handles NOAA CAP feed throttling (429) and transient network errors.
"""
import time
import logging
import random
from dataclasses import dataclass, field
from typing import Callable, TypeVar, ParamSpec

logger = logging.getLogger(__name__)

P = ParamSpec("P")
T = TypeVar("T")


@dataclass
class RetryPolicy:
    max_attempts: int = 5
    base_delay: float = 1.0      # seconds
    max_delay: float = 60.0      # seconds
    backoff_factor: float = 2.0
    jitter: bool = True          # add randomness to prevent thundering herd

    def delay_for(self, attempt: int) -> float:
        """Compute the wait time before the given attempt (0-indexed)."""
        delay = min(self.base_delay * (self.backoff_factor ** attempt), self.max_delay)
        if self.jitter:
            delay *= (0.5 + random.random() * 0.5)
        return delay

    def execute(self, fn: Callable[P, T], *args: P.args, **kwargs: P.kwargs) -> T:
        last_exc: Exception | None = None
        for attempt in range(self.max_attempts):
            try:
                return fn(*args, **kwargs)
            except Exception as exc:
                last_exc = exc
                if attempt < self.max_attempts - 1:
                    delay = self.delay_for(attempt)
                    logger.warning(
                        f"Attempt {attempt + 1}/{self.max_attempts} failed: {exc}. "
                        f"Retrying in {delay:.1f}s..."
                    )
                    time.sleep(delay)
        raise RuntimeError(f"All {self.max_attempts} attempts failed") from last_exc


# Pre-configured policies for different use cases
NOAA_RETRY = RetryPolicy(max_attempts=5, base_delay=2.0, max_delay=120.0)
USGS_RETRY = RetryPolicy(max_attempts=3, base_delay=1.0, max_delay=30.0)
KAFKA_RETRY = RetryPolicy(max_attempts=10, base_delay=0.5, max_delay=10.0, jitter=False)
