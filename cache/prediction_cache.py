"""Redis-compatible prediction cache to avoid redundant model inference.

In disaster scenarios with rapidly updating feeds, the same event
coordinates and features may be submitted multiple times within seconds.
This cache prevents redundant GPU/CPU inference cycles while ensuring
predictions stay fresh within a configurable TTL window.
"""
import hashlib
import json
import logging
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional
from collections import OrderedDict
from threading import Lock

logger = logging.getLogger(__name__)


@dataclass
class CacheEntry:
    """Single cache entry with value and expiration metadata."""
    value: Dict[str, Any]
    created_at: float
    ttl_seconds: float
    hit_count: int = 0

    @property
    def is_expired(self) -> bool:
        return time.time() - self.created_at > self.ttl_seconds


class PredictionCache:
    """LRU prediction cache with TTL-based expiration.

    Features:
    - SHA-256 key derivation from input features for deterministic caching
    - LRU eviction when max capacity is reached
    - Per-entry TTL with lazy expiration on access
    - Hit/miss ratio tracking for cache performance monitoring
    """

    def __init__(self, max_size: int = 10000, default_ttl: float = 300.0):
        self._max_size = max_size
        self._default_ttl = default_ttl
        self._cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self._lock = Lock()
        self._hits = 0
        self._misses = 0
        logger.info(
            "PredictionCache initialized: max_size=%d, ttl=%.0fs",
            max_size, default_ttl,
        )

    def get(self, features: Dict[str, float]) -> Optional[Dict[str, Any]]:
        """Look up a cached prediction result by input features."""
        key = self._compute_key(features)
        with self._lock:
            entry = self._cache.get(key)
            if entry is None:
                self._misses += 1
                return None
            if entry.is_expired:
                del self._cache[key]
                self._misses += 1
                logger.debug("Cache entry expired: key=%s", key[:12])
                return None
            # Move to end for LRU ordering
            self._cache.move_to_end(key)
            entry.hit_count += 1
            self._hits += 1
            return entry.value

    def put(self, features: Dict[str, float], prediction: Dict[str, Any],
            ttl: Optional[float] = None) -> None:
        """Store a prediction result in the cache."""
        key = self._compute_key(features)
        entry = CacheEntry(
            value=prediction,
            created_at=time.time(),
            ttl_seconds=ttl or self._default_ttl,
        )
        with self._lock:
            if key in self._cache:
                del self._cache[key]
            elif len(self._cache) >= self._max_size:
                evicted_key, _ = self._cache.popitem(last=False)
                logger.debug("LRU eviction: key=%s", evicted_key[:12])
            self._cache[key] = entry

    def invalidate(self, features: Dict[str, float]) -> bool:
        """Remove a specific entry from the cache."""
        key = self._compute_key(features)
        with self._lock:
            if key in self._cache:
                del self._cache[key]
                return True
            return False

    def clear(self) -> int:
        """Clear all entries and return the count of removed entries."""
        with self._lock:
            count = len(self._cache)
            self._cache.clear()
            logger.info("Cache cleared: %d entries removed", count)
            return count

    @property
    def stats(self) -> Dict[str, Any]:
        """Return cache performance statistics."""
        total = self._hits + self._misses
        return {
            "size": len(self._cache),
            "max_size": self._max_size,
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": self._hits / total if total > 0 else 0.0,
            "default_ttl_seconds": self._default_ttl,
        }

    @staticmethod
    def _compute_key(features: Dict[str, float]) -> str:
        """Compute a deterministic SHA-256 key from sorted feature values."""
        normalized = json.dumps(features, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(normalized.encode("utf-8")).hexdigest()
