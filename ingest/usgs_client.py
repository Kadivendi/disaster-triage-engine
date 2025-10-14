"""
USGS Earthquake Alert Ingestor
Subscribes to the USGS Earthquake Hazards Program real-time GeoJSON feed
and converts seismic events into normalized disaster alert objects.
"""
import httpx
import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Generator

logger = logging.getLogger(__name__)

USGS_FEED_URL = "https://earthquake.usgs.gov/earthquakes/feed/v1.0/summary/significant_hour.geojson"
USGS_ALL_DAY_URL = "https://earthquake.usgs.gov/earthquakes/feed/v1.0/summary/all_day.geojson"


@dataclass
class SeismicEvent:
    event_id: str
    magnitude: float
    magnitude_type: str
    place: str
    depth_km: float
    latitude: float
    longitude: float
    occurred_at: datetime
    alert_level: str | None  # green/yellow/orange/red
    tsunami_flag: bool
    felt_count: int
    cdi: float | None  # Community Decimal Intensity
    mmi: float | None  # Modified Mercalli Intensity
    url: str


class USGSClient:
    """Real-time USGS seismic event ingestion client."""

    def __init__(self, min_magnitude: float = 2.5, timeout_seconds: int = 10):
        self._client = httpx.Client(timeout=timeout_seconds)
        self.min_magnitude = min_magnitude

    def fetch_significant_events(self) -> Generator[SeismicEvent, None, None]:
        """Fetch significant earthquakes from the past hour."""
        try:
            response = self._client.get(USGS_FEED_URL)
            response.raise_for_status()
            data = response.json()
            for feature in data.get("features", []):
                if feature["properties"]["mag"] >= self.min_magnitude:
                    yield self._parse_feature(feature)
        except Exception as e:
            logger.error(f"USGS fetch failed: {e}")

    def _parse_feature(self, feature: dict) -> SeismicEvent:
        props = feature["properties"]
        coords = feature["geometry"]["coordinates"]
        return SeismicEvent(
            event_id=feature["id"],
            magnitude=props["mag"],
            magnitude_type=props.get("magType", "unknown"),
            place=props.get("place", ""),
            depth_km=coords[2],
            latitude=coords[1],
            longitude=coords[0],
            occurred_at=datetime.utcfromtimestamp(props["time"] / 1000),
            alert_level=props.get("alert"),
            tsunami_flag=bool(props.get("tsunami", 0)),
            felt_count=props.get("felt") or 0,
            cdi=props.get("cdi"),
            mmi=props.get("mmi"),
            url=props.get("url", ""),
        )
