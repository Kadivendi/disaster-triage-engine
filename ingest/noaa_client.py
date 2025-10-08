"""
NOAA Weather Alert Ingestor
Polls the NOAA CAP Atom feed for active weather alerts and publishes
normalized events to the internal Kafka topic for downstream ML processing.
"""
import httpx
import logging
from dataclasses import dataclass
from datetime import datetime
from xml.etree import ElementTree as ET
from typing import Generator

logger = logging.getLogger(__name__)

NOAA_CAP_FEED = "https://api.weather.gov/alerts/active?status=actual&message_type=alert"
NOAA_NAMESPACES = {
    "atom": "http://www.w3.org/2005/Atom",
    "cap": "urn:oasis:names:tc:emergency:cap:1.2",
}


@dataclass
class WeatherAlert:
    alert_id: str
    event_type: str
    severity: str
    urgency: str
    certainty: str
    area_desc: str
    polygon: list[tuple[float, float]] | None
    issued_at: datetime
    expires_at: datetime
    headline: str
    description: str
    raw_xml: str


class NOAAClient:
    """Real-time NOAA weather alert ingestion client."""

    def __init__(self, timeout_seconds: int = 10):
        self._client = httpx.Client(timeout=timeout_seconds)

    def fetch_active_alerts(self) -> Generator[WeatherAlert, None, None]:
        """Fetch all currently active NOAA weather alerts."""
        try:
            response = self._client.get(NOAA_CAP_FEED)
            response.raise_for_status()
            yield from self._parse_feed(response.text)
        except httpx.HTTPError as e:
            logger.error(f"NOAA fetch failed: {e}")

    def _parse_feed(self, xml_text: str) -> Generator[WeatherAlert, None, None]:
        root = ET.fromstring(xml_text)
        for entry in root.findall("atom:entry", NOAA_NAMESPACES):
            try:
                yield self._parse_entry(entry)
            except Exception as e:
                logger.warning(f"Failed to parse NOAA entry: {e}")

    def _parse_entry(self, entry: ET.Element) -> WeatherAlert:
        cap = lambda tag: entry.findtext(f"cap:{tag}", namespaces=NOAA_NAMESPACES) or ""
        polygon_text = cap("polygon")
        polygon = self._parse_polygon(polygon_text) if polygon_text else None

        return WeatherAlert(
            alert_id=entry.findtext("atom:id", namespaces=NOAA_NAMESPACES) or "",
            event_type=cap("event"),
            severity=cap("severity"),
            urgency=cap("urgency"),
            certainty=cap("certainty"),
            area_desc=cap("areaDesc"),
            polygon=polygon,
            issued_at=datetime.fromisoformat(cap("sent").replace("Z", "+00:00")),
            expires_at=datetime.fromisoformat(cap("expires").replace("Z", "+00:00")),
            headline=cap("headline"),
            description=cap("description"),
            raw_xml=ET.tostring(entry, encoding="unicode"),
        )

    @staticmethod
    def _parse_polygon(polygon_text: str) -> list[tuple[float, float]]:
        coords = []
        for pair in polygon_text.strip().split():
            lat, lon = pair.split(",")
            coords.append((float(lat), float(lon)))
        return coords
