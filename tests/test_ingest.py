"""Tests for NOAA and USGS data ingest clients."""
import pytest
from unittest.mock import MagicMock, patch
from ingest.noaa_client import NOAAClient, WeatherAlert
from ingest.usgs_client import USGSClient, SeismicEvent

SAMPLE_NOAA_XML = """<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom"
      xmlns:cap="urn:oasis:names:tc:emergency:cap:1.2">
  <entry>
    <id>https://api.weather.gov/alerts/urn:oid:2.49.0.1.840.0.test</id>
    <cap:event>Wildfire Warning</cap:event>
    <cap:severity>Extreme</cap:severity>
    <cap:urgency>Immediate</cap:urgency>
    <cap:certainty>Observed</cap:certainty>
    <cap:areaDesc>Los Angeles County</cap:areaDesc>
    <cap:polygon>34.05,-118.24 34.10,-118.20 34.08,-118.30 34.05,-118.24</cap:polygon>
    <cap:sent>2025-10-15T14:30:00-07:00</cap:sent>
    <cap:expires>2025-10-16T14:30:00-07:00</cap:expires>
    <cap:headline>Wildfire Warning issued for Los Angeles County</cap:headline>
    <cap:description>Extreme fire danger conditions in effect.</cap:description>
  </entry>
</feed>"""

SAMPLE_USGS_RESPONSE = {
    "features": [
        {
            "id": "us7000test",
            "properties": {
                "mag": 6.2,
                "magType": "mw",
                "place": "15km SW of San Jose, CA",
                "time": 1697385600000,
                "alert": "yellow",
                "tsunami": 0,
                "felt": 1240,
                "cdi": 5.8,
                "mmi": 6.1,
                "url": "https://earthquake.usgs.gov/earthquakes/eventpage/us7000test",
            },
            "geometry": {"coordinates": [-121.89, 37.34, 8.5]},
        }
    ]
}


class TestNOAAClient:

    def test_parse_sample_noaa_feed(self):
        client = NOAAClient()
        alerts = list(client._parse_feed(SAMPLE_NOAA_XML))
        assert len(alerts) == 1
        alert = alerts[0]
        assert alert.event_type == "Wildfire Warning"
        assert alert.severity == "Extreme"
        assert alert.urgency == "Immediate"
        assert alert.area_desc == "Los Angeles County"
        assert len(alert.polygon) == 4

    def test_polygon_parsed_correctly(self):
        client = NOAAClient()
        alerts = list(client._parse_feed(SAMPLE_NOAA_XML))
        poly = alerts[0].polygon
        assert poly[0] == pytest.approx((34.05, -118.24))

    def test_alert_has_headline(self):
        client = NOAAClient()
        alerts = list(client._parse_feed(SAMPLE_NOAA_XML))
        assert "Wildfire Warning" in alerts[0].headline


class TestUSGSClient:

    def test_parse_usgs_feature(self):
        client = USGSClient()
        event = client._parse_feature(SAMPLE_USGS_RESPONSE["features"][0])
        assert isinstance(event, SeismicEvent)
        assert event.magnitude == pytest.approx(6.2)
        assert event.alert_level == "yellow"
        assert event.depth_km == pytest.approx(8.5)
        assert event.latitude == pytest.approx(37.34)
        assert event.longitude == pytest.approx(-121.89)

    def test_magnitude_filter(self):
        client = USGSClient(min_magnitude=7.0)
        with patch.object(client._client, "get") as mock_get:
            mock_get.return_value = MagicMock(
                status_code=200,
                json=lambda: SAMPLE_USGS_RESPONSE,
                raise_for_status=lambda: None,
            )
            events = list(client.fetch_significant_events())
        # mag 6.2 should be filtered out (below 7.0)
        assert len(events) == 0

    def test_tsunami_flag_parsed(self):
        client = USGSClient()
        event = client._parse_feature(SAMPLE_USGS_RESPONSE["features"][0])
        assert event.tsunami_flag is False
