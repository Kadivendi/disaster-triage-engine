"""
NWS Warning Ingest Client
Connects to the National Weather Service API to ingest CAP alerts.
"""
import requests
import logging
from typing import Dict, Any, List

logger = logging.getLogger(__name__)

class NWSClient:
    def __init__(self, endpoint: str = "https://api.weather.gov/alerts/active"):
        self.endpoint = endpoint
        self.headers = {"User-Agent": "DisasterTriageEngine/1.0"}

    def fetch_active_warnings(self) -> List[Dict[str, Any]]:
        logger.info(f"Fetching active NWS warnings from {self.endpoint}")
        try:
            response = requests.get(self.endpoint, headers=self.headers, timeout=10)
            response.raise_for_status()
            data = response.json()
            return data.get("features", [])
        except Exception as e:
            logger.error(f"Failed to fetch NWS warnings: {e}")
            return []
