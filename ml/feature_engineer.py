"""
Feature Engineering Pipeline
Transforms raw disaster events (NOAA/USGS/NWS) into normalized feature
vectors for ML model input. Handles missing values, encoding, and scaling.
"""
import math
import hashlib
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

EVENT_TYPE_MAP = {
    "wildfire": 0, "earthquake": 1, "hurricane": 2, "tornado": 3,
    "flood": 4, "tsunami": 5, "blizzard": 6, "drought": 7,
    "landslide": 8, "volcanic eruption": 9, "heat wave": 10,
    "winter storm": 11, "thunderstorm": 12, "hail": 13, "fog": 14,
}

SEVERITY_TO_LABEL = {
    "Extreme": "EXTREME", "Severe": "SEVERE",
    "Moderate": "MODERATE", "Minor": "MINOR",
    "Unknown": "MINOR",
}


@dataclass
class FeatureVector:
    event_type_encoded: float
    magnitude_or_wind_speed: float
    population_density_at_center: float
    infrastructure_vulnerability_score: float
    prior_24h_event_count: float
    network_degradation_index: float
    time_of_day_risk_factor: float
    evacuation_route_availability: float
    hospital_proximity_score: float
    weather_compounding_factor: float

    def to_list(self) -> list[float]:
        return [
            self.event_type_encoded,
            self.magnitude_or_wind_speed,
            self.population_density_at_center,
            self.infrastructure_vulnerability_score,
            self.prior_24h_event_count,
            self.network_degradation_index,
            self.time_of_day_risk_factor,
            self.evacuation_route_availability,
            self.hospital_proximity_score,
            self.weather_compounding_factor,
        ]


class FeatureEngineer:
    """Transforms raw disaster event data into ML-ready feature vectors."""

    def extract(
        self,
        event_type: str,
        magnitude: float,
        latitude: float,
        longitude: float,
        timestamp: Optional[datetime] = None,
        prior_event_count: int = 0,
        population_density: float = 1000.0,
        vulnerability_score: float = 0.5,
    ) -> FeatureVector:
        ts = timestamp or datetime.utcnow()
        return FeatureVector(
            event_type_encoded=self._encode_event_type(event_type),
            magnitude_or_wind_speed=self._normalize_magnitude(magnitude, event_type),
            population_density_at_center=self._normalize_density(population_density),
            infrastructure_vulnerability_score=vulnerability_score,
            prior_24h_event_count=min(prior_event_count / 10.0, 1.0),
            network_degradation_index=self._compute_network_degradation(magnitude, event_type),
            time_of_day_risk_factor=self._time_risk(ts),
            evacuation_route_availability=self._evac_score(latitude, longitude),
            hospital_proximity_score=self._hospital_score(latitude, longitude),
            weather_compounding_factor=self._weather_compound(event_type, magnitude),
        )

    def _encode_event_type(self, event_type: str) -> float:
        key = event_type.lower().strip()
        for k, v in EVENT_TYPE_MAP.items():
            if k in key:
                return float(v) / len(EVENT_TYPE_MAP)
        return float(int(hashlib.md5(key.encode()).hexdigest(), 16) % len(EVENT_TYPE_MAP)) / len(EVENT_TYPE_MAP)

    def _normalize_magnitude(self, magnitude: float, event_type: str) -> float:
        if "earthquake" in event_type.lower():
            return min(magnitude / 10.0, 1.0)
        return min(magnitude / 200.0, 1.0)  # wind speed in knots

    def _normalize_density(self, density: float) -> float:
        return min(density / 20000.0, 1.0)

    def _compute_network_degradation(self, magnitude: float, event_type: str) -> float:
        base = min(magnitude / 8.0, 1.0)
        if any(t in event_type.lower() for t in ["earthquake", "tsunami", "hurricane"]):
            return min(base * 1.4, 1.0)
        return base * 0.7

    def _time_risk(self, ts: datetime) -> float:
        hour = ts.hour
        if 22 <= hour or hour < 6:
            return 0.9  # night: higher risk (people asleep)
        elif 6 <= hour < 9 or 17 <= hour < 20:
            return 0.7  # rush hour
        return 0.4

    def _evac_score(self, lat: float, lon: float) -> float:
        return max(0.1, 1.0 - abs(lat % 3) / 3.0)

    def _hospital_score(self, lat: float, lon: float) -> float:
        return max(0.1, 1.0 - (abs(lat) % 5) / 5.0)

    def _weather_compound(self, event_type: str, magnitude: float) -> float:
        if "wildfire" in event_type.lower() or "hurricane" in event_type.lower():
            return min(0.3 + magnitude / 15.0, 1.0)
        return 0.2
