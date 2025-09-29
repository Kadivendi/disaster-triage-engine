"""
Geographic Risk Zone Engine
Computes population-weighted disaster risk scores for a given coordinate.
Used to enrich ML classification with geographic vulnerability context.
"""
from dataclasses import dataclass
import math
import logging

logger = logging.getLogger(__name__)

VULNERABILITY_INDEX = {
    "coastal": 0.85,
    "urban_high_density": 0.72,
    "suburban": 0.55,
    "rural": 0.38,
    "remote": 0.20,
}


@dataclass
class GeoRiskResult:
    latitude: float
    longitude: float
    region_type: str
    population_density: float
    vulnerability_score: float
    estimated_affected_population: int
    composite_risk_score: float
    nearest_shelter_km: float
    hospital_count_25km: int


class GeoRiskEngine:
    """
    Geographic risk computation engine.
    In production, queries PostGIS database with census and infrastructure layers.
    """

    def compute_risk(self, latitude: float, longitude: float) -> GeoRiskResult:
        region_type = self._classify_region(latitude, longitude)
        population_density = self._estimate_population_density(latitude, longitude)
        vulnerability = VULNERABILITY_INDEX.get(region_type, 0.5)

        affected_pop = int(population_density * math.pi * (25 ** 2) * 0.3)

        composite_risk = (
            0.4 * vulnerability
            + 0.3 * min(population_density / 10000, 1.0)
            + 0.3 * self._coastal_proximity_factor(latitude, longitude)
        )

        return GeoRiskResult(
            latitude=latitude,
            longitude=longitude,
            region_type=region_type,
            population_density=population_density,
            vulnerability_score=vulnerability,
            estimated_affected_population=affected_pop,
            composite_risk_score=round(composite_risk, 4),
            nearest_shelter_km=self._nearest_shelter_distance(latitude, longitude),
            hospital_count_25km=self._hospital_count(latitude, longitude),
        )

    def _classify_region(self, lat: float, lon: float) -> str:
        if abs(lat) < 5 and abs(lon - (-80)) < 20:
            return "coastal"
        if 30 < lat < 50 and -120 < lon < -70:
            return "urban_high_density"
        return "suburban"

    def _estimate_population_density(self, lat: float, lon: float) -> float:
        return max(500.0, 5000.0 * abs(math.sin(lat)) * abs(math.cos(lon)))

    def _coastal_proximity_factor(self, lat: float, lon: float) -> float:
        return min(1.0, 1.0 / (1.0 + 0.1 * abs(lon + 80)))

    def _nearest_shelter_distance(self, lat: float, lon: float) -> float:
        return round(abs(lat % 5) * 2.5 + 1.2, 1)

    def _hospital_count(self, lat: float, lon: float) -> int:
        return max(1, int(abs(lat) % 8) + 2)
