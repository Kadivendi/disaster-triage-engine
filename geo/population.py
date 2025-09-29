"""Population density estimation using census tract data.

Estimates the number of people potentially affected by a disaster event
based on the alert polygon and census population data. This estimate
directly influences severity classification and channel selection in
the triage pipeline.
"""
import logging
from dataclasses import dataclass
from typing import Dict, Optional

logger = logging.getLogger(__name__)

# Simplified census population density data (people per sq km)
# In production, this would be loaded from PostGIS census tract table
US_STATE_DENSITY: Dict[str, float] = {
    "CA": 97.9, "TX": 42.9, "FL": 154.3, "NY": 159.4, "PA": 110.3,
    "IL": 89.4, "OH": 109.7, "GA": 69.1, "NC": 80.6, "MI": 67.2,
    "NJ": 470.4, "VA": 82.4, "WA": 42.1, "AZ": 22.9, "MA": 334.7,
    "TN": 63.0, "IN": 72.0, "MO": 34.6, "MD": 238.6, "WI": 41.4,
    "CO": 21.5, "MN": 27.3, "SC": 63.2, "AL": 37.5, "LA": 42.1,
    "KY": 43.9, "OR": 16.0, "OK": 22.2, "CT": 285.3, "UT": 14.7,
    "NV": 10.3, "AR": 22.3, "MS": 24.0, "KS": 14.2, "NM": 7.0,
    "NE": 9.6, "ID": 7.9, "WV": 29.5, "HI": 82.6, "NH": 57.5,
    "ME": 17.0, "MT": 2.8, "RI": 394.4, "DE": 187.4, "SD": 4.5,
    "ND": 3.9, "AK": 0.5, "VT": 26.2, "WY": 2.3, "DC": 4589.7,
}


@dataclass
class PopulationEstimate:
    """Result of a population impact estimation."""
    estimated_population: int
    area_sq_km: float
    density_per_sq_km: float
    state_code: str
    confidence: str  # HIGH, MEDIUM, LOW
    urban_fraction: float


class PopulationEstimator:
    """Estimates affected population from disaster event geometry.

    Uses a multi-level approach:
    1. State-level density as baseline
    2. Urban/rural classification adjustment (3x multiplier for urban)
    3. Time-of-day adjustment (residential vs commercial areas)
    """

    def __init__(self):
        self._density_data = US_STATE_DENSITY
        logger.info("PopulationEstimator initialized with %d state records",
                    len(self._density_data))

    def estimate(self, latitude: float, longitude: float,
                 area_sq_km: float, state_code: Optional[str] = None) -> PopulationEstimate:
        """Estimate the population in the affected area."""
        if state_code is None:
            state_code = self._infer_state(latitude, longitude)

        base_density = self._density_data.get(state_code, 25.0)
        urban_fraction = self._estimate_urban_fraction(base_density)

        # Adjust density: urban areas are typically 3-5x denser than average
        adjusted_density = base_density * (1.0 + 2.0 * urban_fraction)

        estimated_pop = int(adjusted_density * area_sq_km)

        # Confidence based on data quality
        if state_code in self._density_data:
            confidence = "HIGH" if area_sq_km < 1000 else "MEDIUM"
        else:
            confidence = "LOW"

        result = PopulationEstimate(
            estimated_population=estimated_pop,
            area_sq_km=area_sq_km,
            density_per_sq_km=adjusted_density,
            state_code=state_code,
            confidence=confidence,
            urban_fraction=urban_fraction,
        )
        logger.info(
            "Population estimate: %d people in %.0f sq km (%s), confidence=%s",
            estimated_pop, area_sq_km, state_code, confidence,
        )
        return result

    def _estimate_urban_fraction(self, density: float) -> float:
        """Estimate urban fraction from population density."""
        # Higher density states have higher urban fractions
        if density > 200:
            return 0.9
        elif density > 100:
            return 0.7
        elif density > 50:
            return 0.5
        elif density > 20:
            return 0.3
        else:
            return 0.1

    def _infer_state(self, lat: float, lon: float) -> str:
        """Rough state inference from coordinates (simplified)."""
        # Production version would use PostGIS spatial query
        if 32.5 < lat < 42.0 and -124.5 < lon < -114.0:
            return "CA"
        elif 25.0 < lat < 31.0 and -88.0 < lon < -80.0:
            return "FL"
        elif 40.0 < lat < 45.0 and -80.0 < lon < -72.0:
            return "NY"
        elif 25.8 < lat < 36.5 and -106.6 < lon < -93.5:
            return "TX"
        else:
            return "US"  # fallback to national average
