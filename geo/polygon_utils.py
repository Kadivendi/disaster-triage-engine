"""
Polygon utilities for geo-fencing and risk zone calculations.
"""
from shapely.geometry import Point, Polygon
from typing import List, Tuple

def is_point_in_polygon(lat: float, lon: float, polygon_coords: List[Tuple[float, float]]) -> bool:
    """
    Check if a given (lat, lon) is inside a polygon defined by a list of (lat, lon) tuples.
    """
    if len(polygon_coords) < 3:
        return False

    point = Point(lon, lat) # Shapely uses (x, y) = (lon, lat)
    poly = Polygon([(lon, lat) for lat, lon in polygon_coords])
    return poly.contains(point)
