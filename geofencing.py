"""
geofencing.py
Real point-in-polygon geofencing for the Illegal Fishing Detection System.

Uses the ray-casting algorithm — no external geometry libraries required.
Zone polygons are [lat, lon] lists, exactly as defined in data_generator.py.
"""

import math
from typing import Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Ray-casting point-in-polygon
# ---------------------------------------------------------------------------

def _point_in_polygon(lat: float, lon: float,
                      polygon: List[List[float]]) -> bool:
    """
    Return True if (lat, lon) lies inside the polygon.

    polygon: list of [lat, lon] vertices (need not be closed).
    Uses the ray-casting (Jordan curve) algorithm.
    """
    n = len(polygon)
    inside = False
    j = n - 1
    for i in range(n):
        yi, xi = polygon[i][0], polygon[i][1]
        yj, xj = polygon[j][0], polygon[j][1]
        # Cast a horizontal ray to the right from the test point
        if ((yi > lat) != (yj > lat)) and \
           (lon < (xj - xi) * (lat - yi) / (yj - yi + 1e-12) + xi):
            inside = not inside
        j = i
    return inside


# ---------------------------------------------------------------------------
# Minimum distance from a point to a polygon edge (in degrees)
# ---------------------------------------------------------------------------

def _segment_min_dist(px: float, py: float,
                      ax: float, ay: float,
                      bx: float, by: float) -> float:
    """Minimum Euclidean distance from point (px,py) to segment (ax,ay)-(bx,by)."""
    dx, dy = bx - ax, by - ay
    if dx == 0 and dy == 0:
        return math.hypot(px - ax, py - ay)
    t = max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / (dx * dx + dy * dy)))
    return math.hypot(px - (ax + t * dx), py - (ay + t * dy))


def _min_dist_to_polygon(lat: float, lon: float,
                         polygon: List[List[float]]) -> float:
    """
    Minimum distance (in degrees) from point to the nearest polygon edge.
    Returns 0.0 if the point is inside.
    """
    if _point_in_polygon(lat, lon, polygon):
        return 0.0
    n = len(polygon)
    min_d = math.inf
    for i in range(n):
        a = polygon[i]
        b = polygon[(i + 1) % n]
        d = _segment_min_dist(lon, lat, a[1], a[0], b[1], b[0])
        if d < min_d:
            min_d = d
    return min_d


# ---------------------------------------------------------------------------
# Public constants
# ---------------------------------------------------------------------------

# 1 degree ≈ 111 km.  Proximity threshold = 0.15° ≈ 16 km.
PROXIMITY_THRESHOLD_DEG: float = 0.15


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def check_vessel_zones(lat: float, lon: float,
                       zones: List[Dict]) -> Dict:
    """
    Check a vessel position against all restricted zones.

    Returns a dict:
      {
        "inside":        bool   — True if inside any zone
        "zone_name":     str    — name of the zone the vessel is in (or "")
        "near":          bool   — True if within PROXIMITY_THRESHOLD_DEG of any zone
        "nearest_zone":  str    — name of the nearest zone (or "")
        "min_dist_deg":  float  — minimum distance to any zone boundary (0 if inside)
        "zone_status":   str    — human-readable status string used by the dashboard
      }
    """
    best_inside_name: str = ""
    nearest_zone_name: str = ""
    min_dist: float = math.inf

    for zone in zones:
        polygon = zone["coords"]   # list of [lat, lon]
        name    = zone["name"]

        if _point_in_polygon(lat, lon, polygon):
            best_inside_name = name
            min_dist = 0.0
            break                  # inside → highest priority, no need to check further

        d = _min_dist_to_polygon(lat, lon, polygon)
        if d < min_dist:
            min_dist = d
            nearest_zone_name = name

    inside = best_inside_name != ""
    near   = (not inside) and (min_dist <= PROXIMITY_THRESHOLD_DEG)

    if inside:
        zone_status = f"INSIDE {best_inside_name}"
    elif near:
        zone_status = f"Near {nearest_zone_name}"
    else:
        zone_status = "Open Waters"

    return {
        "inside":       inside,
        "zone_name":    best_inside_name if inside else "",
        "near":         near,
        "nearest_zone": nearest_zone_name if not inside else best_inside_name,
        "min_dist_deg": round(min_dist, 5),
        "zone_status":  zone_status,
    }


def run_geofencing(vessel_df, zones: List[Dict]) -> List[Dict]:
    """
    Run geofencing for every row in vessel_df.

    vessel_df must have columns: vessel_id, latitude, longitude.
    Returns a list of dicts (one per vessel) with all fields from check_vessel_zones
    plus vessel_id.
    """
    results = []
    for _, row in vessel_df.iterrows():
        result = check_vessel_zones(row["latitude"], row["longitude"], zones)
        result["vessel_id"] = row["vessel_id"]
        results.append(result)
    return results
