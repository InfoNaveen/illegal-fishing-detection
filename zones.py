"""
zones.py
Source-aware geographic zone configuration for the
Illegal Fishing Detection System.

Round 3 — regional geofencing configuration.

Background
---------
Restricted / monitoring zones were previously hard-coded in data_generator.py
around the Bay of Bengal. The simulated vessel fleet lives there, but the
bundled historical AIS dataset is from Danish waters, so those Bay-of-Bengal
polygons never overlap real AIS positions.

This module makes zones configurable per data source WITHOUT changing the
geofencing algorithm. ``get_zones(source)`` returns a list of zone dicts in the
exact shape the rest of the application already expects:

    {
        "name":       str,          # label shown on the map / popups
        "color":      "#RRGGBB",
        "fill_color": "#RRGGBBAA",
        "coords":     [[lat, lon], ...],   # polygon vertices
        "center":     (lat, lon),          # label anchor
        "zone_type":  "restricted" | "monitoring",  # honesty flag (new, optional)
    }

Data honesty
------------
- The Bay of Bengal polygons are **demonstration restricted zones** for the
  synthetic fleet. They are not real legal boundaries.
- The Danish polygons are **demonstration monitoring zones** placed over
  regions where the historical AIS dataset actually has dense vessel traffic.
  They are NOT authoritative Danish restricted/protected/fishing areas and must
  never be presented as legally restricted waters. They exist only to exercise
  the geofencing pipeline against real vessel positions.
"""

from typing import Dict, List

# ---------------------------------------------------------------------------
# Simulated environment — Bay of Bengal demonstration restricted zones.
# These are an EXACT copy of the original data_generator.RESTRICTED_ZONES so
# the simulated path behaves identically. Do not change the geometry.
# ---------------------------------------------------------------------------
BAY_OF_BENGAL_ZONES: List[Dict] = [
    {
        "name": "Restricted Zone A",
        "color": "#FF4444",
        "fill_color": "#FF444433",
        "coords": [
            [12.60, 80.25],
            [12.60, 80.55],
            [12.35, 80.55],
            [12.35, 80.25],
        ],
        "center": (12.475, 80.40),
        "zone_type": "restricted",
    },
    {
        "name": "Restricted Zone B",
        "color": "#FF8C00",
        "fill_color": "#FF8C0033",
        "coords": [
            [13.10, 79.80],
            [13.10, 80.10],
            [12.85, 80.10],
            [12.85, 79.80],
        ],
        "center": (12.975, 79.95),
        "zone_type": "restricted",
    },
    {
        "name": "Restricted Zone C",
        "color": "#FF4444",
        "fill_color": "#FF444433",
        "coords": [
            [11.90, 80.60],
            [11.90, 80.90],
            [11.65, 80.90],
            [11.65, 80.60],
        ],
        "center": (11.775, 80.75),
        "zone_type": "restricted",
    },
]

# ---------------------------------------------------------------------------
# Historical AIS environment — Danish waters DEMONSTRATION MONITORING zones.
#
# Polygon placement is derived from the observed geographic density of the
# bundled historical AIS dataset (2025-02-27) so the geofencing pipeline has
# real vessel positions to evaluate. These are NOT authoritative restricted or
# protected areas — they are demonstration monitoring zones only.
#
# Dense traffic regions used (latest-position vessel counts):
#   - Øresund / Copenhagen approach : lat ~55.5–56.0, lon ~12.5–13.0
#   - Skagerrak / North Jutland     : lat ~57.5–58.0, lon ~10.5–11.0
#   - North Sea / West Jutland      : lat ~55.0–55.5, lon ~ 8.0– 8.5
# ---------------------------------------------------------------------------
DANISH_DEMO_ZONES: List[Dict] = [
    {
        "name": "Monitoring Zone A (Øresund)",
        "color": "#00B4D8",
        "fill_color": "#00B4D833",
        "coords": [
            [56.00, 12.50],
            [56.00, 13.00],
            [55.50, 13.00],
            [55.50, 12.50],
        ],
        "center": (55.75, 12.75),
        "zone_type": "monitoring",
    },
    {
        "name": "Monitoring Zone B (Skagerrak)",
        "color": "#9D4EDD",
        "fill_color": "#9D4EDD33",
        "coords": [
            [58.00, 10.50],
            [58.00, 11.00],
            [57.50, 11.00],
            [57.50, 10.50],
        ],
        "center": (57.75, 10.75),
        "zone_type": "monitoring",
    },
    {
        "name": "Monitoring Zone C (North Sea)",
        "color": "#00B4D8",
        "fill_color": "#00B4D833",
        "coords": [
            [55.50, 8.00],
            [55.50, 8.50],
            [55.00, 8.50],
            [55.00, 8.00],
        ],
        "center": (55.25, 8.25),
        "zone_type": "monitoring",
    },
]


# ---------------------------------------------------------------------------
# Region metadata — used by the dashboard header/info panels.
# ---------------------------------------------------------------------------
REGION_META: Dict[str, Dict] = {
    "Simulated": {
        "region":     "Bay of Bengal",
        "map_center": (12.5, 80.2),
        "map_zoom":   8,
        "zone_label": "Restricted Zones (demonstration)",
    },
    "Historical AIS": {
        "region":     "Danish Waters",
        "map_center": (56.5, 10.5),
        "map_zoom":   6,
        "zone_label": "Monitoring Zones (demonstration)",
    },
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_zones(source: str = "Simulated") -> List[Dict]:
    """
    Return the zone configuration for a given data source.

    Parameters
    ----------
    source : str
        "Simulated"      -> Bay of Bengal demonstration restricted zones
        "Historical AIS" -> Danish waters demonstration monitoring zones

    Returns
    -------
    list of zone dicts (same shape used by geofencing.run_geofencing and
    map_builder.build_map). Unknown sources default to the Bay of Bengal set.
    """
    if source == "Historical AIS":
        return DANISH_DEMO_ZONES
    return BAY_OF_BENGAL_ZONES


def get_region_meta(source: str = "Simulated") -> Dict:
    """Return display metadata (region name, map center/zoom, zone label)."""
    return REGION_META.get(source, REGION_META["Simulated"])
