"""
data_generator.py
Generates simulated vessel base data and trajectories for the
Illegal Fishing Detection System — Round 2.

Changes from Round 1:
  - Hard-coded risk_score / risk_level / zone_status REMOVED from seed data.
  - Hard-coded ALERTS / RISK_BREAKDOWNS REMOVED (now generated dynamically).
  - generate_vessel_dataframe() returns only raw observable fields:
      vessel_id, latitude, longitude, speed, heading, behavior
  - build_all_trajectories() returns {vessel_id: [(lat, lon), ...]}
  - RESTRICTED_ZONES and generate_trajectory() are unchanged — still used by
    map_builder.py and feature_engineering.py.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple

# ---------------------------------------------------------------------------
# Constants / seed
# ---------------------------------------------------------------------------
RNG = np.random.default_rng(seed=42)

# ---------------------------------------------------------------------------
# Restricted zone definitions  (consumed by geofencing.py and map_builder.py)
# ---------------------------------------------------------------------------
RESTRICTED_ZONES: List[Dict] = [
    {
        "name": "Restricted Zone A",
        "color": "#FF4444",
        "fill_color": "#FF444433",
        "coords": [          # polygon vertices [lat, lon]
            [12.60, 80.25],
            [12.60, 80.55],
            [12.35, 80.55],
            [12.35, 80.25],
        ],
        "center": (12.475, 80.40),
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
    },
]

# ---------------------------------------------------------------------------
# Vessel seed definitions
# Each entry: (vessel_id, base_lat, base_lon, speed_kn, heading_deg, behavior)
#
# behavior drives trajectory shape in generate_trajectory().
# risk_score / risk_level / zone_status are NOT seeded here — the pipeline
# calculates them from real geofencing + feature extraction + Isolation Forest.
# ---------------------------------------------------------------------------
_VESSEL_SEEDS: List[Tuple] = [
    # ── Vessels with suspicious / high-risk behaviour profiles ─────────────
    # V102: slow + inside Zone A → pipeline should detect HIGH risk
    ("V102", 12.50, 80.38,  2.1, 185, "Loitering / Zone Violation"),
    # V087: loitering near Zone B
    ("V087", 13.02, 79.92,  1.8, 210, "Suspicious Loitering"),
    # V215: inside Zone C with erratic movement
    ("V215", 11.70, 80.72,  3.2, 160, "Erratic Movement"),

    # ── Vessels with medium-risk behaviour profiles ────────────────────────
    ("V034", 12.80, 80.60,  6.5,  45, "Speed Anomaly"),
    ("V119", 11.50, 80.20,  5.8, 320, "AIS Signal Gap"),
    ("V201", 13.30, 80.40,  4.2, 270, "Night-time Activity"),
    ("V156", 12.20, 79.90,  7.1,  90, "Irregular Route"),
    ("V093", 11.80, 80.90,  3.9, 135, "Proximity to Restricted Zone"),
    ("V178", 13.50, 79.60,  5.5, 200, "Repeated Zone Approach"),

    # ── Normal vessels ────────────────────────────────────────────────────
    ("V011", 12.00, 80.00, 12.3,  10, "Normal Transit"),
    ("V045", 13.70, 80.20, 10.8, 350, "Normal Transit"),
    ("V067", 11.30, 79.70,  9.5,  90, "Normal Fishing"),
    ("V133", 12.90, 81.00, 11.2, 270, "Normal Transit"),
    ("V244", 11.10, 80.50,  8.7, 180, "Normal Fishing"),
    ("V299", 13.90, 79.40, 13.1,  60, "Normal Transit"),
    ("V312", 12.45, 81.20, 10.4, 340, "Normal Fishing"),
    ("V007", 11.60, 79.50, 14.0,  20, "Normal Transit"),
    ("V188", 13.20, 81.10,  9.2, 290, "Normal Fishing"),
]


# ---------------------------------------------------------------------------
# Public API — vessel data
# ---------------------------------------------------------------------------

def generate_vessel_dataframe() -> pd.DataFrame:
    """
    Return a DataFrame of raw observable vessel fields.
    Columns: vessel_id, latitude, longitude, speed, heading, behavior

    risk_score / risk_level / zone_status are NOT included here —
    they are computed by the Round 2 pipeline and merged into this DataFrame
    inside app.py before the dashboard renders.
    """
    rows = []
    for seed in _VESSEL_SEEDS:
        vid, lat, lon, spd, hdg, behavior = seed
        rows.append({
            "vessel_id": vid,
            "latitude":  round(lat, 4),
            "longitude": round(lon, 4),
            "speed":     round(spd, 1),
            "heading":   hdg,
            "behavior":  behavior,
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Trajectory generation
# ---------------------------------------------------------------------------

def generate_trajectory(lat: float, lon: float,
                        behavior: str,
                        n_points: int = 12) -> List[Tuple[float, float]]:
    """
    Return a list of (lat, lon) waypoints simulating historical vessel movement.
    Trajectory shape varies with behaviour type.
    Called by both map_builder.py and feature_engineering.py.
    """
    points: List[Tuple[float, float]] = []
    cur_lat, cur_lon = lat, lon

    if "Loitering" in behavior or "Zone Violation" in behavior:
        # Tight circular pattern — vessel is not making progress
        for i in range(n_points):
            angle = 2 * np.pi * i / n_points
            dlat = 0.03 * np.sin(angle) + RNG.uniform(-0.005, 0.005)
            dlon = 0.03 * np.cos(angle) + RNG.uniform(-0.005, 0.005)
            points.append((round(cur_lat + dlat, 5),
                           round(cur_lon + dlon, 5)))

    elif "Erratic" in behavior:
        # Random walk — no coherent pattern
        for _ in range(n_points):
            cur_lat += RNG.uniform(-0.06, 0.06)
            cur_lon += RNG.uniform(-0.06, 0.06)
            points.append((round(cur_lat, 5), round(cur_lon, 5)))

    elif "Normal Transit" in behavior:
        # Mostly straight line with minor noise
        dlat = RNG.uniform(-0.04, 0.04)
        dlon = RNG.uniform(0.03, 0.08)
        for i in range(n_points):
            t = (n_points - 1 - i) / (n_points - 1)
            points.append((
                round(lat - t * dlat * n_points + RNG.uniform(-0.005, 0.005), 5),
                round(lon - t * dlon * n_points + RNG.uniform(-0.005, 0.005), 5),
            ))

    else:
        # Generic drift (fishing, irregular, proximity behaviour)
        for i in range(n_points):
            t = (n_points - 1 - i) / (n_points - 1)
            points.append((
                round(lat + t * RNG.uniform(-0.15, 0.15), 5),
                round(lon + t * RNG.uniform(-0.15, 0.15), 5),
            ))

    # Always end at the current vessel position
    points.append((lat, lon))
    return points


def build_all_trajectories(vessel_df: pd.DataFrame) -> Dict[str, List[Tuple[float, float]]]:
    """
    Build a trajectory for every vessel in vessel_df.
    Returns {vessel_id: [(lat, lon), ...]}
    """
    trajectories: Dict[str, List[Tuple[float, float]]] = {}
    for _, row in vessel_df.iterrows():
        trajectories[row["vessel_id"]] = generate_trajectory(
            lat=row["latitude"],
            lon=row["longitude"],
            behavior=row["behavior"],
        )
    return trajectories


# ---------------------------------------------------------------------------
# Restricted zones accessor (used by geofencing.py and map_builder.py)
# ---------------------------------------------------------------------------

def get_restricted_zones() -> List[Dict]:
    """Return the list of restricted zone definitions."""
    return RESTRICTED_ZONES
