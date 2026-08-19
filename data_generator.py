"""
data_generator.py
Generates simulated vessel data and trajectories for the
Illegal Fishing Detection System prototype (Round 1).
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple

# ---------------------------------------------------------------------------
# Constants / seed
# ---------------------------------------------------------------------------
RNG = np.random.default_rng(seed=42)

# ---------------------------------------------------------------------------
# Restricted zone definitions
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
# ---------------------------------------------------------------------------
# Each entry: (vessel_id, base_lat, base_lon, speed, heading,
#              risk_score, risk_level, behavior, zone_status)
_VESSEL_SEEDS = [
    # --- HIGH RISK ---
    ("V102", 12.50, 80.38, 2.1, 185, 87, "HIGH",   "Loitering / Zone Violation",   "INSIDE Restricted Zone A"),
    ("V087", 13.02, 79.92, 1.8, 210, 74, "HIGH",   "Suspicious Loitering",          "Near Restricted Zone B"),
    ("V215", 11.70, 80.72, 3.2, 160, 68, "HIGH",   "Erratic Movement",              "INSIDE Restricted Zone C"),

    # --- MEDIUM RISK ---
    ("V034", 12.80, 80.60, 6.5,  45, 52, "MEDIUM", "Speed Anomaly",                 "Open Waters"),
    ("V119", 11.50, 80.20, 5.8, 320, 48, "MEDIUM", "AIS Signal Gap",                "Open Waters"),
    ("V201", 13.30, 80.40, 4.2, 270, 45, "MEDIUM", "Night-time Activity",           "Near Restricted Zone B"),
    ("V156", 12.20, 79.90, 7.1,  90, 42, "MEDIUM", "Irregular Route",               "Open Waters"),
    ("V093", 11.80, 80.90, 3.9, 135, 38, "MEDIUM", "Proximity to Restricted Zone",  "Near Restricted Zone C"),
    ("V178", 13.50, 79.60, 5.5, 200, 35, "MEDIUM", "Repeated Zone Approach",        "Open Waters"),

    # --- LOW RISK ---
    ("V011", 12.00, 80.00, 12.3,  10, 18, "LOW",   "Normal Transit",                "Open Waters"),
    ("V045", 13.70, 80.20, 10.8, 350, 15, "LOW",   "Normal Transit",                "Open Waters"),
    ("V067", 11.30, 79.70, 9.5,   90, 12, "LOW",   "Normal Fishing",                "Open Waters"),
    ("V133", 12.90, 81.00, 11.2, 270, 10, "LOW",   "Normal Transit",                "Open Waters"),
    ("V244", 11.10, 80.50,  8.7, 180, 14, "LOW",   "Normal Fishing",                "Open Waters"),
    ("V299", 13.90, 79.40, 13.1,  60,  8, "LOW",   "Normal Transit",                "Open Waters"),
    ("V312", 12.45, 81.20, 10.4, 340, 11, "LOW",   "Normal Fishing",                "Open Waters"),
    ("V007", 11.60, 79.50, 14.0,  20,  6, "LOW",   "Normal Transit",                "Open Waters"),
    ("V188", 13.20, 81.10,  9.2, 290, 13, "LOW",   "Normal Fishing",                "Open Waters"),
]

# ---------------------------------------------------------------------------
# Risk breakdown for demo panel
# ---------------------------------------------------------------------------
RISK_BREAKDOWNS: Dict[str, List[Dict]] = {
    "V102": [
        {"factor": "Restricted Zone Violation", "score": 50},
        {"factor": "Loitering Detected",         "score": 20},
        {"factor": "Abnormal Movement Pattern",  "score": 15},
        {"factor": "AIS Signal Irregularity",    "score": 2},
    ],
    "V087": [
        {"factor": "Sustained Loitering (>4 hrs)","score": 35},
        {"factor": "Near Restricted Zone B",      "score": 25},
        {"factor": "Night-time Activity",         "score": 14},
    ],
    "V215": [
        {"factor": "Restricted Zone Violation",   "score": 50},
        {"factor": "Erratic Course Changes",      "score": 18},
    ],
}

# ---------------------------------------------------------------------------
# Alert messages
# ---------------------------------------------------------------------------
ALERTS: List[Dict] = [
    {"level": "HIGH",   "icon": "🔴", "vessel": "V102",
     "message": "V102 entered Restricted Zone A — immediate investigation required"},
    {"level": "HIGH",   "icon": "🔴", "vessel": "V087",
     "message": "V087 showing sustained loitering behaviour near Restricted Zone B"},
    {"level": "HIGH",   "icon": "🔴", "vessel": "V215",
     "message": "V215 detected inside Restricted Zone C — possible illegal fishing"},
    {"level": "MEDIUM", "icon": "🟠", "vessel": "V034",
     "message": "V034 speed anomaly detected — vessel speed inconsistent with fishing pattern"},
    {"level": "MEDIUM", "icon": "🟠", "vessel": "V119",
     "message": "V119 AIS signal gap of 47 minutes recorded"},
    {"level": "MEDIUM", "icon": "🟠", "vessel": "V201",
     "message": "V201 conducting night-time activity in monitored area"},
]

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def generate_vessel_dataframe() -> pd.DataFrame:
    """Return a DataFrame with one row per simulated vessel."""
    rows = []
    for seed in _VESSEL_SEEDS:
        (vid, lat, lon, spd, hdg, risk, rlevel, behavior, zone) = seed
        rows.append({
            "vessel_id":   vid,
            "latitude":    round(lat, 4),
            "longitude":   round(lon, 4),
            "speed":       round(spd, 1),
            "heading":     hdg,
            "risk_score":  risk,
            "risk_level":  rlevel,
            "behavior":    behavior,
            "zone_status": zone,
        })
    return pd.DataFrame(rows)


def generate_trajectory(lat: float, lon: float,
                        behavior: str,
                        n_points: int = 12) -> List[Tuple[float, float]]:
    """
    Return a list of (lat, lon) waypoints simulating past vessel movement.
    Trajectory shape varies with behavior type.
    """
    points: List[Tuple[float, float]] = []
    cur_lat, cur_lon = lat, lon

    if "Loitering" in behavior or "Zone Violation" in behavior:
        # Tight circular/looping pattern — vessel isn't going anywhere
        for i in range(n_points):
            angle = 2 * np.pi * i / n_points
            dlat = 0.03 * np.sin(angle) + RNG.uniform(-0.005, 0.005)
            dlon = 0.03 * np.cos(angle) + RNG.uniform(-0.005, 0.005)
            points.append((round(cur_lat + dlat, 5),
                           round(cur_lon + dlon, 5)))

    elif "Erratic" in behavior:
        # Random walk — no clear pattern
        for _ in range(n_points):
            cur_lat += RNG.uniform(-0.06, 0.06)
            cur_lon += RNG.uniform(-0.06, 0.06)
            points.append((round(cur_lat, 5), round(cur_lon, 5)))

    elif "Normal Transit" in behavior:
        # Mostly straight line with small noise
        dlat = RNG.uniform(-0.04, 0.04)
        dlon = RNG.uniform(0.03, 0.08)
        for i in range(n_points):
            t = (n_points - 1 - i) / (n_points - 1)
            points.append((
                round(lat - t * dlat * n_points + RNG.uniform(-0.005, 0.005), 5),
                round(lon - t * dlon * n_points + RNG.uniform(-0.005, 0.005), 5),
            ))

    else:
        # Generic slight drift
        for i in range(n_points):
            t = (n_points - 1 - i) / (n_points - 1)
            points.append((
                round(lat + t * RNG.uniform(-0.15, 0.15), 5),
                round(lon + t * RNG.uniform(-0.15, 0.15), 5),
            ))

    # Always end at the current vessel position
    points.append((lat, lon))
    return points


def get_restricted_zones() -> List[Dict]:
    """Return the list of restricted zone definitions."""
    return RESTRICTED_ZONES


def get_alerts() -> List[Dict]:
    """Return the list of demo alert messages."""
    return ALERTS


def get_risk_breakdown(vessel_id: str) -> List[Dict]:
    """Return risk factor breakdown for a vessel (demo data)."""
    return RISK_BREAKDOWNS.get(vessel_id, [])


def compute_dashboard_stats(df: pd.DataFrame) -> Dict:
    """Compute aggregate statistics shown in the top metrics row."""
    high   = int((df["risk_level"] == "HIGH").sum())
    medium = int((df["risk_level"] == "MEDIUM").sum())
    low    = int((df["risk_level"] == "LOW").sum())
    alerts = len([a for a in ALERTS if a["level"] == "HIGH"])
    return {
        "total_vessels":   len(df),
        "high_risk":       high,
        "medium_risk":     medium,
        "low_risk":        low,
        "active_alerts":   alerts,
    }
