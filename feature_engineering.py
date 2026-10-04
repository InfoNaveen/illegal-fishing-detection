"""
feature_engineering.py
Extracts numerical behavioural features from vessel positions and trajectories.

All inputs come from data_generator.py (vessel base data + trajectories).
No external dependencies beyond numpy.

Feature vector per vessel (9 features):
  0  speed                  — current reported speed (knots)
  1  speed_deviation        — |speed - normal_fishing_speed| normalised
  2  heading_variance       — variance of heading changes along trajectory
  3  trajectory_displacement — net displacement / total path length  (0=loitering, 1=straight)
  4  loitering_score        — 0-1: tight circular movement indicator
  5  path_length_deg        — total arc-length of trajectory (degrees)
  6  bearing_change_rate    — mean absolute bearing change per step
  7  zone_proximity_norm    — normalised proximity to nearest restricted zone (1=inside, 0=far)
  8  erratic_score          — std-dev of step lengths (irregular movement)
"""

import math
import numpy as np
from typing import Dict, List, Tuple

# ---------------------------------------------------------------------------
# Domain constants
# ---------------------------------------------------------------------------

# Typical legitimate fishing vessel speeds (knots)
NORMAL_FISHING_SPEED_MIN: float = 3.0
NORMAL_FISHING_SPEED_MAX: float = 9.0
NORMAL_FISHING_SPEED_MID: float = (NORMAL_FISHING_SPEED_MIN + NORMAL_FISHING_SPEED_MAX) / 2.0

# Distance threshold used for proximity normalisation (same as geofencing module)
PROXIMITY_MAX_DEG: float = 0.30   # beyond this → 0 proximity contribution


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def _bearing(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Approximate bearing (degrees 0-360) between two lat/lon points."""
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    angle = math.degrees(math.atan2(dlon, dlat)) % 360
    return angle


def _step_length(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Euclidean distance in degrees between two lat/lon points."""
    return math.hypot(lat2 - lat1, lon2 - lon1)


def _angular_diff(a: float, b: float) -> float:
    """Smallest signed difference between two bearings (degrees)."""
    d = (b - a + 180) % 360 - 180
    return d


# ---------------------------------------------------------------------------
# Per-vessel feature computation
# ---------------------------------------------------------------------------

def extract_features(vessel_id: str,
                     lat: float,
                     lon: float,
                     speed: float,
                     heading: float,
                     trajectory: List[Tuple[float, float]],
                     geo_result: Dict) -> Dict:
    """
    Compute all 9 numerical features for one vessel.

    Parameters
    ----------
    vessel_id   : str
    lat, lon    : current position
    speed       : current speed in knots
    heading     : current heading in degrees
    trajectory  : list of (lat, lon) historical waypoints including current
    geo_result  : dict returned by geofencing.check_vessel_zones()

    Returns
    -------
    dict with keys: vessel_id + all 9 feature names (float values)
    """
    pts = trajectory  # list of (lat, lon)

    # ── Feature 0: speed ───────────────────────────────────────────────────
    f_speed = float(speed)

    # ── Feature 1: speed_deviation ────────────────────────────────────────
    if speed < NORMAL_FISHING_SPEED_MIN:
        # Very slow — loitering-like
        dev = (NORMAL_FISHING_SPEED_MIN - speed) / NORMAL_FISHING_SPEED_MIN
    elif speed > NORMAL_FISHING_SPEED_MAX:
        # Very fast — transit or fleeing
        dev = (speed - NORMAL_FISHING_SPEED_MAX) / NORMAL_FISHING_SPEED_MAX
    else:
        dev = 0.0
    f_speed_deviation = min(dev, 1.0)

    # ── Features derived from trajectory ──────────────────────────────────
    if len(pts) < 2:
        # Degenerate: no movement data
        f_heading_variance    = 0.0
        f_displacement_ratio  = 1.0   # assume straight
        f_loitering_score     = 0.0
        f_path_length         = 0.0
        f_bearing_change_rate = 0.0
        f_erratic_score       = 0.0
    else:
        # Step lengths and bearings
        step_lengths: List[float] = []
        bearings: List[float]     = []
        for i in range(len(pts) - 1):
            la1, lo1 = pts[i]
            la2, lo2 = pts[i + 1]
            step_lengths.append(_step_length(la1, lo1, la2, lo2))
            bearings.append(_bearing(la1, lo1, la2, lo2))

        path_length = sum(step_lengths)
        f_path_length = path_length

        # Net displacement: start → end of trajectory
        net_dist = _step_length(pts[0][0], pts[0][1], pts[-1][0], pts[-1][1])

        # Displacement ratio: 0 = loitering (came back to start), 1 = straight transit
        if path_length > 1e-6:
            f_displacement_ratio = min(net_dist / path_length, 1.0)
        else:
            f_displacement_ratio = 1.0

        # Bearing changes between consecutive steps
        if len(bearings) >= 2:
            bearing_changes = [abs(_angular_diff(bearings[i], bearings[i + 1]))
                               for i in range(len(bearings) - 1)]
            f_heading_variance    = float(np.var(bearing_changes))
            f_bearing_change_rate = float(np.mean(bearing_changes))
        else:
            f_heading_variance    = 0.0
            f_bearing_change_rate = 0.0

        # Loitering: low displacement ratio + slow speed → high loitering score
        slow_factor = max(0.0, 1.0 - speed / NORMAL_FISHING_SPEED_MIN)
        f_loitering_score = min(1.0, (1.0 - f_displacement_ratio) * 0.7
                                     + slow_factor * 0.3)

        # Erratic score: std-dev of step lengths normalised by mean
        if len(step_lengths) > 1 and np.mean(step_lengths) > 1e-8:
            f_erratic_score = min(float(np.std(step_lengths) / np.mean(step_lengths)), 1.0)
        else:
            f_erratic_score = 0.0

    # ── Feature 7: zone_proximity_norm ────────────────────────────────────
    if geo_result["inside"]:
        f_zone_proximity = 1.0
    else:
        dist = geo_result.get("min_dist_deg", PROXIMITY_MAX_DEG)
        f_zone_proximity = max(0.0, 1.0 - dist / PROXIMITY_MAX_DEG)

    return {
        "vessel_id":            vessel_id,
        "speed":                round(f_speed, 3),
        "speed_deviation":      round(f_speed_deviation, 4),
        "heading_variance":     round(f_heading_variance, 4),
        "displacement_ratio":   round(f_displacement_ratio, 4),
        "loitering_score":      round(f_loitering_score, 4),
        "path_length_deg":      round(f_path_length, 5),
        "bearing_change_rate":  round(f_bearing_change_rate, 4),
        "zone_proximity_norm":  round(f_zone_proximity, 4),
        "erratic_score":        round(f_erratic_score, 4),
    }


# ---------------------------------------------------------------------------
# Batch extraction
# ---------------------------------------------------------------------------

# Column order used by the anomaly detector — must stay stable
FEATURE_COLUMNS: List[str] = [
    "speed",
    "speed_deviation",
    "heading_variance",
    "displacement_ratio",
    "loitering_score",
    "path_length_deg",
    "bearing_change_rate",
    "zone_proximity_norm",
    "erratic_score",
]


def build_feature_matrix(vessel_df,
                         trajectories: Dict[str, List[Tuple[float, float]]],
                         geo_results: Dict[str, Dict],
                         behavior_map: Dict[str, Dict] = None) -> "pd.DataFrame":
    """
    Build a feature DataFrame for all vessels.

    Parameters
    ----------
    vessel_df    : DataFrame with columns vessel_id, latitude, longitude,
                   speed, heading
    trajectories : {vessel_id: [(lat, lon), ...]}
    geo_results  : {vessel_id: geo_result_dict}
    behavior_map : optional {vessel_id: behaviour_dict} from behavior_analysis.
                   When provided (M4), the numeric behavioural feature columns
                   (BEHAVIOR_FEATURE_COLUMNS) are appended to each row and to
                   the returned column set. When None, the original 9-column
                   contract is preserved exactly (back-compatible).

    Returns
    -------
    DataFrame indexed by vessel_id. Columns are FEATURE_COLUMNS, plus the
    numeric behaviour columns when behavior_map is supplied.
    """
    import pandas as pd

    use_behavior = behavior_map is not None
    if use_behavior:
        from behavior_analysis import BEHAVIOR_FEATURE_COLUMNS, behavior_feature_row

    rows = []
    for _, row in vessel_df.iterrows():
        vid = row["vessel_id"]
        traj = trajectories.get(vid, [(row["latitude"], row["longitude"])])
        geo  = geo_results.get(vid, {
            "inside": False, "near": False,
            "min_dist_deg": 0.30, "zone_status": "Open Waters",
        })
        feat = extract_features(
            vessel_id=vid,
            lat=row["latitude"],
            lon=row["longitude"],
            speed=row["speed"],
            heading=row["heading"],
            trajectory=traj,
            geo_result=geo,
        )
        if use_behavior:
            feat.update(behavior_feature_row(behavior_map.get(vid, {})))
        rows.append(feat)

    df = pd.DataFrame(rows).set_index("vessel_id")
    columns = list(FEATURE_COLUMNS)
    if use_behavior:
        columns = columns + list(BEHAVIOR_FEATURE_COLUMNS)
    return df[columns]
