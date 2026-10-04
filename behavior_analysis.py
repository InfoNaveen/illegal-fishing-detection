"""
behavior_analysis.py
Trajectory & vessel-behaviour analysis (Round 3 — Milestone M4).

Operates on CHRONOLOGICAL vessel trajectories and derives explainable
behavioural indicators that help answer *why* a vessel may be suspicious.

Important honesty note
----------------------
These are behavioural indicators / suspicious-activity signals. They do NOT,
on their own, prove illegal fishing. No positions are fabricated, no missing
AIS values are interpolated, and no machine learning is used for the
classification here — all thresholds are transparent and documented.

Input
-----
A per-vessel trajectory can be supplied as either:
  * a list of (lat, lon) tuples (simulated path), or
  * a pandas DataFrame slice with columns
    [latitude, longitude, speed, heading, timestamp] (historical AIS).

The DataFrame form unlocks the time-aware metrics (speed/heading change rates,
AIS gaps, duration). The tuple form still yields the geometric metrics.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Configurable thresholds
# ---------------------------------------------------------------------------

LOW_SPEED_THRESHOLD_KN: float = 1.0        # <= this counts as stationary/low-speed
SIGNIFICANT_GAP_MINUTES: float = 30.0      # AIS gaps >= this are "significant"
HIGH_TURNING_DEG: float = 45.0             # mean heading change above this = high turning
LOITERING_FLAG: float = 0.6                # loitering_score above this flags LOITERING
SPEED_ANOMALY_STD: float = 6.0             # speed_std above this flags SPEED_ANOMALY
EARTH_RADIUS_KM: float = 6371.0088

# Numeric behavioural features exposed to the ML feature matrix (M4).
# Only plain floats — never lists/strings/timestamps.
BEHAVIOR_FEATURE_COLUMNS: List[str] = [
    "b_distance_km",
    "b_avg_speed",
    "b_max_speed",
    "b_speed_std",
    "b_mean_heading_change",
    "b_max_heading_change",
    "b_stationary_ratio",
    "b_loitering_score",
    "b_max_gap_minutes",
    "b_significant_gap_count",
    "b_duration_minutes",
    "b_position_count",
]


# ---------------------------------------------------------------------------
# Geographic distance (Haversine)
# ---------------------------------------------------------------------------

def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Great-circle distance between two lat/lon points in kilometres
    (Haversine formula). Never uses raw degree subtraction for distance.
    """
    rlat1, rlat2 = math.radians(lat1), math.radians(lat2)
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (math.sin(dlat / 2) ** 2
         + math.cos(rlat1) * math.cos(rlat2) * math.sin(dlon / 2) ** 2)
    return 2 * EARTH_RADIUS_KM * math.asin(min(1.0, math.sqrt(a)))


def _angular_diff(a: float, b: float) -> float:
    """Smallest signed difference between two bearings, normalised to [-180, 180]."""
    return (b - a + 180.0) % 360.0 - 180.0


# ---------------------------------------------------------------------------
# Trajectory → arrays helper
# ---------------------------------------------------------------------------

def _extract_arrays(trajectory: Union[pd.DataFrame, Sequence[Tuple[float, float]]]) -> Dict:
    """
    Normalise a trajectory input into numpy arrays.

    Returns dict with: lat, lon, speed (or None), heading (or None),
    timestamp (pandas DatetimeIndex or None), n.
    Rows with invalid coordinates are skipped. Chronological order is assumed
    (callers pass already-ordered data); if a timestamp column exists it is used
    to sort defensively.
    """
    if isinstance(trajectory, pd.DataFrame):
        df = trajectory
        if df.empty:
            return {"lat": np.array([]), "lon": np.array([]),
                    "speed": None, "heading": None, "timestamp": None, "n": 0}
        if "timestamp" in df.columns:
            ts = pd.to_datetime(df["timestamp"], errors="coerce")
            df = df.assign(_ts=ts).sort_values("_ts")
        else:
            df = df.copy()
            df["_ts"] = pd.NaT
        lat = pd.to_numeric(df["latitude"], errors="coerce").to_numpy()
        lon = pd.to_numeric(df["longitude"], errors="coerce").to_numpy()
        valid = ~(np.isnan(lat) | np.isnan(lon))
        lat, lon = lat[valid], lon[valid]
        speed = (pd.to_numeric(df["speed"], errors="coerce").to_numpy()[valid]
                 if "speed" in df.columns else None)
        heading = (pd.to_numeric(df["heading"], errors="coerce").to_numpy()[valid]
                   if "heading" in df.columns else None)
        ts_arr = df["_ts"].to_numpy()[valid] if df["_ts"].notna().any() else None
        return {"lat": lat, "lon": lon, "speed": speed,
                "heading": heading, "timestamp": ts_arr, "n": int(len(lat))}

    # Sequence of (lat, lon) tuples
    pts = [p for p in trajectory
           if p is not None and not (math.isnan(p[0]) or math.isnan(p[1]))]
    if not pts:
        return {"lat": np.array([]), "lon": np.array([]),
                "speed": None, "heading": None, "timestamp": None, "n": 0}
    arr = np.asarray(pts, dtype=float)
    return {"lat": arr[:, 0], "lon": arr[:, 1], "speed": None,
            "heading": None, "timestamp": None, "n": int(len(arr))}


# ---------------------------------------------------------------------------
# Core behaviour metrics (per vessel)
# ---------------------------------------------------------------------------

def analyze_trajectory(
    trajectory: Union[pd.DataFrame, Sequence[Tuple[float, float]]],
    low_speed_threshold: float = LOW_SPEED_THRESHOLD_KN,
    significant_gap_minutes: float = SIGNIFICANT_GAP_MINUTES,
) -> Dict:
    """
    Compute behavioural metrics for one vessel's chronological trajectory.

    Robust to: single-position vessels, empty trajectories, missing speed /
    heading / timestamps, zero-duration tracks, and duplicate timestamps —
    never raises on those; it returns neutral (0.0) values instead.

    Returns a dict containing all BEHAVIOR_FEATURE_COLUMNS (floats) plus the
    richer named metrics used by the dashboard, and the behaviour
    classification (behavior_labels + behavior_summary).
    """
    data = _extract_arrays(trajectory)
    n = data["n"]
    lat, lon = data["lat"], data["lon"]
    speed, heading, ts = data["speed"], data["heading"], data["timestamp"]

    # ── Defaults (safe for empty / single-position) ──────────────────────
    distance_km = 0.0
    avg_speed = max_speed = speed_std = 0.0
    speed_change_rate = 0.0
    mean_heading_change = max_heading_change = heading_change_std = 0.0
    turning_rate = 0.0
    stationary_ratio = 0.0
    loitering_score = 0.0
    max_gap_minutes = mean_gap_minutes = 0.0
    significant_gap_count = 0
    duration_minutes = 0.0

    # ── Distance (Haversine over consecutive points) ─────────────────────
    if n >= 2:
        seg = np.array([
            haversine_km(lat[i], lon[i], lat[i + 1], lon[i + 1])
            for i in range(n - 1)
        ])
        distance_km = float(seg.sum())

    # ── Speed statistics (from AIS SOG where available) ──────────────────
    if speed is not None:
        sp = speed[~np.isnan(speed)]
        if sp.size > 0:
            avg_speed = float(np.mean(sp))
            max_speed = float(np.max(sp))
            speed_std = float(np.std(sp)) if sp.size > 1 else 0.0

    # ── Heading change / turning ─────────────────────────────────────────
    if heading is not None:
        hd = heading
        changes = []
        for i in range(len(hd) - 1):
            if not (np.isnan(hd[i]) or np.isnan(hd[i + 1])):
                changes.append(abs(_angular_diff(hd[i], hd[i + 1])))
        if changes:
            changes_arr = np.array(changes)
            mean_heading_change = float(np.mean(changes_arr))
            max_heading_change = float(np.max(changes_arr))
            heading_change_std = float(np.std(changes_arr)) if changes_arr.size > 1 else 0.0

    # ── Timestamp-derived metrics (gaps, duration, rates) ────────────────
    if ts is not None and len(ts) >= 2:
        ts_series = pd.to_datetime(pd.Series(ts))
        deltas_min = ts_series.diff().dropna().dt.total_seconds().to_numpy() / 60.0
        deltas_min = deltas_min[deltas_min >= 0]  # guard against ordering noise
        if deltas_min.size > 0:
            max_gap_minutes = float(np.max(deltas_min))
            mean_gap_minutes = float(np.mean(deltas_min))
            significant_gap_count = int(np.sum(deltas_min >= significant_gap_minutes))
            duration_minutes = float(np.sum(deltas_min))
            # speed change rate: |Δspeed| per minute, only on positive intervals
            if speed is not None and len(speed) == len(ts):
                dspeed = np.abs(np.diff(speed))
                with np.errstate(divide="ignore", invalid="ignore"):
                    rates = dspeed / np.where(deltas_min > 0, deltas_min, np.nan)
                rates = rates[np.isfinite(rates)]
                if rates.size > 0:
                    speed_change_rate = float(np.mean(rates))
            # turning rate: mean heading change per minute
            if heading is not None and mean_heading_change > 0 and mean_gap_minutes > 0:
                turning_rate = float(mean_heading_change / mean_gap_minutes)

    # ── Stationary / low-speed ratio ─────────────────────────────────────
    if speed is not None:
        sp = speed[~np.isnan(speed)]
        if sp.size > 0:
            stationary_ratio = float(np.mean(sp <= low_speed_threshold))

    # ── Loitering score (explainable, [0,1]) ─────────────────────────────
    # Logic: a vessel that covers little NET ground relative to how far it
    # travelled, while spending time slow/stationary, is "loitering".
    #   spatial_spread  = diagonal of lat/lon bounding box (km, Haversine)
    #   containment     = 1 - min(spread / (distance+eps), 1)   # 1 = stayed put
    #   loitering_score = 0.6*containment + 0.4*stationary_ratio
    if n >= 2 and distance_km > 1e-6:
        bbox_diag_km = haversine_km(float(lat.min()), float(lon.min()),
                                    float(lat.max()), float(lon.max()))
        containment = 1.0 - min(bbox_diag_km / (distance_km + 1e-6), 1.0)
        loitering_score = float(min(1.0, 0.6 * containment + 0.4 * stationary_ratio))
    else:
        # No/again single movement: loitering driven purely by stationarity.
        loitering_score = float(min(1.0, 0.4 * stationary_ratio))

    # ── Classification (transparent thresholds, multi-label) ─────────────
    labels, summary = _classify(
        avg_speed=avg_speed, max_speed=max_speed, speed_std=speed_std,
        mean_heading_change=mean_heading_change, stationary_ratio=stationary_ratio,
        loitering_score=loitering_score, max_gap_minutes=max_gap_minutes,
        significant_gap_count=significant_gap_count, position_count=n,
    )

    return {
        # Rich named metrics (dashboard)
        "distance_travelled_km":    round(distance_km, 3),
        "avg_speed":                round(avg_speed, 2),
        "max_speed":                round(max_speed, 2),
        "speed_std":                round(speed_std, 3),
        "speed_change_rate":        round(speed_change_rate, 4),
        "mean_heading_change":      round(mean_heading_change, 2),
        "max_heading_change":       round(max_heading_change, 2),
        "heading_change_std":       round(heading_change_std, 3),
        "turning_rate":             round(turning_rate, 4),
        "stationary_ratio":         round(stationary_ratio, 4),
        "loitering_score":          round(loitering_score, 4),
        "max_ais_gap_minutes":      round(max_gap_minutes, 2),
        "mean_ais_gap_minutes":     round(mean_gap_minutes, 2),
        "significant_gap_count":    int(significant_gap_count),
        "trajectory_duration_minutes": round(duration_minutes, 2),
        "position_count":           int(n),
        # Classification
        "behavior_labels":          labels,
        "behavior_summary":         summary,
    }


def _classify(avg_speed, max_speed, speed_std, mean_heading_change,
              stationary_ratio, loitering_score, max_gap_minutes,
              significant_gap_count, position_count) -> Tuple[List[str], str]:
    """
    Transparent, non-ML, multi-label behaviour classification.

    A vessel is not forced into a suspicious category — a clean track returns
    just ["NORMAL_TRANSIT"].
    """
    labels: List[str] = []
    parts: List[str] = []

    if position_count <= 1:
        return (["INSUFFICIENT_DATA"], "Only one position available — insufficient for behaviour analysis.")

    if loitering_score >= LOITERING_FLAG:
        labels.append("LOITERING")
        parts.append("prolonged loitering")
    if stationary_ratio >= 0.5:
        labels.append("SLOW_MOVEMENT")
        parts.append("sustained low-speed/stationary movement")
    if mean_heading_change >= HIGH_TURNING_DEG:
        labels.append("HIGH_TURNING")
        parts.append("frequent sharp course changes")
    if significant_gap_count > 0 or max_gap_minutes >= SIGNIFICANT_GAP_MINUTES:
        labels.append("AIS_GAP")
        parts.append(f"significant AIS gap ({int(max_gap_minutes)} min)")
    if speed_std >= SPEED_ANOMALY_STD:
        labels.append("SPEED_ANOMALY")
        parts.append("highly variable speed")

    if len(labels) >= 2:
        labels.append("MIXED_SUSPICIOUS")

    if not labels:
        labels = ["NORMAL_TRANSIT"]
        summary = "Consistent transit movement — no notable suspicious indicators."
    else:
        summary = _sentence_case(", ".join(parts)) + "."

    return labels, summary


def _sentence_case(text: str) -> str:
    return text[:1].upper() + text[1:] if text else text


# ---------------------------------------------------------------------------
# Batch analysis
# ---------------------------------------------------------------------------

def analyze_all(
    trajectories: Dict[str, Union[pd.DataFrame, Sequence[Tuple[float, float]]]],
) -> Dict[str, Dict]:
    """
    Run analyze_trajectory() for every vessel.

    Parameters
    ----------
    trajectories : {vessel_id: trajectory} where trajectory is either a
                   DataFrame slice (historical AIS) or a list of (lat, lon).

    Returns
    -------
    {vessel_id: behaviour_dict}
    """
    return {vid: analyze_trajectory(traj) for vid, traj in trajectories.items()}


def behavior_feature_row(behavior: Dict) -> Dict[str, float]:
    """
    Project a behaviour dict down to the numeric-only feature subset that is
    safe to feed into the ML feature matrix (no lists/strings/timestamps).
    """
    return {
        "b_distance_km":            float(behavior.get("distance_travelled_km", 0.0)),
        "b_avg_speed":              float(behavior.get("avg_speed", 0.0)),
        "b_max_speed":              float(behavior.get("max_speed", 0.0)),
        "b_speed_std":              float(behavior.get("speed_std", 0.0)),
        "b_mean_heading_change":    float(behavior.get("mean_heading_change", 0.0)),
        "b_max_heading_change":     float(behavior.get("max_heading_change", 0.0)),
        "b_stationary_ratio":       float(behavior.get("stationary_ratio", 0.0)),
        "b_loitering_score":        float(behavior.get("loitering_score", 0.0)),
        "b_max_gap_minutes":        float(behavior.get("max_ais_gap_minutes", 0.0)),
        "b_significant_gap_count":  float(behavior.get("significant_gap_count", 0)),
        "b_duration_minutes":       float(behavior.get("trajectory_duration_minutes", 0.0)),
        "b_position_count":         float(behavior.get("position_count", 0)),
    }
