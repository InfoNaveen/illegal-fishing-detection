"""
live_risk_cache.py
Incremental risk computation for the Live AIS and Real-Time Simulation modes.

Extracted from app.py so this logic can be imported and tested independently,
without triggering app.py's top-level Streamlit calls (st.set_page_config etc.).

Intentionally has ZERO Streamlit imports.
"""

from __future__ import annotations

from typing import Dict, Optional

import pandas as pd


def compute_live_risk(
    state_df: pd.DataFrame,
    recent_tracks: Dict[str, pd.DataFrame],
    zones,
) -> Dict[str, Dict]:
    """
    Compute unified risk for a set of currently-active vessels.

    Parameters
    ----------
    state_df      : one-row-per-vessel current positions (vessel_id, latitude,
                    longitude, speed, heading, behavior columns required).
    recent_tracks : {vessel_id: recent-trajectory DataFrame with latitude,
                    longitude, speed, heading, timestamp columns}.
    zones         : list of zone dicts ({name, coords, color, center}).

    Returns
    -------
    {vessel_id: unified_risk_result}  — same schema as run_risk_engine output.

    Notes
    -----
    - Used by the Real-Time Simulation path (called on every tick) and the
      Live AIS path (called on every dashboard refresh).
    - Both call sites pass dirty_vessels=None, which always rescores all
      active vessels.  The Live AIS path could pass a dirty set for incremental
      updates, but correctness is preferred over micro-optimisation here.
    - Temporal scoring is included when trajectory history is long enough;
      vessels with insufficient history return temporal_anomaly_score=0.
    """
    if state_df is None or state_df.empty:
        return {}

    # Import here (not at module level) to avoid circular imports and to keep
    # this module free of Streamlit dependencies.
    from behavior_analysis import analyze_all
    from feature_engineering import build_feature_matrix
    from geofencing import run_geofencing
    from anomaly_detector import run_anomaly_detection_with_status
    from temporal_model_manager import run_temporal_detection
    from risk_engine import run_risk_engine

    # Behaviour analysis works directly on the time-aware DataFrame tracks.
    behavior_map = analyze_all(recent_tracks)

    # Feature extraction expects (lat, lon) tuple lists per vessel.
    tuple_tracks: Dict[str, list] = {}
    for _vid, _trk in recent_tracks.items():
        try:
            tuple_tracks[_vid] = list(
                zip(_trk["latitude"].tolist(), _trk["longitude"].tolist())
            )
        except Exception:
            tuple_tracks[_vid] = []

    geo_list = run_geofencing(state_df, zones)
    geo_map = {g["vessel_id"]: g for g in geo_list}

    feature_df = build_feature_matrix(
        state_df, tuple_tracks, geo_map, behavior_map=behavior_map
    )
    anomaly_map, _st, _m = run_anomaly_detection_with_status(feature_df)
    temporal_map, _ts, _tm = run_temporal_detection(recent_tracks)

    return run_risk_engine(
        state_df["vessel_id"].tolist(),
        geo_map,
        feature_df,
        anomaly_map,
        behavior_map=behavior_map,
        temporal_map=temporal_map,
    )
