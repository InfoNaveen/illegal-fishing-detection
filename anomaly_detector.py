"""
anomaly_detector.py
Isolation Forest anomaly detection for the Illegal Fishing Detection System.

Fits on the vessel feature matrix produced by feature_engineering.py.
Returns a per-vessel anomaly score normalised to [0, 1] where
1.0 = most anomalous.

IsolationForest.decision_function() returns a raw score where
  lower (more negative) = more anomalous.
We invert and normalise to produce an intuitive 0→1 scale.
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import MinMaxScaler

from feature_engineering import FEATURE_COLUMNS

# ---------------------------------------------------------------------------
# Model configuration
# ---------------------------------------------------------------------------

IF_PARAMS = {
    "n_estimators":  200,
    "max_samples":   "auto",
    "contamination": 0.20,   # ~20 % of vessels expected to be anomalous
    "random_state":  42,
    "n_jobs":        -1,
}


# ---------------------------------------------------------------------------
# Training + scoring
# ---------------------------------------------------------------------------

def _scale_features(feature_df: pd.DataFrame) -> Tuple[np.ndarray, MinMaxScaler]:
    """StandardScaler-style normalisation → returns scaled array + fitted scaler."""
    scaler = MinMaxScaler()
    X = scaler.fit_transform(feature_df[FEATURE_COLUMNS].values.astype(float))
    return X, scaler


def train_and_score(feature_df: pd.DataFrame) -> Dict[str, float]:
    """
    Fit an IsolationForest on the feature matrix and return a dict
    mapping vessel_id → anomaly_score in [0, 1].

    Score interpretation
    --------------------
    0.0 – 0.35 : normal behaviour
    0.35 – 0.65 : mildly anomalous
    0.65 – 1.0  : strongly anomalous
    """
    X, _ = _scale_features(feature_df)

    model = IsolationForest(**IF_PARAMS)
    model.fit(X)

    # decision_function: higher = more normal, lower = more anomalous
    raw_scores = model.decision_function(X)          # shape (n_vessels,)

    # Invert: anomaly score = high when raw is low
    inverted = -raw_scores

    # Normalise to [0, 1]
    s_min, s_max = inverted.min(), inverted.max()
    if s_max > s_min:
        normalised = (inverted - s_min) / (s_max - s_min)
    else:
        normalised = np.zeros_like(inverted)

    vessel_ids = feature_df.index.tolist()
    return {vid: round(float(score), 4)
            for vid, score in zip(vessel_ids, normalised)}


def get_anomaly_labels(anomaly_scores: Dict[str, float],
                       threshold: float = 0.55) -> Dict[str, bool]:
    """
    Return {vessel_id: is_anomalous} based on a score threshold.
    Default threshold 0.55 flags the clearly anomalous vessels.
    """
    return {vid: score >= threshold
            for vid, score in anomaly_scores.items()}


def run_anomaly_detection(feature_df: pd.DataFrame) -> Dict[str, Dict]:
    """
    Full anomaly detection pass.

    Returns
    -------
    {vessel_id: {"anomaly_score": float, "is_anomalous": bool}}
    """
    scores = train_and_score(feature_df)
    labels = get_anomaly_labels(scores)
    return {
        vid: {
            "anomaly_score": scores[vid],
            "is_anomalous":  labels[vid],
        }
        for vid in scores
    }
