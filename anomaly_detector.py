"""
anomaly_detector.py
Isolation Forest anomaly detection for the Illegal Fishing Detection System.

Fits (or, since M5, loads) on the vessel feature matrix produced by
feature_engineering.py. Returns a per-vessel anomaly score normalised to
[0, 1] where 1.0 = most anomalous.

IsolationForest.decision_function() returns a raw score where
  lower (more negative) = more anomalous.
We invert and normalise to produce an intuitive 0→1 scale.

M5 (persistent model lifecycle)
-------------------------------
Model fitting is delegated to model_manager, which trains once and reuses a
saved artifact on subsequent runs (with a feature-schema compatibility check).
The mathematical meaning of the anomaly score is unchanged: scale →
decision_function → invert → min-max normalise over the current batch.
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import MinMaxScaler

from feature_engineering import FEATURE_COLUMNS
import model_manager
from model_manager import IF_PARAMS, DEFAULT_MODEL_PATH


# ---------------------------------------------------------------------------
# Legacy fit-and-score (retained for backward compatibility / direct use)
# ---------------------------------------------------------------------------

def _scale_features(feature_df: pd.DataFrame,
                    columns=None) -> Tuple[np.ndarray, MinMaxScaler]:
    """Min-max scale the feature columns → returns scaled array + fitted scaler."""
    cols = list(columns) if columns is not None else list(FEATURE_COLUMNS)
    scaler = MinMaxScaler()
    X = scaler.fit_transform(feature_df[cols].values.astype(float))
    return X, scaler


def train_and_score(feature_df: pd.DataFrame) -> Dict[str, float]:
    """
    Fit an IsolationForest in-memory and return {vessel_id: anomaly_score}.

    Retained for backward compatibility. The persistent path is
    run_anomaly_detection(), which trains once and reuses the saved artifact.
    """
    cols = _resolve_columns(feature_df)
    X, _ = _scale_features(feature_df, cols)

    model = IsolationForest(**IF_PARAMS)
    model.fit(X)

    raw_scores = model.decision_function(X)
    inverted = -raw_scores
    s_min, s_max = inverted.min(), inverted.max()
    normalised = ((inverted - s_min) / (s_max - s_min)
                  if s_max > s_min else np.zeros_like(inverted))

    vessel_ids = feature_df.index.tolist()
    return {vid: round(float(score), 4)
            for vid, score in zip(vessel_ids, normalised)}


def get_anomaly_labels(anomaly_scores: Dict[str, float],
                       threshold: float = 0.55) -> Dict[str, bool]:
    """Return {vessel_id: is_anomalous} based on a score threshold (default 0.55)."""
    return {vid: score >= threshold for vid, score in anomaly_scores.items()}


# ---------------------------------------------------------------------------
# Feature-column resolution
# ---------------------------------------------------------------------------

def _resolve_columns(feature_df: pd.DataFrame):
    """
    Use the actual columns present on the feature matrix (so the M4 behaviour
    columns are included when present), falling back to the base FEATURE_COLUMNS.
    Never hard-codes the count.
    """
    cols = [c for c in feature_df.columns]
    return cols if cols else list(FEATURE_COLUMNS)


# ---------------------------------------------------------------------------
# M5 persistent lifecycle entry points
# ---------------------------------------------------------------------------

def run_anomaly_detection(feature_df: pd.DataFrame,
                          model_path: str = DEFAULT_MODEL_PATH,
                          use_persistent: bool = True) -> Dict[str, Dict]:
    """
    Full anomaly detection pass (persistent by default — M5).

    Returns {vessel_id: {"anomaly_score": float, "is_anomalous": bool}}.

    When use_persistent is False, falls back to the legacy in-memory fit
    (used by older callers/tests that do not want to touch disk).
    """
    result, _status, _meta = run_anomaly_detection_with_status(
        feature_df, model_path=model_path, use_persistent=use_persistent)
    return result


def run_anomaly_detection_with_status(
        feature_df: pd.DataFrame,
        model_path: str = DEFAULT_MODEL_PATH,
        use_persistent: bool = True) -> Tuple[Dict[str, Dict], str, Dict]:
    """
    Same as run_anomaly_detection but also returns the model status and
    metadata so the dashboard can report "loaded"/"trained"/"retrained".

    Returns (results, status, metadata).
    status is one of "loaded" | "trained" | "retrained" | "in-memory".
    """
    cols = _resolve_columns(feature_df)

    if use_persistent:
        artifact, status = model_manager.load_or_train(feature_df, cols, model_path)
        scores = model_manager.predict_scores(artifact, feature_df, cols)
        meta = model_manager.get_model_metadata(artifact)
    else:
        scores = train_and_score(feature_df)
        status = "in-memory"
        meta = {}

    labels = get_anomaly_labels(scores)
    results = {
        vid: {"anomaly_score": scores[vid], "is_anomalous": labels[vid]}
        for vid in scores
    }
    return results, status, meta
