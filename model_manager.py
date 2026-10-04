"""
model_manager.py
Persistent Isolation Forest model lifecycle (Round 3 — Milestone M5).

Goal
----
Avoid re-fitting the Isolation Forest on every pipeline run. Instead:

    feature matrix
          |
    compatible artifact exists?  ──no──►  train + save
          |yes
      load + reuse
          |
       predict  ──►  anomaly scores

The artifact bundles the fitted IsolationForest, the fitted MinMaxScaler
(scaling is part of the scoring contract), and metadata used to decide whether
a saved model is compatible with the current feature pipeline.

Honesty note
------------
This is unsupervised anomaly detection. No labelled illegal-fishing ground
truth is used; the artifact records configuration/schema only, not accuracy
metrics.

Serialization: joblib. Artifact and model directory are git-ignored and created
at runtime. The model is never stored in SQLite and never committed.
"""

from __future__ import annotations

import os
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import MinMaxScaler

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DEFAULT_MODEL_DIR: str = "models"
DEFAULT_MODEL_PATH: str = os.path.join(DEFAULT_MODEL_DIR, "isolation_forest.joblib")

MODEL_VERSION: str = "m5"

# Isolation Forest configuration — must match the historical anomaly_detector
# values so model behaviour is unchanged.
IF_PARAMS: Dict = {
    "n_estimators":  200,
    "max_samples":   "auto",
    "contamination": 0.20,
    "random_state":  42,
    "n_jobs":        -1,
}


# ---------------------------------------------------------------------------
# Training / serialization
# ---------------------------------------------------------------------------

def train_model(feature_df: pd.DataFrame, feature_columns: List[str]) -> Dict:
    """
    Fit a MinMaxScaler + IsolationForest on the given feature matrix.

    Returns an artifact dict (NOT yet saved) containing the model, scaler,
    and metadata. Scaling is included because scoring depends on it.
    """
    cols = list(feature_columns)
    X = MinMaxScaler().fit_transform(feature_df[cols].values.astype(float))
    scaler = MinMaxScaler().fit(feature_df[cols].values.astype(float))

    model = IsolationForest(**IF_PARAMS)
    model.fit(X)

    return {
        "model":            model,
        "scaler":           scaler,
        "feature_columns":  cols,
        "feature_count":    len(cols),
        "random_state":     IF_PARAMS["random_state"],
        "contamination":    IF_PARAMS["contamination"],
        "n_estimators":     IF_PARAMS["n_estimators"],
        "trained_at":       datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "model_version":    MODEL_VERSION,
    }


def save_model(artifact: Dict, model_path: str = DEFAULT_MODEL_PATH) -> None:
    """Serialize an artifact to disk, creating the directory if needed."""
    parent = os.path.dirname(os.path.abspath(model_path))
    if parent and not os.path.exists(parent):
        os.makedirs(parent, exist_ok=True)
    joblib.dump(artifact, model_path)


def load_model(model_path: str = DEFAULT_MODEL_PATH) -> Optional[Dict]:
    """
    Deserialize an artifact. Returns None if the file is missing or corrupted
    (never raises — callers safely retrain on None).
    """
    if not os.path.exists(model_path):
        return None
    try:
        artifact = joblib.load(model_path)
    except Exception:
        return None
    if not isinstance(artifact, dict) or "model" not in artifact:
        return None
    return artifact


def model_exists(model_path: str = DEFAULT_MODEL_PATH) -> bool:
    """True if an artifact file is present on disk."""
    return os.path.exists(model_path)


# ---------------------------------------------------------------------------
# Compatibility
# ---------------------------------------------------------------------------

def model_is_compatible(artifact: Optional[Dict],
                        feature_columns: List[str]) -> Tuple[bool, str]:
    """
    Decide whether a loaded artifact is usable with the current pipeline.

    Checks (in order): artifact present & deserialized, metadata present,
    model object can predict, feature count matches, feature NAMES match,
    feature ORDER matches, model_version matches.

    Returns (is_compatible, reason). The reason explains an incompatibility
    (used for retrain status / tests).
    """
    if artifact is None:
        return False, "no artifact (missing or corrupted)"

    model = artifact.get("model")
    if model is None or not hasattr(model, "decision_function"):
        return False, "artifact has no usable model object"

    if "feature_columns" not in artifact or "feature_count" not in artifact:
        return False, "artifact metadata incomplete"

    saved_cols = list(artifact["feature_columns"])
    cur_cols = list(feature_columns)

    if artifact.get("feature_count") != len(cur_cols):
        return False, (f"feature count mismatch "
                       f"(saved {artifact.get('feature_count')}, now {len(cur_cols)})")

    if set(saved_cols) != set(cur_cols):
        return False, "feature column names changed"

    if saved_cols != cur_cols:
        return False, "feature column order changed"

    if artifact.get("model_version") != MODEL_VERSION:
        return False, (f"model version mismatch "
                       f"(saved {artifact.get('model_version')}, now {MODEL_VERSION})")

    return True, "compatible"


def get_model_metadata(artifact: Dict) -> Dict:
    """Return the metadata subset of an artifact (no model/scaler objects)."""
    if not artifact:
        return {}
    return {k: artifact[k] for k in (
        "feature_columns", "feature_count", "random_state", "contamination",
        "n_estimators", "trained_at", "model_version") if k in artifact}


# ---------------------------------------------------------------------------
# Prediction + high-level lifecycle
# ---------------------------------------------------------------------------

def predict_scores(artifact: Dict,
                   feature_df: pd.DataFrame,
                   feature_columns: List[str]) -> Dict[str, float]:
    """
    Produce per-vessel normalised anomaly scores in [0, 1] (1 = most anomalous)
    using a loaded/ trained artifact.

    Preserves the exact scoring contract of the original anomaly_detector:
      scale → decision_function → invert → min-max normalise over the batch.
    """
    cols = list(feature_columns)
    scaler = artifact.get("scaler")
    raw_X = feature_df[cols].values.astype(float)
    X = scaler.transform(raw_X) if scaler is not None else raw_X

    raw_scores = artifact["model"].decision_function(X)   # higher = more normal
    inverted = -raw_scores                                 # higher = more anomalous

    s_min, s_max = inverted.min(), inverted.max()
    if s_max > s_min:
        normalised = (inverted - s_min) / (s_max - s_min)
    else:
        normalised = np.zeros_like(inverted)

    vessel_ids = feature_df.index.tolist()
    return {vid: round(float(score), 4)
            for vid, score in zip(vessel_ids, normalised)}


def load_or_train(feature_df: pd.DataFrame,
                  feature_columns: List[str],
                  model_path: str = DEFAULT_MODEL_PATH) -> Tuple[Dict, str]:
    """
    Central M5 lifecycle entry point.

    Returns (artifact, status) where status is one of:
      "loaded"    — a compatible artifact was reused (NOT retrained)
      "trained"   — no artifact existed, a new one was trained + saved
      "retrained" — an incompatible/corrupted artifact was replaced

    Never raises on a missing/corrupt artifact — it safely retrains.
    """
    existing = load_model(model_path)
    compatible, _reason = model_is_compatible(existing, feature_columns)

    if compatible:
        return existing, "loaded"

    # Train fresh. Distinguish "trained" (nothing was there / unreadable) from
    # "retrained" (a file existed but was incompatible).
    status = "retrained" if model_exists(model_path) else "trained"
    artifact = train_model(feature_df, feature_columns)
    save_model(artifact, model_path)
    return artifact, status
