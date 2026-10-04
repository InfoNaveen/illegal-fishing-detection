"""
temporal_model_manager.py
Persistent lifecycle for the temporal sequence anomaly model (M6).

Mirrors the M5 model_manager pattern: train once, reuse a saved artifact on
later runs, validate compatibility, and safely retrain on
missing/corrupt/incompatible artifacts. Serialized with joblib.

Artifact (git-ignored): models/temporal_model.joblib
"""

from __future__ import annotations

import os
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

import joblib
import pandas as pd

from temporal_model import (
    TemporalAnomalyModel, build_training_matrix, score_vessels,
    SEQUENCE_LENGTH, HIDDEN_SIZE, STEP_FEATURES, MODEL_VERSION,
)

DEFAULT_MODEL_DIR: str = "models"
DEFAULT_TEMPORAL_PATH: str = os.path.join(DEFAULT_MODEL_DIR, "temporal_model.joblib")


# ---------------------------------------------------------------------------
# Train / serialize
# ---------------------------------------------------------------------------

def train_model(tracks: Dict[str, pd.DataFrame],
                sequence_length: int = SEQUENCE_LENGTH,
                hidden_size: int = HIDDEN_SIZE) -> Dict:
    """
    Fit a TemporalAnomalyModel on sequence windows built from the tracks.
    Returns an artifact dict (model + metadata), not yet saved.
    """
    X = build_training_matrix(tracks, sequence_length)
    model = TemporalAnomalyModel(sequence_length=sequence_length,
                                 hidden_size=hidden_size).fit(X)
    return {
        "model":            model,
        "feature_columns":  list(STEP_FEATURES),
        "sequence_length":  sequence_length,
        "hidden_size":      hidden_size,
        "n_training_windows": int(X.shape[0]),
        "trained_at":       datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "model_version":    MODEL_VERSION,
        "training_config":  {"sequence_length": sequence_length,
                             "hidden_size": hidden_size,
                             "max_total_windows": 5000},
    }


def save_model(artifact: Dict, model_path: str = DEFAULT_TEMPORAL_PATH) -> None:
    parent = os.path.dirname(os.path.abspath(model_path))
    if parent and not os.path.exists(parent):
        os.makedirs(parent, exist_ok=True)
    joblib.dump(artifact, model_path)


def load_model(model_path: str = DEFAULT_TEMPORAL_PATH) -> Optional[Dict]:
    """Deserialize an artifact; return None if missing or corrupted."""
    if not os.path.exists(model_path):
        return None
    try:
        artifact = joblib.load(model_path)
    except Exception:
        return None
    if not isinstance(artifact, dict) or "model" not in artifact:
        return None
    return artifact


def model_exists(model_path: str = DEFAULT_TEMPORAL_PATH) -> bool:
    return os.path.exists(model_path)


# ---------------------------------------------------------------------------
# Compatibility
# ---------------------------------------------------------------------------

def model_is_compatible(artifact: Optional[Dict],
                        sequence_length: int = SEQUENCE_LENGTH,
                        feature_columns: List[str] = None) -> Tuple[bool, str]:
    """
    Validate a loaded artifact against the current temporal configuration:
    present & deserialized, metadata present, model can score, sequence length
    matches, step feature list matches, model version matches.
    """
    feature_columns = feature_columns or list(STEP_FEATURES)
    if artifact is None:
        return False, "no artifact (missing or corrupted)"
    model = artifact.get("model")
    if model is None or not hasattr(model, "score_norm"):
        return False, "artifact has no usable model object"
    if "sequence_length" not in artifact or "feature_columns" not in artifact:
        return False, "artifact metadata incomplete"
    if artifact.get("sequence_length") != sequence_length:
        return False, (f"sequence length mismatch "
                       f"(saved {artifact.get('sequence_length')}, now {sequence_length})")
    if list(artifact.get("feature_columns")) != list(feature_columns):
        return False, "step feature set changed"
    if artifact.get("model_version") != MODEL_VERSION:
        return False, (f"model version mismatch "
                       f"(saved {artifact.get('model_version')}, now {MODEL_VERSION})")
    return True, "compatible"


def get_model_metadata(artifact: Dict) -> Dict:
    if not artifact:
        return {}
    return {k: artifact[k] for k in (
        "feature_columns", "sequence_length", "hidden_size",
        "n_training_windows", "trained_at", "model_version") if k in artifact}


# ---------------------------------------------------------------------------
# High-level lifecycle
# ---------------------------------------------------------------------------

def load_or_train(tracks: Dict[str, pd.DataFrame],
                  sequence_length: int = SEQUENCE_LENGTH,
                  hidden_size: int = HIDDEN_SIZE,
                  model_path: str = DEFAULT_TEMPORAL_PATH) -> Tuple[Dict, str]:
    """
    Central M6 lifecycle entry point.

    Returns (artifact, status) with status in:
      "loaded"    — compatible artifact reused (NOT retrained)
      "trained"   — nothing existed, trained + saved
      "retrained" — incompatible/corrupt artifact replaced

    Never raises on a missing/corrupt artifact.
    """
    existing = load_model(model_path)
    compatible, _reason = model_is_compatible(existing, sequence_length)
    if compatible:
        return existing, "loaded"
    status = "retrained" if model_exists(model_path) else "trained"
    artifact = train_model(tracks, sequence_length, hidden_size)
    save_model(artifact, model_path)
    return artifact, status


def run_temporal_detection(
        tracks: Dict[str, pd.DataFrame],
        model_path: str = DEFAULT_TEMPORAL_PATH,
        sequence_length: int = SEQUENCE_LENGTH) -> Tuple[Dict[str, Dict], str, Dict]:
    """
    Full temporal detection pass: load-or-train, then score every vessel.

    Returns (results, status, metadata) where
      results = {vessel_id: {temporal_anomaly_score, temporal_model_status}}.
    """
    artifact, status = load_or_train(tracks, sequence_length, model_path=model_path)
    results = score_vessels(artifact["model"], tracks, sequence_length)
    meta = get_model_metadata(artifact)
    return results, status, meta
