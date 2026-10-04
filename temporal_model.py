"""
temporal_model.py
Temporal sequence anomaly detection (Round 3 — Milestone M6).

Approach
--------
This is an UNSUPERVISED sequence reconstruction anomaly detector built with
scikit-learn (PCA), deliberately avoiding a heavyweight deep-learning
dependency. It behaves as a linear sequence autoencoder:

    sequence window  →  PCA encode (compress)  →  PCA decode (reconstruct)
                     →  reconstruction error    →  temporal anomaly score

The model learns the dominant recurring sequence patterns across the fleet.
Windows it reconstructs poorly (high error) are flagged as temporally unusual.
It evaluates behaviour ACROSS TIME (ordered windows), not independent points.

Honesty note
------------
There is no labelled illegal-fishing ground truth. The output is a
``temporal_anomaly_score`` / ``sequence_anomaly_score`` in [0, 1] (higher = more
unusual sequence). No "illegal fishing probability" is produced, and no accuracy
metric is claimed.

Sequence feature set (6 per step, documented and fixed)
-------------------------------------------------------
For each consecutive observation pair in a vessel's chronological track:
    speed            — SOG at the step (knots)
    heading_change   — normalised bearing change in [-180, 180] degrees
    lat_change       — latitude delta (degrees)
    lon_change       — longitude delta (degrees)
    step_distance_km — Haversine distance of the step (km)
    gap_minutes      — time gap between observations (minutes; 0 if no timestamp)
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from behavior_analysis import haversine_km, _angular_diff

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

SEQUENCE_LENGTH: int = 10          # window length (steps) — configurable
STEP_FEATURES: List[str] = [
    "speed", "heading_change", "lat_change", "lon_change",
    "step_distance_km", "gap_minutes",
]
N_STEP_FEATURES: int = len(STEP_FEATURES)          # 6
HIDDEN_SIZE: int = 32              # PCA components cap (the "bottleneck")
RANDOM_STATE: int = 42
MODEL_VERSION: str = "m6"

# A vessel needs at least SEQUENCE_LENGTH+1 raw observations to form one window
# of SEQUENCE_LENGTH steps (steps are between consecutive observations).
MIN_OBS_FOR_SEQUENCE: int = SEQUENCE_LENGTH + 1


# ---------------------------------------------------------------------------
# Sequence construction
# ---------------------------------------------------------------------------

def _steps_from_track(track: pd.DataFrame) -> np.ndarray:
    """
    Convert one vessel's chronological observations into a (n_steps, 6) array
    of per-step features. Returns an empty array if fewer than 2 valid points.

    Robust to missing speed/heading/timestamp (filled with neutral 0.0 — never
    fabricated movement) and invalid coordinates (skipped).
    """
    if track is None or len(track) < 2:
        return np.empty((0, N_STEP_FEATURES), dtype=float)

    df = track
    if "timestamp" in df.columns:
        ts = pd.to_datetime(df["timestamp"], errors="coerce")
        df = df.assign(_ts=ts).sort_values("_ts")
    else:
        df = df.copy()
        df["_ts"] = pd.NaT

    lat = pd.to_numeric(df["latitude"], errors="coerce").to_numpy()
    lon = pd.to_numeric(df["longitude"], errors="coerce").to_numpy()
    spd = (pd.to_numeric(df["speed"], errors="coerce").to_numpy()
           if "speed" in df.columns else np.full(len(df), np.nan))
    hdg = (pd.to_numeric(df["heading"], errors="coerce").to_numpy()
           if "heading" in df.columns else np.full(len(df), np.nan))
    ts_vals = df["_ts"].to_numpy()

    valid = ~(np.isnan(lat) | np.isnan(lon))
    lat, lon, spd, hdg, ts_vals = lat[valid], lon[valid], spd[valid], hdg[valid], ts_vals[valid]
    n = len(lat)
    if n < 2:
        return np.empty((0, N_STEP_FEATURES), dtype=float)

    steps = []
    for i in range(n - 1):
        speed_i = 0.0 if np.isnan(spd[i + 1]) else float(spd[i + 1])
        if np.isnan(hdg[i]) or np.isnan(hdg[i + 1]):
            hchange = 0.0
        else:
            hchange = float(_angular_diff(hdg[i], hdg[i + 1]))
        lat_ch = float(lat[i + 1] - lat[i])
        lon_ch = float(lon[i + 1] - lon[i])
        dist = haversine_km(lat[i], lon[i], lat[i + 1], lon[i + 1])
        # gap minutes
        gap = 0.0
        try:
            if not pd.isna(ts_vals[i]) and not pd.isna(ts_vals[i + 1]):
                gap = max(0.0, (pd.Timestamp(ts_vals[i + 1]) - pd.Timestamp(ts_vals[i])).total_seconds() / 60.0)
        except Exception:
            gap = 0.0
        steps.append([speed_i, hchange, lat_ch, lon_ch, dist, gap])

    return np.asarray(steps, dtype=float)


def build_sequences(track: pd.DataFrame,
                    sequence_length: int = SEQUENCE_LENGTH) -> np.ndarray:
    """
    Build fixed-length non-overlapping-is-not-required sliding windows for a
    single vessel.

    Returns an array of shape (n_windows, sequence_length * 6). Returns an empty
    array when the track is too short (safe for short/insufficient trajectories).
    """
    steps = _steps_from_track(track)
    if steps.shape[0] < sequence_length:
        return np.empty((0, sequence_length * N_STEP_FEATURES), dtype=float)

    windows = []
    # Sliding windows, stride 1. Cap the number of windows per vessel to keep
    # training/scoring bounded on long tracks.
    max_windows = 50
    stride = 1
    idxs = range(0, steps.shape[0] - sequence_length + 1, stride)
    idxs = list(idxs)[:max_windows]
    for start in idxs:
        win = steps[start:start + sequence_length].reshape(-1)
        windows.append(win)
    return np.asarray(windows, dtype=float)


def build_training_matrix(
        tracks: Dict[str, pd.DataFrame],
        sequence_length: int = SEQUENCE_LENGTH,
        max_total_windows: int = 5000,
) -> np.ndarray:
    """
    Stack sequence windows from many vessels into one training matrix.

    Caps the total number of windows (default 5000) so training stays fast on a
    student laptop. Returns shape (n_windows, sequence_length * 6); may be empty.
    """
    all_windows = []
    for _vid, track in tracks.items():
        w = build_sequences(track, sequence_length)
        if w.shape[0]:
            all_windows.append(w)
        if sum(x.shape[0] for x in all_windows) >= max_total_windows:
            break
    if not all_windows:
        return np.empty((0, sequence_length * N_STEP_FEATURES), dtype=float)
    stacked = np.vstack(all_windows)
    if stacked.shape[0] > max_total_windows:
        stacked = stacked[:max_total_windows]
    return stacked


# ---------------------------------------------------------------------------
# PCA sequence autoencoder
# ---------------------------------------------------------------------------

class TemporalAnomalyModel:
    """
    A PCA-based linear sequence autoencoder.

    fit(X)        : standardise windows, fit PCA (bottleneck = min(HIDDEN_SIZE,
                    n_features, n_samples)), and record a reconstruction-error
                    scale from the training data for deterministic normalisation.
    score(X)      : reconstruction error per window.
    score_norm(X) : reconstruction error mapped to [0, 1] using the stored
                    training-error scale (deterministic across runs).
    """

    def __init__(self, sequence_length: int = SEQUENCE_LENGTH,
                 hidden_size: int = HIDDEN_SIZE):
        self.sequence_length = sequence_length
        self.hidden_size = hidden_size
        self.scaler: Optional[StandardScaler] = None
        self.pca: Optional[PCA] = None
        self.err_ref_: float = 1.0   # reference error for normalisation (p95)

    def fit(self, X: np.ndarray) -> "TemporalAnomalyModel":
        if X.shape[0] == 0:
            # Degenerate: nothing to learn. Leave as identity-ish; scores → 0.
            self.scaler = None
            self.pca = None
            self.err_ref_ = 1.0
            return self
        self.scaler = StandardScaler().fit(X)
        Xs = self.scaler.transform(X)
        n_comp = int(min(self.hidden_size, Xs.shape[1], Xs.shape[0]))
        n_comp = max(1, n_comp)
        self.pca = PCA(n_components=n_comp, random_state=RANDOM_STATE).fit(Xs)
        errs = self._recon_error(Xs)
        # p95 of training error as the normalisation reference (robust to tails).
        self.err_ref_ = float(np.percentile(errs, 95)) if errs.size else 1.0
        if self.err_ref_ <= 1e-9:
            self.err_ref_ = 1.0
        return self

    def _recon_error(self, Xs: np.ndarray) -> np.ndarray:
        recon = self.pca.inverse_transform(self.pca.transform(Xs))
        return np.mean((Xs - recon) ** 2, axis=1)

    def score(self, X: np.ndarray) -> np.ndarray:
        """Raw reconstruction error per window (0.0 if the model is degenerate)."""
        if self.pca is None or self.scaler is None or X.shape[0] == 0:
            return np.zeros(X.shape[0], dtype=float)
        Xs = self.scaler.transform(X)
        return self._recon_error(Xs)

    def score_norm(self, X: np.ndarray) -> np.ndarray:
        """
        Normalised reconstruction error in [0, 1] using the stored training
        reference (deterministic — NOT per-call min/max). error/ref clipped to 1.
        """
        raw = self.score(X)
        return np.clip(raw / self.err_ref_, 0.0, 1.0)


# ---------------------------------------------------------------------------
# High-level per-vessel scoring
# ---------------------------------------------------------------------------

def score_vessels(model: TemporalAnomalyModel,
                  tracks: Dict[str, pd.DataFrame],
                  sequence_length: int = SEQUENCE_LENGTH) -> Dict[str, Dict]:
    """
    Produce a per-vessel temporal anomaly result.

    A vessel's score is the MAX normalised window error across its windows
    (worst sequence drives the signal). Vessels without enough observations get
    temporal_anomaly_score = 0.0 and temporal_model_status = "insufficient_data".

    Returns {vessel_id: {"temporal_anomaly_score": float,
                         "temporal_model_status": str}}
    """
    results: Dict[str, Dict] = {}
    for vid, track in tracks.items():
        windows = build_sequences(track, sequence_length)
        if windows.shape[0] == 0:
            results[vid] = {"temporal_anomaly_score": 0.0,
                            "temporal_model_status": "insufficient_data"}
            continue
        norm = model.score_norm(windows)
        results[vid] = {
            "temporal_anomaly_score": round(float(np.max(norm)), 4),
            "temporal_model_status": "scored",
        }
    return results
