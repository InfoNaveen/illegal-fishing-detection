"""
test_temporal_model.py
Tests for the M6 temporal sequence anomaly model (temporal_model.py).

Run:
    python test_temporal_model.py
or:
    python -m pytest test_temporal_model.py -q
"""

import numpy as np
import pandas as pd

from temporal_model import (
    build_sequences, build_training_matrix, score_vessels,
    TemporalAnomalyModel, SEQUENCE_LENGTH, N_STEP_FEATURES, MIN_OBS_FOR_SEQUENCE,
    _steps_from_track,
)


def _track(n, start_lat=55.0, start_lon=10.0, dstep=0.01, speed=8.0, gap_min=5):
    rows = []
    for i in range(n):
        rows.append({
            "latitude":  start_lat + dstep * i,
            "longitude": start_lon + dstep * i,
            "speed":     speed,
            "heading":   90,
            "timestamp": pd.Timestamp("2025-02-27 00:00:00") + pd.Timedelta(minutes=gap_min * i),
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 1. sequence creation
# ---------------------------------------------------------------------------

def test_1_sequence_creation():
    track = _track(MIN_OBS_FOR_SEQUENCE)  # exactly enough for 1 window
    wins = build_sequences(track)
    assert wins.shape[1] == SEQUENCE_LENGTH * N_STEP_FEATURES
    assert wins.shape[0] >= 1
    print(f"1. sequence creation  OK (window dim={wins.shape[1]})")


# ---------------------------------------------------------------------------
# 2. chronological ordering (out-of-order input is sorted)
# ---------------------------------------------------------------------------

def test_2_chronological_ordering():
    track = _track(MIN_OBS_FOR_SEQUENCE)
    shuffled = track.sample(frac=1.0, random_state=1).reset_index(drop=True)
    steps_ordered = _steps_from_track(track)
    steps_shuffled = _steps_from_track(shuffled)
    assert np.allclose(steps_ordered, steps_shuffled), "steps must be timestamp-ordered"
    print("2. chronological ordering  OK")


# ---------------------------------------------------------------------------
# 3. short trajectories -> no windows
# ---------------------------------------------------------------------------

def test_3_short_trajectory():
    short = _track(MIN_OBS_FOR_SEQUENCE - 1)  # one short of a window
    assert build_sequences(short).shape[0] == 0
    assert build_sequences(_track(1)).shape[0] == 0
    assert build_sequences(pd.DataFrame(columns=["latitude","longitude","speed","heading","timestamp"])).shape[0] == 0
    print("3. short trajectory handled  OK")


# ---------------------------------------------------------------------------
# 4. missing values (speed/heading/timestamp) do not crash
# ---------------------------------------------------------------------------

def test_4_missing_values():
    track = _track(MIN_OBS_FOR_SEQUENCE)
    track["speed"] = np.nan
    track["heading"] = np.nan
    track = track.drop(columns=["timestamp"])
    wins = build_sequences(track)   # must not raise
    assert wins.shape[0] >= 1
    assert np.isfinite(wins).all(), "windows must contain finite values"
    print("4. missing values handled  OK")


# ---------------------------------------------------------------------------
# 5. model training
# ---------------------------------------------------------------------------

def test_5_training():
    tracks = {f"V{i}": _track(MIN_OBS_FOR_SEQUENCE + 5, start_lon=10.0 + i) for i in range(10)}
    X = build_training_matrix(tracks)
    assert X.shape[0] > 0
    model = TemporalAnomalyModel().fit(X)
    assert model.pca is not None and model.scaler is not None
    print(f"5. training  OK (train windows={X.shape[0]})")


# ---------------------------------------------------------------------------
# 6. prediction + score bounds
# ---------------------------------------------------------------------------

def test_6_prediction_bounds():
    tracks = {f"V{i}": _track(MIN_OBS_FOR_SEQUENCE + 5, start_lon=10.0 + i) for i in range(10)}
    model = TemporalAnomalyModel().fit(build_training_matrix(tracks))
    wins = build_sequences(tracks["V0"])
    norm = model.score_norm(wins)
    assert norm.min() >= 0.0 and norm.max() <= 1.0, "normalised scores must be in [0,1]"
    print("6. prediction + score bounds [0,1]  OK")


# ---------------------------------------------------------------------------
# 7. score_vessels: scored vs insufficient_data
# ---------------------------------------------------------------------------

def test_7_score_vessels_statuses():
    model = TemporalAnomalyModel().fit(
        build_training_matrix({f"V{i}": _track(MIN_OBS_FOR_SEQUENCE + 5, start_lon=10.0 + i) for i in range(10)}))
    tracks = {
        "LONG":  _track(MIN_OBS_FOR_SEQUENCE + 3),
        "SHORT": _track(3),
        "EMPTY": pd.DataFrame(columns=["latitude","longitude","speed","heading","timestamp"]),
    }
    res = score_vessels(model, tracks)
    assert res["LONG"]["temporal_model_status"] == "scored"
    assert res["SHORT"]["temporal_model_status"] == "insufficient_data"
    assert res["EMPTY"]["temporal_model_status"] == "insufficient_data"
    assert res["SHORT"]["temporal_anomaly_score"] == 0.0
    assert 0.0 <= res["LONG"]["temporal_anomaly_score"] <= 1.0
    print("7. score_vessels statuses  OK")


# ---------------------------------------------------------------------------
# 8. anomalous sequence scores higher than normal recurring pattern
# ---------------------------------------------------------------------------

def test_8_anomaly_discrimination():
    # Train on many steady straight tracks; score a wild erratic track.
    normal = {f"N{i}": _track(MIN_OBS_FOR_SEQUENCE + 10, start_lon=10.0 + i, dstep=0.01, speed=8.0)
              for i in range(20)}
    model = TemporalAnomalyModel().fit(build_training_matrix(normal))

    rng = np.random.default_rng(0)
    erratic_rows = []
    for i in range(MIN_OBS_FOR_SEQUENCE + 5):
        erratic_rows.append({
            "latitude": 55.0 + rng.uniform(-0.5, 0.5),
            "longitude": 10.0 + rng.uniform(-0.5, 0.5),
            "speed": rng.uniform(0, 30),
            "heading": rng.uniform(0, 360),
            "timestamp": pd.Timestamp("2025-02-27") + pd.Timedelta(minutes=5 * i),
        })
    erratic = pd.DataFrame(erratic_rows)

    normal_score = score_vessels(model, {"N0": normal["N0"]})["N0"]["temporal_anomaly_score"]
    erratic_score = score_vessels(model, {"E": erratic})["E"]["temporal_anomaly_score"]
    assert erratic_score > normal_score, (erratic_score, normal_score)
    print(f"8. anomaly discrimination  OK (erratic={erratic_score:.3f} > normal={normal_score:.3f})")


# ---------------------------------------------------------------------------
# 9. degenerate model (empty training) scores 0 safely
# ---------------------------------------------------------------------------

def test_9_degenerate_model():
    model = TemporalAnomalyModel().fit(np.empty((0, SEQUENCE_LENGTH * N_STEP_FEATURES)))
    wins = build_sequences(_track(MIN_OBS_FOR_SEQUENCE))
    norm = model.score_norm(wins)
    assert np.all(norm == 0.0), "degenerate model must score 0"
    print("9. degenerate model safe  OK")


if __name__ == "__main__":
    tests = [
        test_1_sequence_creation, test_2_chronological_ordering,
        test_3_short_trajectory, test_4_missing_values, test_5_training,
        test_6_prediction_bounds, test_7_score_vessels_statuses,
        test_8_anomaly_discrimination, test_9_degenerate_model,
    ]
    p = 0
    for t in tests:
        t(); p += 1
    print(f"\n=== {p}/{len(tests)} TESTS PASSED ===")
