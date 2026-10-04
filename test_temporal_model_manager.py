"""
test_temporal_model_manager.py
Tests for the M6 temporal model lifecycle (temporal_model_manager.py).

Uses TEMPORARY artifact paths — never the real models/temporal_model.joblib.

Run:
    python test_temporal_model_manager.py
or:
    python -m pytest test_temporal_model_manager.py -q
"""

import os
import tempfile

import pandas as pd

import temporal_model_manager as tmm
from temporal_model_manager import (
    train_model, save_model, load_model, model_exists,
    model_is_compatible, get_model_metadata, load_or_train, run_temporal_detection,
)
from temporal_model import SEQUENCE_LENGTH, STEP_FEATURES, MIN_OBS_FOR_SEQUENCE, MODEL_VERSION


def _track(n, lon0=10.0):
    rows = []
    for i in range(n):
        rows.append({
            "latitude": 55.0 + 0.01 * i,
            "longitude": lon0 + 0.01 * i,
            "speed": 8.0,
            "heading": 90,
            "timestamp": pd.Timestamp("2025-02-27") + pd.Timedelta(minutes=5 * i),
        })
    return pd.DataFrame(rows)


def _tracks(k=12, n=MIN_OBS_FOR_SEQUENCE + 5):
    return {f"V{i}": _track(n, lon0=10.0 + i) for i in range(k)}


def _tmp():
    d = tempfile.mkdtemp()
    return os.path.join(d, "sub", "temporal_model.joblib")


# ---------------------------------------------------------------------------
# train / save / load / exists / metadata
# ---------------------------------------------------------------------------

def test_train_save_load_meta():
    path = _tmp()
    art = train_model(_tracks())
    save_model(art, path)
    assert model_exists(path)
    loaded = load_model(path)
    assert loaded is not None
    meta = get_model_metadata(loaded)
    assert meta["sequence_length"] == SEQUENCE_LENGTH
    assert meta["feature_columns"] == STEP_FEATURES
    assert meta["model_version"] == MODEL_VERSION
    print("train/save/load/metadata  OK")


# ---------------------------------------------------------------------------
# compatibility checks
# ---------------------------------------------------------------------------

def test_compatibility():
    art = train_model(_tracks())
    ok, _ = model_is_compatible(art, SEQUENCE_LENGTH, STEP_FEATURES)
    assert ok
    # changed sequence length
    ok, reason = model_is_compatible(art, SEQUENCE_LENGTH + 1, STEP_FEATURES)
    assert not ok and "sequence length" in reason
    # changed feature set
    ok, reason = model_is_compatible(art, SEQUENCE_LENGTH, STEP_FEATURES[:-1])
    assert not ok and "feature set" in reason
    print("compatibility (seq length + feature set)  OK")


# ---------------------------------------------------------------------------
# corrupted artifact
# ---------------------------------------------------------------------------

def test_corrupted_artifact():
    path = _tmp()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write("not a joblib artifact")
    assert load_model(path) is None
    ok, _ = model_is_compatible(load_model(path), SEQUENCE_LENGTH, STEP_FEATURES)
    assert not ok
    print("corrupted artifact handled  OK")


# ---------------------------------------------------------------------------
# prediction after load
# ---------------------------------------------------------------------------

def test_predict_after_load():
    path = _tmp()
    save_model(train_model(_tracks()), path)
    res, status, _ = run_temporal_detection(_tracks(), model_path=path)
    assert status == "loaded"
    assert all(0.0 <= r["temporal_anomaly_score"] <= 1.0 for r in res.values())
    print("prediction after load  OK")


# ---------------------------------------------------------------------------
# REUSE proof — train once, then load (no retrain), retrain on schema change
# ---------------------------------------------------------------------------

def test_reuse_no_retrain():
    path = _tmp()
    tracks = _tracks()

    calls = {"n": 0}
    real_train = tmm.train_model

    def _spy(t, sequence_length=SEQUENCE_LENGTH, hidden_size=32):
        calls["n"] += 1
        return real_train(t, sequence_length, hidden_size)

    tmm.train_model = _spy
    try:
        art1, s1 = load_or_train(tracks, model_path=path)
        assert s1 == "trained" and calls["n"] == 1
        meta1 = get_model_metadata(art1)

        art2, s2 = load_or_train(tracks, model_path=path)
        assert s2 == "loaded" and calls["n"] == 1, "must not retrain on reuse"
        meta2 = get_model_metadata(art2)
        assert meta1["trained_at"] == meta2["trained_at"]

        # schema change (seq length) → retrain
        art3, s3 = load_or_train(tracks, sequence_length=SEQUENCE_LENGTH + 2, model_path=path)
        assert s3 == "retrained" and calls["n"] == 2
    finally:
        tmm.train_model = real_train
    print("reuse proof (train once, load, retrain on schema change)  OK")


if __name__ == "__main__":
    tests = [
        test_train_save_load_meta, test_compatibility, test_corrupted_artifact,
        test_predict_after_load, test_reuse_no_retrain,
    ]
    p = 0
    for t in tests:
        t(); p += 1
    print(f"\n=== {p}/{len(tests)} TESTS PASSED ===")
