"""
test_model_manager.py
Tests for the Round 3 M5 persistent Isolation Forest lifecycle (model_manager.py).

Uses TEMPORARY artifact paths — never writes the real models/isolation_forest.joblib.

Run:
    python test_model_manager.py
or:
    python -m pytest test_model_manager.py -q
"""

import os
import tempfile

import numpy as np
import pandas as pd

import model_manager
from model_manager import (
    train_model, save_model, load_model, model_exists,
    model_is_compatible, get_model_metadata, predict_scores, load_or_train,
    MODEL_VERSION,
)

COLS = ["f1", "f2", "f3"]


def _feature_df(n=20, cols=COLS, seed=0):
    rng = np.random.default_rng(seed)
    data = {c: rng.random(n) for c in cols}
    df = pd.DataFrame(data)
    df.index = [f"V{i:03d}" for i in range(n)]
    df.index.name = "vessel_id"
    return df


def _tmp_path() -> str:
    d = tempfile.mkdtemp()
    return os.path.join(d, "sub", "isolation_forest.joblib")  # sub dir must be auto-created


# ---------------------------------------------------------------------------
# 1-5. train / save / load / exists / metadata
# ---------------------------------------------------------------------------

def test_1_5_train_save_load_exists_metadata():
    path = _tmp_path()
    df = _feature_df()
    art = train_model(df, COLS)
    assert "model" in art and "scaler" in art
    save_model(art, path)
    assert model_exists(path), "artifact file should exist after save"
    loaded = load_model(path)
    assert loaded is not None
    meta = get_model_metadata(loaded)
    assert meta["feature_count"] == 3
    assert meta["feature_columns"] == COLS
    assert meta["model_version"] == MODEL_VERSION
    assert meta["n_estimators"] == 200
    assert meta["contamination"] == 0.20
    assert meta["random_state"] == 42
    print("1-5. train/save/load/exists/metadata  OK")


# ---------------------------------------------------------------------------
# 6. compatible schema loads successfully
# ---------------------------------------------------------------------------

def test_6_compatible_schema():
    df = _feature_df()
    art = train_model(df, COLS)
    ok, reason = model_is_compatible(art, COLS)
    assert ok, reason
    print("6. compatible schema OK  OK")


# ---------------------------------------------------------------------------
# 7. changed feature order is incompatible
# ---------------------------------------------------------------------------

def test_7_changed_order_incompatible():
    df = _feature_df()
    art = train_model(df, COLS)
    ok, reason = model_is_compatible(art, ["f2", "f1", "f3"])
    assert not ok and "order" in reason, reason
    print("7. changed feature order incompatible  OK")


# ---------------------------------------------------------------------------
# 8. changed feature count is incompatible
# ---------------------------------------------------------------------------

def test_8_changed_count_incompatible():
    df = _feature_df()
    art = train_model(df, COLS)
    ok, reason = model_is_compatible(art, ["f1", "f2"])
    assert not ok and "count" in reason, reason
    print("8. changed feature count incompatible  OK")


# ---------------------------------------------------------------------------
# 9. changed feature name is incompatible
# ---------------------------------------------------------------------------

def test_9_changed_name_incompatible():
    df = _feature_df()
    art = train_model(df, COLS)
    ok, reason = model_is_compatible(art, ["f1", "f2", "fX"])
    assert not ok and "name" in reason, reason
    print("9. changed feature name incompatible  OK")


# ---------------------------------------------------------------------------
# 10. corrupted artifact triggers safe handling
# ---------------------------------------------------------------------------

def test_10_corrupted_artifact():
    path = _tmp_path()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write("this is not a joblib file")
    assert load_model(path) is None, "corrupted artifact must load as None"
    ok, reason = model_is_compatible(load_model(path), COLS)
    assert not ok
    print("10. corrupted artifact handled safely  OK")


# ---------------------------------------------------------------------------
# 11. prediction works after load
# ---------------------------------------------------------------------------

def test_11_predict_after_load():
    path = _tmp_path()
    df = _feature_df()
    save_model(train_model(df, COLS), path)
    loaded = load_model(path)
    scores = predict_scores(loaded, df, COLS)
    assert len(scores) == len(df)
    assert all(0.0 <= s <= 1.0 for s in scores.values()), "scores must be in [0,1]"
    print("11. prediction after load  OK")


# ---------------------------------------------------------------------------
# 12 + 16. REUSE PROOF — first run trains, second run loads (no retrain)
# ---------------------------------------------------------------------------

def test_12_16_reuse_no_retrain(monkeypatch=None):
    path = _tmp_path()
    df = _feature_df()

    # Spy on train_model to count how many times it is actually called.
    calls = {"n": 0}
    real_train = model_manager.train_model

    def _spy(fdf, cols):
        calls["n"] += 1
        return real_train(fdf, cols)

    model_manager.train_model = _spy
    try:
        # FIRST RUN: no artifact → must train + save
        art1, status1 = load_or_train(df, COLS, path)
        assert status1 == "trained", status1
        assert calls["n"] == 1, "first run must train exactly once"
        assert model_exists(path)
        trained_at_1 = get_model_metadata(art1)["trained_at"]

        # SECOND RUN: compatible artifact exists → must load, NOT train
        art2, status2 = load_or_train(df, COLS, path)
        assert status2 == "loaded", status2
        assert calls["n"] == 1, "second run must NOT retrain (train count unchanged)"
        trained_at_2 = get_model_metadata(art2)["trained_at"]
        assert trained_at_1 == trained_at_2, "reused model keeps original trained_at"

        # THIRD RUN with changed schema → must retrain
        new_cols = COLS + ["f4"]
        df2 = _feature_df(cols=new_cols)
        art3, status3 = load_or_train(df2, new_cols, path)
        assert status3 == "retrained", status3
        assert calls["n"] == 2, "schema change must trigger exactly one retrain"
    finally:
        model_manager.train_model = real_train
    print("12+16. reuse proof (train once, load, retrain on schema change)  OK")


# ---------------------------------------------------------------------------
# Score-semantics regression: persistent path matches the legacy fit closely
# ---------------------------------------------------------------------------

def test_score_semantics_preserved():
    from anomaly_detector import train_and_score, run_anomaly_detection
    df = _feature_df(seed=7)
    path = _tmp_path()
    legacy = train_and_score(df)                       # in-memory legacy fit
    persistent = run_anomaly_detection(df, model_path=path)  # trains + saves + predicts
    # Both use IF(seed=42) + same scaling + same normalisation → identical scores.
    for vid in legacy:
        assert abs(legacy[vid] - persistent[vid]["anomaly_score"]) < 1e-9, vid
    print("    score semantics preserved vs legacy  OK")


if __name__ == "__main__":
    tests = [
        test_1_5_train_save_load_exists_metadata,
        test_6_compatible_schema,
        test_7_changed_order_incompatible,
        test_8_changed_count_incompatible,
        test_9_changed_name_incompatible,
        test_10_corrupted_artifact,
        test_11_predict_after_load,
        test_12_16_reuse_no_retrain,
        test_score_semantics_preserved,
    ]
    passed = 0
    for t in tests:
        t()
        passed += 1
    print(f"\n=== {passed}/{len(tests)} TESTS PASSED ===")
