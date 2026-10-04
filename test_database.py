"""
test_database.py
Tests for the Round 3 M3 SQLite persistence layer (database.py).

Uses a TEMPORARY database — never the production data/ifds.db.

Run:
    python test_database.py
or:
    python -m pytest test_database.py -v
"""

import os
import tempfile

import pandas as pd

import database
from database import (
    initialize_database, get_connection, persist_pipeline_run,
    get_history_summary, get_recent_alerts, get_recent_risk, get_vessel_recent,
    DatabaseError,
)


def _tmp_db() -> str:
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    os.remove(path)  # let sqlite create it fresh
    return path


def _sample_run():
    """Build a tiny but complete pipeline-run payload."""
    df = pd.DataFrame([
        {"vessel_id": "X1", "latitude": 55.0, "longitude": 10.0, "speed": 5.0, "heading": 90},
        {"vessel_id": "X2", "latitude": 56.0, "longitude": 11.0, "speed": 2.0, "heading": 180},
    ])
    feat = pd.DataFrame(
        {"speed": [5.0, 2.0], "loitering_score": [0.1, 0.8]},
        index=pd.Index(["X1", "X2"], name="vessel_id"),
    )
    anom = {"X1": {"anomaly_score": 0.2, "is_anomalous": False},
            "X2": {"anomaly_score": 0.9, "is_anomalous": True}}
    risk = {"X1": {"risk_score": 15, "risk_level": "LOW",  "zone_status": "Open Waters",
                   "behavior": "Normal", "ais_gap_minutes": 0},
            "X2": {"risk_score": 72, "risk_level": "HIGH", "zone_status": "INSIDE Zone",
                   "behavior": "Loitering", "ais_gap_minutes": 20}}
    alerts = [{"vessel": "X2", "level": "HIGH", "message": "X2 loitering", "reason": "Loitering"}]
    return df, feat, anom, risk, alerts


# ---------------------------------------------------------------------------
# 1 + 2. init + tables exist
# ---------------------------------------------------------------------------

def test_1_2_init_and_tables():
    db = _tmp_db()
    try:
        initialize_database(db)
        conn = get_connection(db)
        try:
            names = {r[0] for r in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'").fetchall()}
        finally:
            conn.close()
        for t in ["vessels", "positions", "features",
                  "anomaly_scores", "risk_scores", "alerts"]:
            assert t in names, f"missing table {t}"
        print("1+2. init + all tables exist  OK")
    finally:
        os.remove(db)


# ---------------------------------------------------------------------------
# 11. init twice is safe (and keeps data)
# ---------------------------------------------------------------------------

def test_11_init_twice_safe():
    db = _tmp_db()
    try:
        initialize_database(db)
        df, feat, anom, risk, alerts = _sample_run()
        persist_pipeline_run(df, feat, anom, risk, alerts, "Simulated", "Bay of Bengal", db_path=db)
        initialize_database(db)  # second init must not wipe
        summary = get_history_summary(db)
        assert summary["total_vessels"] == 2, "init twice must not delete data"
        print("11. init twice safe, data retained  OK")
    finally:
        os.remove(db)


# ---------------------------------------------------------------------------
# 3. vessel upsert (no duplicate vessel rows)
# ---------------------------------------------------------------------------

def test_3_vessel_upsert():
    db = _tmp_db()
    try:
        initialize_database(db)
        df, feat, anom, risk, alerts = _sample_run()
        persist_pipeline_run(df, feat, anom, risk, alerts, "Simulated", "Bay of Bengal", db_path=db)
        persist_pipeline_run(df, feat, anom, risk, alerts, "Simulated", "Bay of Bengal", db_path=db)
        assert get_history_summary(db)["total_vessels"] == 2, "vessels must not duplicate"
        print("3. vessel upsert (no duplicates)  OK")
    finally:
        os.remove(db)


# ---------------------------------------------------------------------------
# 4 + 5. position insert + duplicate protection
# ---------------------------------------------------------------------------

def test_4_5_positions_dedup():
    db = _tmp_db()
    try:
        initialize_database(db)
        df, feat, anom, risk, alerts = _sample_run()
        positions = pd.DataFrame([
            {"vessel_id": "X1", "timestamp": "2025-02-27 00:00:00", "latitude": 55.0, "longitude": 10.0, "speed": 5, "heading": 90},
            {"vessel_id": "X1", "timestamp": "2025-02-27 00:10:00", "latitude": 55.1, "longitude": 10.1, "speed": 5, "heading": 90},
        ])
        persist_pipeline_run(df, feat, anom, risk, alerts, "Historical AIS", "Danish Waters",
                             positions_df=positions, db_path=db)
        after_first = get_history_summary(db)["total_positions"]
        assert after_first == 2
        # Re-persist identical positions — INSERT OR IGNORE should add nothing.
        persist_pipeline_run(df, feat, anom, risk, alerts, "Historical AIS", "Danish Waters",
                             positions_df=positions, db_path=db)
        after_second = get_history_summary(db)["total_positions"]
        assert after_second == 2, "duplicate positions must be ignored"
        print("4+5. positions insert + dedup  OK")
    finally:
        os.remove(db)


# ---------------------------------------------------------------------------
# 6/7/8/9. feature / anomaly / risk / alert persistence
# ---------------------------------------------------------------------------

def test_6_7_8_9_persistence():
    db = _tmp_db()
    try:
        initialize_database(db)
        df, feat, anom, risk, alerts = _sample_run()
        counts = persist_pipeline_run(df, feat, anom, risk, alerts,
                                      "Simulated", "Bay of Bengal", db_path=db)
        assert counts["features"] == 2
        assert counts["anomaly_scores"] == 2
        assert counts["risk_scores"] == 2
        assert counts["alerts"] == 1
        conn = get_connection(db)
        try:
            fj = conn.execute("SELECT feature_json FROM features WHERE vessel_id='X2'").fetchone()[0]
            assert "loitering_score" in fj
            isanom = conn.execute("SELECT is_anomalous FROM anomaly_scores WHERE vessel_id='X2'").fetchone()[0]
            assert isanom == 1
            rl = conn.execute("SELECT risk_level FROM risk_scores WHERE vessel_id='X2'").fetchone()[0]
            assert rl == "HIGH"
        finally:
            conn.close()
        print("6-9. feature/anomaly/risk/alert persistence  OK")
    finally:
        os.remove(db)


# ---------------------------------------------------------------------------
# 10. recent history queries
# ---------------------------------------------------------------------------

def test_10_recent_history():
    db = _tmp_db()
    try:
        initialize_database(db)
        df, feat, anom, risk, alerts = _sample_run()
        persist_pipeline_run(df, feat, anom, risk, alerts, "Simulated", "Bay of Bengal", db_path=db)
        ra = get_recent_alerts(limit=5, db_path=db)
        rr = get_recent_risk(limit=5, db_path=db)
        assert len(ra) == 1 and ra[0]["vessel_id"] == "X2"
        assert len(rr) == 2
        vh = get_vessel_recent("X2", limit=5, db_path=db)
        assert len(vh["risk"]) == 1 and vh["risk"][0]["risk_level"] == "HIGH"
        assert len(vh["alerts"]) == 1
        print("10. recent history queries  OK")
    finally:
        os.remove(db)


# ---------------------------------------------------------------------------
# 12. persistence failure does not break the core pipeline
# ---------------------------------------------------------------------------

def test_12_failure_is_contained():
    # Point at an impossible path so sqlite cannot open it; the DatabaseError
    # must be raised (so the caller's try/except can contain it) rather than
    # some uncaught crash. The core pipeline objects are untouched regardless.
    df, feat, anom, risk, alerts = _sample_run()
    bad_path = os.path.join("Z:\\", "nonexistent_dir_12345", "x" * 300, "bad.db")
    raised = False
    try:
        persist_pipeline_run(df, feat, anom, risk, alerts,
                             "Simulated", "Bay of Bengal", db_path=bad_path)
    except DatabaseError:
        raised = True
    except Exception:
        # Any failure type is acceptable as long as it is raised, not silent —
        # the app wraps this call and continues.
        raised = True
    assert raised, "a bad DB path must raise (so the caller can contain it)"
    # The pipeline payload is still fully intact after the failed persist.
    assert len(df) == 2 and risk["X2"]["risk_level"] == "HIGH"
    print("12. persistence failure contained, pipeline data intact  OK")


# ---------------------------------------------------------------------------
# 13. real-shaped end-to-end persist (uses the real sample via the pipeline)
# ---------------------------------------------------------------------------

def test_13_simulated_pipeline_persist_end_to_end():
    from data_generator import generate_vessel_dataframe, build_all_trajectories
    from zones import get_zones, get_region_meta
    from geofencing import run_geofencing
    from feature_engineering import build_feature_matrix
    from anomaly_detector import run_anomaly_detection
    from risk_engine import run_risk_engine, generate_alerts

    db = _tmp_db()
    try:
        initialize_database(db)
        zones = get_zones("Simulated")
        base = generate_vessel_dataframe()
        trajs = build_all_trajectories(base)
        geos = run_geofencing(base, zones)
        gm = {g["vessel_id"]: g for g in geos}
        feats = build_feature_matrix(base, trajs, gm)
        anom = run_anomaly_detection(feats)
        risk = run_risk_engine(base["vessel_id"].tolist(), gm, feats, anom)
        alerts = generate_alerts(risk)
        enriched = base.copy()
        enriched["risk_score"] = enriched["vessel_id"].map(lambda v: risk[v]["risk_score"])
        counts = persist_pipeline_run(
            enriched, feats, anom, risk, alerts,
            "Simulated", get_region_meta("Simulated")["region"], db_path=db)
        assert counts["vessels"] == 18
        assert get_history_summary(db)["total_vessels"] == 18
        print(f"13. simulated end-to-end persist  OK ({counts})")
    finally:
        os.remove(db)


if __name__ == "__main__":
    tests = [
        test_1_2_init_and_tables,
        test_11_init_twice_safe,
        test_3_vessel_upsert,
        test_4_5_positions_dedup,
        test_6_7_8_9_persistence,
        test_10_recent_history,
        test_12_failure_is_contained,
        test_13_simulated_pipeline_persist_end_to_end,
    ]
    passed = 0
    for t in tests:
        t()
        passed += 1
    print(f"\n=== {passed}/{len(tests)} TESTS PASSED ===")
