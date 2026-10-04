"""
test_realtime_simulation.py
Tests for M8 AIS replay simulation (realtime_simulator.py) and the alert
engine (alert_engine.py), including SQLite alert persistence.

Uses a temporary SQLite DB — never the production data/ifds.db.

Run:
    python test_realtime_simulation.py
or:
    python -m pytest test_realtime_simulation.py -q
"""

import os
import tempfile

import pandas as pd

from realtime_simulator import AISReplaySimulator
from alert_engine import AlertEngine
import database


def _clean_df(n_vessels=4, per=8):
    rows = []
    base = pd.Timestamp("2025-02-27 00:00:00")
    for v in range(n_vessels):
        for i in range(per):
            rows.append({
                "vessel_id": f"V{v}",
                "timestamp": base + pd.Timedelta(minutes=5 * (i + v)),  # interleaved
                "latitude": 55.0 + 0.01 * i,
                "longitude": 10.0 + 0.01 * i + 0.5 * v,
                "speed": 8.0,
                "heading": 90.0,
            })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# replay ordering + chronological
# ---------------------------------------------------------------------------

def test_replay_chronological_order():
    sim = AISReplaySimulator.from_dataframe(_clean_df(), n_ticks=5)
    seen_ts = []
    while not sim.finished:
        for obs in sim.advance():
            seen_ts.append(obs["timestamp"])
    assert seen_ts == sorted(seen_ts), "replay must emit observations in time order"
    print("replay chronological order  OK")


# ---------------------------------------------------------------------------
# state updates
# ---------------------------------------------------------------------------

def test_state_updates():
    sim = AISReplaySimulator.from_dataframe(_clean_df(), n_ticks=5)
    sim.advance()
    state = sim.current_state()
    assert len(state) >= 1
    for vid, obs in state.items():
        assert "latitude" in obs and "timestamp" in obs
    # recent track grows but is capped
    sim.advance(); sim.advance()
    for vid in sim.active_vessel_ids():
        rt = sim.recent_track(vid)
        assert len(rt) <= 20
    print("state updates + recent track  OK")


# ---------------------------------------------------------------------------
# replay completion
# ---------------------------------------------------------------------------

def test_replay_completion():
    sim = AISReplaySimulator.from_dataframe(_clean_df(), n_ticks=4)
    steps = 0
    while not sim.finished and steps < 100:
        sim.advance(); steps += 1
    assert sim.finished
    assert sim.advance() == [], "advancing past the end yields nothing"
    assert abs(sim.progress - 1.0) < 1e-9
    print("replay completion  OK")


# ---------------------------------------------------------------------------
# alert triggering (HIGH risk + zone)
# ---------------------------------------------------------------------------

def _risk(vid, level="HIGH", geo=1.0, score=70, gap=0, gap_c=0.0,
          beh=0.0, temp=0.0):
    return {
        "vessel_id": vid, "risk_level": level, "overall_score": score,
        "risk_score": score, "zone_status": "INSIDE Zone A",
        "geofence_component": geo, "ais_gap_component": gap_c,
        "behaviour_component": beh, "temporal_component": temp,
        "ais_gap_minutes": gap,
    }


def test_alert_triggering():
    ae = AlertEngine(cooldown_ticks=5)
    # Single strong signal (zone) + HIGH level → HIGH_RISK + ZONE_ENTRY, no COMBINED.
    results = {"V1": _risk("V1", "HIGH", geo=1.0, score=70)}
    types = {a["alert_type"] for a in ae.evaluate(results, tick=0)}
    assert "HIGH_RISK" in types
    assert "ZONE_ENTRY" in types
    assert "COMBINED" not in types, "single strong signal must not trigger COMBINED"

    # Two strong signals (zone + behaviour) → COMBINED fires.
    ae2 = AlertEngine(cooldown_ticks=5)
    combo = {"V2": _risk("V2", "HIGH", geo=1.0, score=80, beh=0.9)}
    types2 = {a["alert_type"] for a in ae2.evaluate(combo, tick=0)}
    assert "COMBINED" in types2, "zone + behaviour (2 signals) must trigger COMBINED"
    assert "BEHAVIOUR_ANOMALY" in types2
    print(f"alert triggering  OK (single={sorted(types)}, combo={sorted(types2)})")


# ---------------------------------------------------------------------------
# cooldown / dedup suppression
# ---------------------------------------------------------------------------

def test_alert_cooldown_dedup():
    ae = AlertEngine(cooldown_ticks=5)
    results = {"V1": _risk("V1", "HIGH", geo=1.0, score=70)}
    first = ae.evaluate(results, tick=0)
    assert len(first) > 0
    # Same tick+1 → within cooldown → suppressed
    again = ae.evaluate(results, tick=1)
    assert again == [], "same alert within cooldown must be suppressed"
    # After cooldown window → may fire again
    later = ae.evaluate(results, tick=6)
    assert len(later) > 0, "alert should be allowed again after cooldown"
    print("alert cooldown / dedup  OK")


# ---------------------------------------------------------------------------
# alert DB persistence
# ---------------------------------------------------------------------------

def test_alert_db_persistence():
    fd, db = tempfile.mkstemp(suffix=".db"); os.close(fd); os.remove(db)
    try:
        database.initialize_database(db)
        ae = AlertEngine(cooldown_ticks=5)
        alerts = ae.evaluate({"V1": _risk("V1", "HIGH")}, tick=0)
        n = database.insert_alerts(alerts, db_path=db)
        assert n == len(alerts) and n > 0
        summary = database.get_history_summary(db)
        assert summary["total_alerts"] == n
        recent = database.get_recent_alerts(limit=10, db_path=db)
        assert recent and recent[0]["vessel_id"] == "V1"
        print(f"alert DB persistence  OK ({n} alerts)")
    finally:
        os.remove(db)


# ---------------------------------------------------------------------------
# low-risk vessel produces no alerts
# ---------------------------------------------------------------------------

def test_low_risk_no_alerts():
    ae = AlertEngine()
    results = {"V9": _risk("V9", "LOW", geo=0.0, score=5)}
    results["V9"]["zone_status"] = "Open Waters"
    assert ae.evaluate(results, tick=0) == []
    print("low-risk vessel -> no alerts  OK")


if __name__ == "__main__":
    tests = [
        test_replay_chronological_order, test_state_updates,
        test_replay_completion, test_alert_triggering,
        test_alert_cooldown_dedup, test_alert_db_persistence,
        test_low_risk_no_alerts,
    ]
    p = 0
    for t in tests:
        t(); p += 1
    print(f"\n=== {p}/{len(tests)} TESTS PASSED ===")
