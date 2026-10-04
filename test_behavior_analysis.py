"""
test_behavior_analysis.py
Tests for the Round 3 M4 trajectory & behaviour analysis (behavior_analysis.py).

Run:
    python test_behavior_analysis.py
or:
    python -m pytest test_behavior_analysis.py -q
"""

import math

import pandas as pd

from behavior_analysis import (
    haversine_km, analyze_trajectory, analyze_all,
    BEHAVIOR_FEATURE_COLUMNS, behavior_feature_row,
    SIGNIFICANT_GAP_MINUTES,
)


def _df(rows):
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 1. Haversine distance
# ---------------------------------------------------------------------------

def test_1_haversine():
    # ~111.19 km per degree of latitude near the equator
    d = haversine_km(0.0, 0.0, 1.0, 0.0)
    assert 110.0 < d < 112.0, d
    assert haversine_km(55.0, 10.0, 55.0, 10.0) == 0.0
    print("1. haversine distance  OK")


# ---------------------------------------------------------------------------
# 2. Straight-line trajectory
# ---------------------------------------------------------------------------

def test_2_straight_line():
    b = analyze_trajectory([(55.0, 10.0), (55.0, 10.1), (55.0, 10.2)])
    # Monotonic east movement — distance ≈ 2 segments of ~0.1° lon at lat 55
    assert b["distance_travelled_km"] > 10.0
    assert b["position_count"] == 3
    print(f"2. straight-line trajectory  OK (dist={b['distance_travelled_km']}km)")


# ---------------------------------------------------------------------------
# 3. Multi-point trajectory distance (sum of segments)
# ---------------------------------------------------------------------------

def test_3_multipoint_distance():
    pts = [(55.0, 10.0), (55.1, 10.0), (55.2, 10.0)]
    b = analyze_trajectory(pts)
    expected = (haversine_km(55.0, 10.0, 55.1, 10.0)
                + haversine_km(55.1, 10.0, 55.2, 10.0))
    assert abs(b["distance_travelled_km"] - expected) < 0.01
    print("3. multi-point distance sums segments  OK")


# ---------------------------------------------------------------------------
# 4. Speed statistics
# ---------------------------------------------------------------------------

def test_4_speed_stats():
    df = _df([
        {"latitude": 55.0, "longitude": 10.0, "speed": 4.0, "heading": 90, "timestamp": "2025-02-27 00:00:00"},
        {"latitude": 55.0, "longitude": 10.1, "speed": 8.0, "heading": 90, "timestamp": "2025-02-27 00:10:00"},
        {"latitude": 55.0, "longitude": 10.2, "speed": 12.0, "heading": 90, "timestamp": "2025-02-27 00:20:00"},
    ])
    b = analyze_trajectory(df)
    assert b["avg_speed"] == 8.0
    assert b["max_speed"] == 12.0
    assert b["speed_std"] > 0
    print("4. speed statistics  OK")


# ---------------------------------------------------------------------------
# 5 + 6. Heading normalization + change
# ---------------------------------------------------------------------------

def test_5_6_heading():
    # 350 -> 10 should be a +20° change (not 340)
    df = _df([
        {"latitude": 55.0, "longitude": 10.0, "speed": 5, "heading": 350, "timestamp": "2025-02-27 00:00:00"},
        {"latitude": 55.0, "longitude": 10.1, "speed": 5, "heading": 10,  "timestamp": "2025-02-27 00:10:00"},
    ])
    b = analyze_trajectory(df)
    assert abs(b["mean_heading_change"] - 20.0) < 1e-6, b["mean_heading_change"]
    assert abs(b["max_heading_change"] - 20.0) < 1e-6
    print("5+6. heading normalization + change  OK")


# ---------------------------------------------------------------------------
# 7. Stationary ratio
# ---------------------------------------------------------------------------

def test_7_stationary_ratio():
    df = _df([
        {"latitude": 55.0, "longitude": 10.0, "speed": 0.2, "heading": 90, "timestamp": "2025-02-27 00:00:00"},
        {"latitude": 55.0, "longitude": 10.0, "speed": 0.5, "heading": 90, "timestamp": "2025-02-27 00:10:00"},
        {"latitude": 55.0, "longitude": 10.1, "speed": 9.0, "heading": 90, "timestamp": "2025-02-27 00:20:00"},
        {"latitude": 55.0, "longitude": 10.2, "speed": 10.0, "heading": 90, "timestamp": "2025-02-27 00:30:00"},
    ])
    b = analyze_trajectory(df)
    assert abs(b["stationary_ratio"] - 0.5) < 1e-6, b["stationary_ratio"]
    print("7. stationary ratio  OK")


# ---------------------------------------------------------------------------
# 8. AIS gap detection
# ---------------------------------------------------------------------------

def test_8_ais_gap():
    df = _df([
        {"latitude": 55.0, "longitude": 10.0, "speed": 5, "heading": 90, "timestamp": "2025-02-27 00:00:00"},
        {"latitude": 55.0, "longitude": 10.1, "speed": 5, "heading": 90, "timestamp": "2025-02-27 00:05:00"},  # 5 min
        {"latitude": 55.0, "longitude": 10.2, "speed": 5, "heading": 90, "timestamp": "2025-02-27 01:00:00"},  # 55 min gap
    ])
    b = analyze_trajectory(df)
    assert abs(b["max_ais_gap_minutes"] - 55.0) < 1e-6, b["max_ais_gap_minutes"]
    assert b["significant_gap_count"] == 1  # only the 55-min gap exceeds 30
    print("8. AIS gap detection  OK")


# ---------------------------------------------------------------------------
# 9. Trajectory duration
# ---------------------------------------------------------------------------

def test_9_duration():
    df = _df([
        {"latitude": 55.0, "longitude": 10.0, "speed": 5, "heading": 90, "timestamp": "2025-02-27 00:00:00"},
        {"latitude": 55.0, "longitude": 10.1, "speed": 5, "heading": 90, "timestamp": "2025-02-27 02:00:00"},
    ])
    b = analyze_trajectory(df)
    assert abs(b["trajectory_duration_minutes"] - 120.0) < 1e-6
    print("9. trajectory duration  OK")


# ---------------------------------------------------------------------------
# 10. Loitering calculation
# ---------------------------------------------------------------------------

def test_10_loitering():
    # Tight circle, slow speed -> high loitering score
    import numpy as np
    circ = []
    for i in range(12):
        ang = 2 * math.pi * i / 12
        circ.append({"latitude": 55.0 + 0.01 * math.sin(ang),
                     "longitude": 10.0 + 0.01 * math.cos(ang),
                     "speed": 0.5, "heading": (ang * 57.3) % 360,
                     "timestamp": f"2025-02-27 00:{i*5:02d}:00"})
    loiter = analyze_trajectory(_df(circ))["loitering_score"]
    # Straight fast transit -> low loitering score
    line = [{"latitude": 55.0, "longitude": 10.0 + 0.1 * i, "speed": 12.0,
             "heading": 90, "timestamp": f"2025-02-27 00:{i*5:02d}:00"} for i in range(6)]
    transit = analyze_trajectory(_df(line))["loitering_score"]
    assert loiter > transit, (loiter, transit)
    assert loiter > 0.5, loiter
    print(f"10. loitering (circle={loiter:.2f} > transit={transit:.2f})  OK")


# ---------------------------------------------------------------------------
# 11. Behaviour classification
# ---------------------------------------------------------------------------

def test_11_classification():
    # Clean transit
    line = [{"latitude": 55.0, "longitude": 10.0 + 0.1 * i, "speed": 11.0,
             "heading": 90, "timestamp": f"2025-02-27 00:{i*5:02d}:00"} for i in range(6)]
    labels = analyze_trajectory(_df(line))["behavior_labels"]
    assert "NORMAL_TRANSIT" in labels, labels
    # Loitering + gap -> multi-label incl MIXED_SUSPICIOUS
    susp = _df([
        {"latitude": 55.0, "longitude": 10.0, "speed": 0.3, "heading": 90, "timestamp": "2025-02-27 00:00:00"},
        {"latitude": 55.001, "longitude": 10.001, "speed": 0.4, "heading": 95, "timestamp": "2025-02-27 00:05:00"},
        {"latitude": 55.0, "longitude": 10.0, "speed": 0.2, "heading": 90, "timestamp": "2025-02-27 01:00:00"},  # 55-min gap
    ])
    slabels = analyze_trajectory(susp)["behavior_labels"]
    assert "AIS_GAP" in slabels
    assert any(l in slabels for l in ("LOITERING", "SLOW_MOVEMENT"))
    print(f"11. classification  OK (transit={labels}, suspicious={slabels})")


# ---------------------------------------------------------------------------
# 12. Single-position vessel
# ---------------------------------------------------------------------------

def test_12_single_position():
    b = analyze_trajectory([(55.0, 10.0)])
    assert b["position_count"] == 1
    assert b["distance_travelled_km"] == 0.0
    assert b["behavior_labels"] == ["INSUFFICIENT_DATA"]
    # Must still contain all numeric feature columns
    row = behavior_feature_row(b)
    assert set(BEHAVIOR_FEATURE_COLUMNS).issubset(row.keys())
    print("12. single-position vessel  OK")


# ---------------------------------------------------------------------------
# 13. Missing speed / heading handling
# ---------------------------------------------------------------------------

def test_13_missing_speed_heading():
    df = _df([
        {"latitude": 55.0, "longitude": 10.0, "speed": None, "heading": None, "timestamp": "2025-02-27 00:00:00"},
        {"latitude": 55.0, "longitude": 10.1, "speed": None, "heading": None, "timestamp": "2025-02-27 00:10:00"},
    ])
    b = analyze_trajectory(df)  # must not crash
    assert b["avg_speed"] == 0.0
    assert b["mean_heading_change"] == 0.0
    assert b["distance_travelled_km"] > 0.0  # distance still works from coords
    print("13. missing speed/heading handled  OK")


# ---------------------------------------------------------------------------
# 14. Empty trajectory handling
# ---------------------------------------------------------------------------

def test_14_empty_trajectory():
    b = analyze_trajectory([])
    assert b["position_count"] == 0
    assert b["distance_travelled_km"] == 0.0
    b2 = analyze_trajectory(pd.DataFrame(columns=["latitude", "longitude", "speed", "heading", "timestamp"]))
    assert b2["position_count"] == 0
    # analyze_all tolerates mixed/empty
    allres = analyze_all({"A": [], "B": [(55.0, 10.0), (55.0, 10.1)]})
    assert allres["A"]["position_count"] == 0
    assert allres["B"]["position_count"] == 2
    print("14. empty trajectory handled  OK")


# ---------------------------------------------------------------------------
# Extra: duplicate timestamps / zero-duration must not divide by zero
# ---------------------------------------------------------------------------

def test_zero_duration_safe():
    df = _df([
        {"latitude": 55.0, "longitude": 10.0, "speed": 5, "heading": 90, "timestamp": "2025-02-27 00:00:00"},
        {"latitude": 55.0, "longitude": 10.1, "speed": 5, "heading": 90, "timestamp": "2025-02-27 00:00:00"},  # dup ts
    ])
    b = analyze_trajectory(df)  # must not raise
    assert b["position_count"] == 2
    print("    duplicate-timestamp / zero-duration safe  OK")


if __name__ == "__main__":
    tests = [
        test_1_haversine, test_2_straight_line, test_3_multipoint_distance,
        test_4_speed_stats, test_5_6_heading, test_7_stationary_ratio,
        test_8_ais_gap, test_9_duration, test_10_loitering,
        test_11_classification, test_12_single_position,
        test_13_missing_speed_heading, test_14_empty_trajectory,
        test_zero_duration_safe,
    ]
    passed = 0
    for t in tests:
        t()
        passed += 1
    print(f"\n=== {passed}/{len(tests)} TESTS PASSED ===")
