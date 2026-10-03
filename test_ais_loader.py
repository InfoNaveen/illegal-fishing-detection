"""
test_ais_loader.py
Focused tests for the Round 3 M1 historical AIS ingestion layer.

Run with:
    python -m pytest test_ais_loader.py -v
or, without pytest installed:
    python test_ais_loader.py
"""

import os
import tempfile

import pandas as pd

from ais_loader import (
    load_vessel_dataframe,
    build_trajectories_from_ais,
    summarize_latest_positions,
    AISColumnError,
    AISFileError,
    DEFAULT_AIS_CSV_PATH,
)

HERE = os.path.dirname(os.path.abspath(__file__))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_csv(text: str) -> str:
    """Write CSV text to a temp file and return its path."""
    fd, path = tempfile.mkstemp(suffix=".csv")
    with os.fdopen(fd, "w", newline="") as f:
        f.write(text)
    return path


# ---------------------------------------------------------------------------
# A. Valid AIS CSV loads correctly
# ---------------------------------------------------------------------------

def test_A_valid_csv_loads():
    csv = (
        "MMSI,BaseDateTime,LAT,LON,SOG,COG\n"
        "A1,2024-01-01T00:00:00,12.5,80.3,2.1,185\n"
        "A1,2024-01-01T00:10:00,12.6,80.4,2.0,180\n"
        "A2,2024-01-01T00:00:00,13.0,79.9,1.8,210\n"
    )
    path = _write_csv(csv)
    try:
        df = load_vessel_dataframe(path)
        assert len(df) == 3
        for col in ["vessel_id", "latitude", "longitude", "speed", "heading", "timestamp"]:
            assert col in df.columns, f"missing normalized column {col}"
        assert set(df["vessel_id"]) == {"A1", "A2"}
    finally:
        os.remove(path)
    print("A. valid csv loads correctly  OK")


# ---------------------------------------------------------------------------
# B. Common source column names are normalized
# ---------------------------------------------------------------------------

def test_B_column_normalization():
    # Use a different but recognised alias set: lon/course/time
    csv = (
        "id,time,latitude,lon,speed,course\n"
        "B1,2024-01-01T00:00:00,10.0,70.0,5.0,100\n"
    )
    path = _write_csv(csv)
    try:
        df = load_vessel_dataframe(path)
        assert df.iloc[0]["vessel_id"] == "B1"
        assert df.iloc[0]["latitude"] == 10.0
        assert df.iloc[0]["longitude"] == 70.0
        assert df.iloc[0]["speed"] == 5.0
        assert df.iloc[0]["heading"] == 100.0
    finally:
        os.remove(path)
    print("B. column normalization  OK")


# ---------------------------------------------------------------------------
# C. Missing required columns produce a clear error
# ---------------------------------------------------------------------------

def test_C_missing_required_column():
    # No latitude alias present
    csv = (
        "MMSI,BaseDateTime,LON,SOG,COG\n"
        "C1,2024-01-01T00:00:00,80.3,2.1,185\n"
    )
    path = _write_csv(csv)
    try:
        raised = False
        try:
            load_vessel_dataframe(path)
        except AISColumnError as e:
            raised = True
            assert "latitude" in str(e), "error must name the missing column"
        assert raised, "AISColumnError was not raised for missing latitude"
    finally:
        os.remove(path)
    print("C. missing required column raises clear error  OK")


# ---------------------------------------------------------------------------
# D. Invalid coordinates are rejected/filtered
# ---------------------------------------------------------------------------

def test_D_invalid_coordinates_filtered():
    csv = (
        "MMSI,BaseDateTime,LAT,LON,SOG,COG\n"
        "D1,2024-01-01T00:00:00,12.5,80.3,2.1,185\n"       # valid
        "D1,2024-01-01T00:10:00,999.0,80.4,2.0,180\n"      # bad lat
        "D1,2024-01-01T00:20:00,12.7,500.0,2.0,180\n"      # bad lon
        "D1,2024-01-01T00:30:00,,80.5,2.0,180\n"           # missing lat
    )
    path = _write_csv(csv)
    try:
        df = load_vessel_dataframe(path)
        assert len(df) == 1, f"expected 1 valid row, got {len(df)}"
        assert df.attrs.get("dropped_rows") == 3
    finally:
        os.remove(path)
    print("D. invalid coordinates filtered  OK")


# ---------------------------------------------------------------------------
# E. Trajectory dictionary is generated correctly
# ---------------------------------------------------------------------------

def test_E_trajectory_dict():
    csv = (
        "MMSI,BaseDateTime,LAT,LON,SOG,COG\n"
        "E1,2024-01-01T00:00:00,12.5,80.3,2.1,185\n"
        "E1,2024-01-01T00:10:00,12.6,80.4,2.0,180\n"
        "E2,2024-01-01T00:00:00,13.0,79.9,1.8,210\n"
    )
    path = _write_csv(csv)
    try:
        df = load_vessel_dataframe(path)
        trajs = build_trajectories_from_ais(df)
        assert set(trajs.keys()) == {"E1", "E2"}
        assert trajs["E1"] == [(12.5, 80.3), (12.6, 80.4)]
        assert trajs["E2"] == [(13.0, 79.9)]
    finally:
        os.remove(path)
    print("E. trajectory dict generated correctly  OK")


# ---------------------------------------------------------------------------
# F. Vessel trajectories are chronologically ordered
# ---------------------------------------------------------------------------

def test_F_chronological_order():
    # Deliberately out of chronological order in the file
    csv = (
        "MMSI,BaseDateTime,LAT,LON,SOG,COG\n"
        "F1,2024-01-01T00:20:00,12.7,80.5,2.0,180\n"   # latest
        "F1,2024-01-01T00:00:00,12.5,80.3,2.1,185\n"   # earliest
        "F1,2024-01-01T00:10:00,12.6,80.4,2.0,180\n"   # middle
    )
    path = _write_csv(csv)
    try:
        df = load_vessel_dataframe(path)
        trajs = build_trajectories_from_ais(df)
        assert trajs["F1"] == [(12.5, 80.3), (12.6, 80.4), (12.7, 80.5)], \
            "trajectory not chronologically ordered"

        # summarize_latest_positions must return the LAST chronological point
        latest = summarize_latest_positions(df)
        assert len(latest) == 1
        assert latest.iloc[0]["latitude"] == 12.7
        assert latest.iloc[0]["behavior"] == "AIS Track"
    finally:
        os.remove(path)
    print("F. chronological ordering  OK")


# ---------------------------------------------------------------------------
# G. Existing simulated pipeline still works
# ---------------------------------------------------------------------------

def test_G_simulated_pipeline_intact():
    from data_generator import generate_vessel_dataframe, build_all_trajectories, get_restricted_zones
    from geofencing import run_geofencing
    from feature_engineering import build_feature_matrix
    from anomaly_detector import run_anomaly_detection
    from risk_engine import run_risk_engine, generate_alerts

    zones  = get_restricted_zones()
    base   = generate_vessel_dataframe()
    assert len(base) == 18, "simulated fleet must still be 18 vessels"
    trajs  = build_all_trajectories(base)
    geos   = run_geofencing(base, zones)
    geomap = {g["vessel_id"]: g for g in geos}
    feats  = build_feature_matrix(base, trajs, geomap)
    anom   = run_anomaly_detection(feats)
    risk   = run_risk_engine(base["vessel_id"].tolist(), geomap, feats, anom)
    alerts = generate_alerts(risk)

    # Known Round 2 test vessels must still be HIGH risk inside zones
    for vid in ["V102", "V087", "V215"]:
        assert risk[vid]["risk_level"] == "HIGH", f"{vid} should be HIGH"
    assert len(alerts) > 0
    print("G. simulated pipeline intact (18 vessels, V102/V087/V215 HIGH)  OK")


# ---------------------------------------------------------------------------
# H. Missing file raises AISFileError
# ---------------------------------------------------------------------------

def test_H_missing_file():
    raised = False
    try:
        load_vessel_dataframe(os.path.join("data", "does_not_exist_12345.csv"))
    except AISFileError as e:
        raised = True
        assert "not found" in str(e).lower()
    assert raised, "AISFileError not raised for missing file"
    print("H. missing file raises AISFileError  OK")


# ---------------------------------------------------------------------------
# I. Bundled sample fixture loads end-to-end through AIS path
# ---------------------------------------------------------------------------

def test_I_sample_fixture_end_to_end():
    sample = os.path.join(HERE, DEFAULT_AIS_CSV_PATH)
    if not os.path.exists(sample):
        print("I. sample fixture not present — skipped")
        return
    df     = load_vessel_dataframe(sample)
    trajs  = build_trajectories_from_ais(df)
    latest = summarize_latest_positions(df)
    assert len(latest) == len(trajs)
    assert all(c in latest.columns for c in
               ["vessel_id", "latitude", "longitude", "speed", "heading", "behavior"])
    print(f"I. sample fixture end-to-end  OK ({len(latest)} vessels)")


# ---------------------------------------------------------------------------
# Manual runner (no pytest required)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    tests = [
        test_A_valid_csv_loads,
        test_B_column_normalization,
        test_C_missing_required_column,
        test_D_invalid_coordinates_filtered,
        test_E_trajectory_dict,
        test_F_chronological_order,
        test_G_simulated_pipeline_intact,
        test_H_missing_file,
        test_I_sample_fixture_end_to_end,
    ]
    passed = 0
    for t in tests:
        t()
        passed += 1
    print(f"\n=== {passed}/{len(tests)} TESTS PASSED ===")
