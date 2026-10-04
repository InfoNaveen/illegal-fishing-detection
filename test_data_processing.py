"""
test_data_processing.py
Tests for the Round 3 M2 AIS cleaning layer (data_processing.py)
and the source-aware zone configuration (zones.py).

Run:
    python test_data_processing.py
or:
    python -m pytest test_data_processing.py -v
"""

import os
import numpy as np
import pandas as pd

from data_processing import clean_ais_dataframe
from zones import get_zones, get_region_meta

HERE = os.path.dirname(os.path.abspath(__file__))


def _df(rows):
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 1 + 2. Timestamp parsing (including DD/MM/YYYY day-first)
# ---------------------------------------------------------------------------

def test_1_2_timestamp_dayfirst_parsing():
    df = _df([
        {"vessel_id": "A", "latitude": 55.0, "longitude": 10.0,
         "speed": 5.0, "heading": 90, "timestamp": "27/02/2025 08:30:00"},
        {"vessel_id": "A", "latitude": 55.1, "longitude": 10.1,
         "speed": 5.0, "heading": 90, "timestamp": "27/02/2025 09:30:00"},
    ])
    clean, rep = clean_ais_dataframe(df)
    assert pd.api.types.is_datetime64_any_dtype(clean["timestamp"])
    # 27/02 must parse as 27 Feb (day-first), not fail as month=27
    assert clean["timestamp"].iloc[0].day == 27
    assert clean["timestamp"].iloc[0].month == 2
    assert rep["missing_timestamp_rows"] == 0
    print("1+2. timestamp dayfirst parsing  OK")


def test_timestamp_unparseable_dropped():
    df = _df([
        {"vessel_id": "A", "latitude": 55.0, "longitude": 10.0,
         "speed": 5.0, "heading": 90, "timestamp": "not-a-date"},
        {"vessel_id": "A", "latitude": 55.1, "longitude": 10.1,
         "speed": 5.0, "heading": 90, "timestamp": "27/02/2025 09:30:00"},
    ])
    clean, rep = clean_ais_dataframe(df)
    assert len(clean) == 1
    assert rep["missing_timestamp_rows"] == 1
    print("    unparseable timestamp dropped  OK")


# ---------------------------------------------------------------------------
# 3. Invalid / missing coordinate removal
# ---------------------------------------------------------------------------

def test_3_invalid_coordinates_removed():
    df = _df([
        {"vessel_id": "A", "latitude": 55.0,  "longitude": 10.0, "speed": 5, "heading": 90, "timestamp": "27/02/2025 00:00:00"},
        {"vessel_id": "A", "latitude": 95.0,  "longitude": 10.0, "speed": 5, "heading": 90, "timestamp": "27/02/2025 00:10:00"},  # bad lat
        {"vessel_id": "A", "latitude": 55.0,  "longitude": 190.0,"speed": 5, "heading": 90, "timestamp": "27/02/2025 00:20:00"},  # bad lon
    ])
    clean, rep = clean_ais_dataframe(df)
    assert len(clean) == 1
    assert rep["invalid_coordinates"] == 2
    print("3. invalid coordinates removed  OK")


def test_4_missing_coordinates_removed():
    df = _df([
        {"vessel_id": "A", "latitude": 55.0, "longitude": 10.0, "speed": 5, "heading": 90, "timestamp": "27/02/2025 00:00:00"},
        {"vessel_id": "A", "latitude": None, "longitude": 10.0, "speed": 5, "heading": 90, "timestamp": "27/02/2025 00:10:00"},
        {"vessel_id": "",  "latitude": 55.0, "longitude": 10.0, "speed": 5, "heading": 90, "timestamp": "27/02/2025 00:20:00"},  # missing id
    ])
    clean, rep = clean_ais_dataframe(df)
    assert len(clean) == 1
    assert rep["missing_coordinate_rows"] >= 1
    print("4. missing coordinates / id removed  OK")


# ---------------------------------------------------------------------------
# 5. Negative speed removal (keep NaN and high positives)
# ---------------------------------------------------------------------------

def test_5_negative_speed_removed():
    df = _df([
        {"vessel_id": "A", "latitude": 55.0, "longitude": 10.0, "speed": 5.0,  "heading": 90, "timestamp": "27/02/2025 00:00:00"},
        {"vessel_id": "A", "latitude": 55.1, "longitude": 10.1, "speed": -2.0, "heading": 90, "timestamp": "27/02/2025 00:10:00"},  # negative
        {"vessel_id": "A", "latitude": 55.2, "longitude": 10.2, "speed": 40.0, "heading": 90, "timestamp": "27/02/2025 00:20:00"},  # high but valid -> KEEP
    ])
    clean, rep = clean_ais_dataframe(df)
    assert rep["invalid_speed_rows"] == 1
    assert 40.0 in clean["speed"].values, "high-but-valid speed must be kept"
    print("5. negative speed removed, high speed kept  OK")


# ---------------------------------------------------------------------------
# 6. Exact duplicate removal (keep stationary across different timestamps)
# ---------------------------------------------------------------------------

def test_6_exact_duplicates_removed():
    df = _df([
        {"vessel_id": "A", "latitude": 55.0, "longitude": 10.0, "speed": 0, "heading": 90, "timestamp": "27/02/2025 00:00:00"},
        {"vessel_id": "A", "latitude": 55.0, "longitude": 10.0, "speed": 0, "heading": 90, "timestamp": "27/02/2025 00:00:00"},  # exact dup
        {"vessel_id": "A", "latitude": 55.0, "longitude": 10.0, "speed": 0, "heading": 90, "timestamp": "27/02/2025 00:10:00"},  # stationary, diff time -> KEEP
    ])
    clean, rep = clean_ais_dataframe(df)
    assert rep["duplicate_rows"] == 1
    assert len(clean) == 2, "stationary vessel at a different timestamp must be kept"
    print("6. exact duplicates removed, stationary kept  OK")


# ---------------------------------------------------------------------------
# 7. Chronological ordering per vessel
# ---------------------------------------------------------------------------

def test_7_chronological_ordering():
    df = _df([
        {"vessel_id": "A", "latitude": 55.2, "longitude": 10.2, "speed": 5, "heading": 90, "timestamp": "27/02/2025 00:20:00"},
        {"vessel_id": "A", "latitude": 55.0, "longitude": 10.0, "speed": 5, "heading": 90, "timestamp": "27/02/2025 00:00:00"},
        {"vessel_id": "A", "latitude": 55.1, "longitude": 10.1, "speed": 5, "heading": 90, "timestamp": "27/02/2025 00:10:00"},
    ])
    clean, _ = clean_ais_dataframe(df)
    ts = list(clean["timestamp"])
    assert ts == sorted(ts), "rows must be chronologically ordered"
    assert clean["latitude"].iloc[0] == 55.0
    print("7. chronological ordering  OK")


# ---------------------------------------------------------------------------
# 8. Quality report completeness
# ---------------------------------------------------------------------------

def test_8_quality_report():
    df = _df([
        {"vessel_id": "A", "latitude": 55.0, "longitude": 10.0, "speed": 5, "heading": 90, "timestamp": "27/02/2025 00:00:00"},
        {"vessel_id": "B", "latitude": 56.0, "longitude": 11.0, "speed": 5, "heading": 90, "timestamp": "27/02/2025 00:00:00"},
    ])
    _, rep = clean_ais_dataframe(df)
    for key in ["input_rows", "output_rows", "rows_removed",
                "missing_timestamp_rows", "missing_coordinate_rows",
                "invalid_coordinates", "invalid_speed_rows", "duplicate_rows",
                "unique_vessels", "speed_defaults_filled", "heading_defaults_filled"]:
        assert key in rep, f"report missing key {key}"
    assert rep["input_rows"] == 2
    assert rep["output_rows"] == 2
    assert rep["unique_vessels"] == 2
    print("8. quality report complete  OK")


def test_heading_zero_and_sentinel():
    df = _df([
        {"vessel_id": "A", "latitude": 55.0, "longitude": 10.0, "speed": 5, "heading": 0,   "timestamp": "27/02/2025 00:00:00"},   # 0 is valid
        {"vessel_id": "A", "latitude": 55.1, "longitude": 10.1, "speed": 5, "heading": 511, "timestamp": "27/02/2025 00:10:00"},   # sentinel -> NaN -> default
        {"vessel_id": "A", "latitude": 55.2, "longitude": 10.2, "speed": 5, "heading": 370, "timestamp": "27/02/2025 00:20:00"},   # wrap -> 10
    ])
    clean, _ = clean_ais_dataframe(df)
    assert len(clean) == 3, "heading 0 / 511 must not drop rows"
    assert clean["heading"].iloc[0] == 0.0
    assert clean["heading"].iloc[2] == 10.0  # 370 mod 360
    print("    heading 0 valid, 511 sentinel handled, wrap OK  OK")


# ---------------------------------------------------------------------------
# 9. Real AIS sample processing
# ---------------------------------------------------------------------------

def test_9_real_sample_processing():
    from ais_loader import load_vessel_dataframe
    sample = os.path.join(HERE, "data", "ifds_ais_sample.csv")
    if not os.path.exists(sample):
        print("9. real sample not present — skipped")
        return
    raw = load_vessel_dataframe(sample)
    clean, rep = clean_ais_dataframe(raw)
    assert rep["input_rows"] == len(raw)
    assert rep["output_rows"] == len(clean)
    # Real AIS contains many exact-duplicate broadcasts (moored/anchored vessels
    # and redundant feed records). Removing them is correct and expected.
    # The key invariants: every vessel is preserved, nothing is fabricated, and
    # only exact duplicates / invalid rows are removed.
    assert rep["unique_vessels"] == raw["vessel_id"].nunique(), \
        "cleaning must not drop any vessel entirely"
    assert rep["invalid_coordinates"] == 0
    assert rep["missing_coordinate_rows"] == 0
    assert rep["invalid_speed_rows"] == 0
    # rows_removed should be fully accounted for by duplicates here
    assert rep["rows_removed"] == rep["duplicate_rows"]
    assert rep["output_rows"] > 0
    print(f"9. real sample processed  OK "
          f"(in={rep['input_rows']:,} out={rep['output_rows']:,} "
          f"vessels={rep['unique_vessels']:,} removed={rep['rows_removed']:,} "
          f"[all duplicates])")


# ---------------------------------------------------------------------------
# Zone configuration tests
# ---------------------------------------------------------------------------

def test_zones_source_aware():
    sim = get_zones("Simulated")
    ais = get_zones("Historical AIS")
    assert [z["name"] for z in sim] == ["Restricted Zone A", "Restricted Zone B", "Restricted Zone C"]
    assert all("Monitoring Zone" in z["name"] for z in ais)
    assert all(z["zone_type"] == "monitoring" for z in ais)
    assert get_region_meta("Simulated")["region"] == "Bay of Bengal"
    assert get_region_meta("Historical AIS")["region"] == "Danish Waters"
    print("Z. source-aware zones  OK")


# ---------------------------------------------------------------------------
# Simulated-mode regression
# ---------------------------------------------------------------------------

def test_simulated_regression():
    from data_generator import generate_vessel_dataframe, build_all_trajectories
    from geofencing import run_geofencing
    from feature_engineering import build_feature_matrix
    from anomaly_detector import run_anomaly_detection
    from risk_engine import run_risk_engine

    zones  = get_zones("Simulated")
    base   = generate_vessel_dataframe()
    assert len(base) == 18
    trajs  = build_all_trajectories(base)
    geos   = run_geofencing(base, zones)
    geomap = {g["vessel_id"]: g for g in geos}
    feats  = build_feature_matrix(base, trajs, geomap)
    anom   = run_anomaly_detection(feats)
    risk   = run_risk_engine(base["vessel_id"].tolist(), geomap, feats, anom)
    for vid in ["V102", "V087", "V215"]:
        assert risk[vid]["risk_level"] == "HIGH", f"{vid} must stay HIGH"
    # Simulated geofencing must still detect vessels inside Bay of Bengal zones
    assert sum(1 for g in geos if g["inside"]) >= 3
    print("R. simulated regression (18 vessels, V102/V087/V215 HIGH, zones hit)  OK")


if __name__ == "__main__":
    tests = [
        test_1_2_timestamp_dayfirst_parsing,
        test_timestamp_unparseable_dropped,
        test_3_invalid_coordinates_removed,
        test_4_missing_coordinates_removed,
        test_5_negative_speed_removed,
        test_6_exact_duplicates_removed,
        test_7_chronological_ordering,
        test_8_quality_report,
        test_heading_zero_and_sentinel,
        test_9_real_sample_processing,
        test_zones_source_aware,
        test_simulated_regression,
    ]
    passed = 0
    for t in tests:
        t()
        passed += 1
    print(f"\n=== {passed}/{len(tests)} TESTS PASSED ===")
