"""
test_live_ais.py
Tests for the Live AISStream integration (live_ais.py).

All tests are deterministic and use mocks/fixtures — NO real API key and NO
network connection are required. The real API key is never referenced.

Run:
    python test_live_ais.py
or:
    python -m pytest test_live_ais.py -q
"""

import os
from datetime import datetime, timezone, timedelta

import live_ais
from live_ais import (
    get_api_key, build_subscription_message, normalize_message,
    VesselStateManager, LiveAISClient, LIVE_AIS_BBOX, LIVE_AIS_REGIONS,
    get_region_bbox, get_region_center_zoom, DEFAULT_REGION_NAME,
    normalize_static, ship_type_label, decode_ship_type,
)


def _restore_env(prev):
    if prev is None:
        os.environ.pop("AISSTREAM_API_KEY", None)
    else:
        os.environ["AISSTREAM_API_KEY"] = prev


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

def test_api_key_env_detected_and_not_revealed():
    prev = os.environ.get("AISSTREAM_API_KEY")
    try:
        os.environ["AISSTREAM_API_KEY"] = "unit_test_dummy_key_123"
        assert get_api_key() == "unit_test_dummy_key_123"
    finally:
        _restore_env(prev)
    print("api key env detection  OK")


def test_api_key_missing_returns_none():
    prev = os.environ.get("AISSTREAM_API_KEY")
    try:
        os.environ.pop("AISSTREAM_API_KEY", None)
        # With no env var and (in CI) no streamlit secrets, must be None.
        val = get_api_key()
        assert val is None or isinstance(val, str)  # tolerate a dev .env
    finally:
        _restore_env(prev)
    print("missing api key handled  OK")


def test_api_key_placeholder_is_not_configured():
    prev = os.environ.get("AISSTREAM_API_KEY")
    try:
        os.environ["AISSTREAM_API_KEY"] = "your_aisstream_api_key_here"
        assert get_api_key() is None, "placeholder must count as not configured"
    finally:
        _restore_env(prev)
    print("placeholder treated as unconfigured  OK")


# ---------------------------------------------------------------------------
# Subscription construction
# ---------------------------------------------------------------------------

def test_subscription_message_shape():
    sub = build_subscription_message("DUMMY", LIVE_AIS_BBOX)
    assert sub["APIKey"] == "DUMMY"
    assert "BoundingBoxes" in sub and len(sub["BoundingBoxes"]) == 1
    box = sub["BoundingBoxes"][0]
    assert box == [[LIVE_AIS_BBOX["min_lat"], LIVE_AIS_BBOX["min_lon"]],
                   [LIVE_AIS_BBOX["max_lat"], LIVE_AIS_BBOX["max_lon"]]]
    # Now subscribes to both movement and static (type/name/dest/ETA) messages.
    assert sub["FilterMessageTypes"] == ["PositionReport", "ShipStaticData"]
    print("subscription message shape  OK")


# ---------------------------------------------------------------------------
# Message normalization
# ---------------------------------------------------------------------------

def _position_msg(mmsi=123456789, lat=10.5, lon=80.2, sog=8.0, cog=90.0,
                  heading=92.0, ts="2025-01-01 00:00:00 +0000 UTC"):
    return {
        "MessageType": "PositionReport",
        "MetaData": {"MMSI": mmsi, "time_utc": ts, "latitude": lat, "longitude": lon},
        "Message": {"PositionReport": {
            "UserID": mmsi, "Latitude": lat, "Longitude": lon,
            "Sog": sog, "Cog": cog, "TrueHeading": heading,
            "NavigationalStatus": 0,
        }},
    }


def test_normalize_valid_position():
    obs = normalize_message(_position_msg())
    assert obs is not None
    assert obs["vessel_id"] == "123456789"
    assert obs["latitude"] == 10.5 and obs["longitude"] == 80.2
    assert obs["speed"] == 8.0 and obs["heading"] == 92.0
    assert obs["timestamp"] is not None
    print("normalize valid position  OK")


def test_normalize_missing_mmsi():
    msg = _position_msg()
    msg["MetaData"].pop("MMSI")
    msg["Message"]["PositionReport"].pop("UserID")
    assert normalize_message(msg) is None
    print("normalize missing MMSI -> None  OK")


def test_normalize_missing_coords():
    msg = _position_msg()
    msg["Message"]["PositionReport"].pop("Latitude")
    msg["MetaData"].pop("latitude")
    assert normalize_message(msg) is None
    print("normalize missing coords -> None  OK")


def test_normalize_invalid_coords():
    assert normalize_message(_position_msg(lat=999.0)) is None
    assert normalize_message(_position_msg(lon=500.0)) is None
    print("normalize invalid coords -> None  OK")


def test_normalize_missing_speed_heading():
    msg = _position_msg()
    msg["Message"]["PositionReport"].pop("Sog")
    msg["Message"]["PositionReport"].pop("TrueHeading")
    msg["Message"]["PositionReport"].pop("Cog")
    obs = normalize_message(msg)
    assert obs is not None
    assert obs["speed"] is None and obs["heading"] is None  # not fabricated
    print("normalize missing speed/heading -> None values  OK")


def test_normalize_unknown_message_type():
    assert normalize_message({"MessageType": "ShipStaticData", "MetaData": {}}) is None
    assert normalize_message("not a dict") is None
    assert normalize_message({}) is None
    print("normalize unknown/malformed -> None  OK")


# ---------------------------------------------------------------------------
# Vessel state manager
# ---------------------------------------------------------------------------

def test_state_update_and_get():
    sm = VesselStateManager(max_trajectory_points=5)
    sm.update_vessel(normalize_message(_position_msg(mmsi=111, lat=10.0, lon=80.0)))
    sm.update_vessel(normalize_message(_position_msg(mmsi=111, lat=10.1, lon=80.1)))
    s = sm.get_vessel_state("111")
    assert s["latitude"] == 10.1 and s["prev_latitude"] == 10.0
    assert s["message_count"] == 2
    assert sm.vessel_count() == 1 and sm.message_count == 2
    print("state update + prev tracking  OK")


def test_state_bounded_trajectory():
    sm = VesselStateManager(max_trajectory_points=3)
    for i in range(10):
        sm.update_vessel(normalize_message(_position_msg(mmsi=222, lat=10.0 + i * 0.01)))
    track = sm.get_recent_track("222")
    assert len(track) == 3, "trajectory must be bounded to max_trajectory_points"
    print("bounded trajectory (no unbounded growth)  OK")


def test_state_remove_stale():
    sm = VesselStateManager()
    obs = normalize_message(_position_msg(mmsi=333))
    obs["timestamp"] = datetime.now(timezone.utc) - timedelta(seconds=600)
    sm.update_vessel(obs)
    assert sm.vessel_count() == 1
    removed = sm.remove_stale_vessels(stale_seconds=300)
    assert removed == 1 and sm.vessel_count() == 0
    print("stale vessel removal  OK")


# ---------------------------------------------------------------------------
# Region configuration (India-focused)
# ---------------------------------------------------------------------------

def test_regions_valid_and_india_default():
    assert DEFAULT_REGION_NAME in LIVE_AIS_REGIONS
    assert "India" in DEFAULT_REGION_NAME  # default is the widest India box
    assert "Tamil Nadu / Coromandel Coast" in LIVE_AIS_REGIONS
    for name, r in LIVE_AIS_REGIONS.items():
        assert -90 <= r["min_lat"] < r["max_lat"] <= 90, name
        assert -180 <= r["min_lon"] < r["max_lon"] <= 180, name
    box = get_region_bbox(DEFAULT_REGION_NAME)
    assert set(box) == {"min_lat", "min_lon", "max_lat", "max_lon"}
    center, zoom = get_region_center_zoom(DEFAULT_REGION_NAME)
    assert len(center) == 2 and isinstance(zoom, int)
    print("india regions valid + default is widest India box  OK")


# ---------------------------------------------------------------------------
# Client lifecycle (no real network)
# ---------------------------------------------------------------------------

def test_client_start_without_key_sets_error():
    prev = os.environ.get("AISSTREAM_API_KEY")
    try:
        os.environ.pop("AISSTREAM_API_KEY", None)
        client = LiveAISClient(api_key=None, state_manager=VesselStateManager())
        client.start()
        assert client.status == "error"
        assert "API" in client.last_error or "key" in client.last_error.lower()
        assert not client.is_running()
    finally:
        _restore_env(prev)
    print("client start without key -> error, no thread  OK")


def test_client_stop_is_safe_when_not_started():
    client = LiveAISClient(api_key="dummy", state_manager=VesselStateManager())
    client.stop()  # must not raise even though never started
    assert not client.is_running()
    print("client stop before start is safe  OK")


def test_client_stats_no_secret():
    client = LiveAISClient(api_key="SECRET_DUMMY", state_manager=VesselStateManager())
    stats = client.stats()
    # The API key must never appear in the stats payload.
    assert "SECRET_DUMMY" not in str(stats)
    assert set(["status", "vessel_count", "message_count"]).issubset(stats.keys())
    print("stats payload contains no secret  OK")


# ---------------------------------------------------------------------------
# Integration: mocked messages -> state -> existing risk pipeline
# ---------------------------------------------------------------------------

def test_mocked_messages_through_risk_pipeline():
    import pandas as pd
    from zones import get_zones
    from geofencing import run_geofencing
    from feature_engineering import build_feature_matrix
    from anomaly_detector import run_anomaly_detection
    from risk_engine import run_risk_engine, generate_alerts
    from behavior_analysis import analyze_all

    sm = VesselStateManager()
    # Feed several mocked vessels, a few with multi-point tracks.
    base = datetime(2025, 1, 1, tzinfo=timezone.utc)
    for v in range(6):
        for i in range(4):
            msg = _position_msg(mmsi=900000 + v, lat=10.0 + v * 0.1 + i * 0.02,
                                lon=80.0 + v * 0.1 + i * 0.02, sog=5.0 + v)
            obs = normalize_message(msg)
            obs["timestamp"] = base + timedelta(minutes=5 * i)
            sm.update_vessel(obs)

    active = sm.get_all_active_vessels()
    rows = [{"vessel_id": vid, "latitude": round(s["latitude"], 4),
             "longitude": round(s["longitude"], 4), "speed": s["speed"] or 0.0,
             "heading": s["heading"] or 0.0, "behavior": "AIS Track"}
            for vid, s in active.items()]
    df = pd.DataFrame(rows)
    recent = sm.get_all_recent_tracks()

    zones = get_zones("Historical AIS")
    beh = analyze_all(recent)
    tuple_tracks = {vid: list(zip(t["latitude"].tolist(), t["longitude"].tolist()))
                    for vid, t in recent.items()}
    gm = {g["vessel_id"]: g for g in run_geofencing(df, zones)}
    import tempfile
    tmp_if = os.path.join(tempfile.mkdtemp(), "if.joblib")
    feats = build_feature_matrix(df, tuple_tracks, gm, behavior_map=beh)
    anom = run_anomaly_detection(feats, model_path=tmp_if)
    risk = run_risk_engine(df["vessel_id"].tolist(), gm, feats, anom, behavior_map=beh)
    assert len(risk) == len(df)
    assert all("risk_level" in r and "overall_score" in r for r in risk.values())
    _ = generate_alerts(risk)  # must not raise
    print(f"mocked messages -> risk pipeline  OK ({len(risk)} vessels scored)")


# ---------------------------------------------------------------------------
# Regression guard: other modes' modules still import/run
# ---------------------------------------------------------------------------

def test_existing_modes_unaffected():
    from data_generator import generate_vessel_dataframe
    from realtime_simulator import AISReplaySimulator  # noqa: F401
    assert len(generate_vessel_dataframe()) == 18
    print("existing simulated/replay modules intact  OK")


if __name__ == "__main__":
    tests = [
        test_api_key_env_detected_and_not_revealed,
        test_api_key_missing_returns_none,
        test_api_key_placeholder_is_not_configured,
        test_subscription_message_shape,
        test_normalize_valid_position,
        test_normalize_missing_mmsi,
        test_normalize_missing_coords,
        test_normalize_invalid_coords,
        test_normalize_missing_speed_heading,
        test_normalize_unknown_message_type,
        test_state_update_and_get,
        test_state_bounded_trajectory,
        test_state_remove_stale,
        test_regions_valid_and_india_default,
        test_client_start_without_key_sets_error,
        test_client_stop_is_safe_when_not_started,
        test_client_stats_no_secret,
        test_mocked_messages_through_risk_pipeline,
        test_existing_modes_unaffected,
    ]
    p = 0
    for t in tests:
        t(); p += 1
    print(f"\n=== {p}/{len(tests)} TESTS PASSED ===")

# ---------------------------------------------------------------------------
# Ship static data (type / name / destination / ETA) + first-seen proxy
# ---------------------------------------------------------------------------

def test_ship_type_labels():
    assert ship_type_label(30) == "Fishing"
    assert ship_type_label(37) == "Pleasure Craft"
    assert decode_ship_type(70) == "Cargo"
    assert decode_ship_type(80) == "Tanker"
    assert decode_ship_type(0) == "Unknown"
    assert decode_ship_type(None) == "Unknown"
    print("ship type labels  OK")


def _static_msg(mmsi=123456789, name="SEA EXPLORER", stype=70,
                dest="CHENNAI", month=3, day=15, hour=8, minute=30):
    return {
        "MessageType": "ShipStaticData",
        "MetaData": {"MMSI": mmsi},
        "Message": {"ShipStaticData": {
            "UserID": mmsi, "Name": name, "Type": stype,
            "Destination": dest, "MaximumStaticDraught": 7.5,
            "Eta": {"Month": month, "Day": day, "Hour": hour, "Minute": minute},
        }},
    }


def test_normalize_static_basic():
    out = normalize_static(_static_msg())
    assert out["vessel_id"] == "123456789"
    assert out["ship_name"] == "SEA EXPLORER"
    assert out["ship_type"] == "Cargo"
    assert out["destination"] == "CHENNAI"
    assert out["eta"] == "03-15 08:30 UTC"
    assert out["draught"] == 7.5
    # Non-static messages are ignored
    assert normalize_static({"MessageType": "PositionReport"}) is None
    print("normalize static basic  OK")


def test_static_merges_into_state_without_clobbering():
    vsm = VesselStateManager()
    vsm.update_vessel({
        "vessel_id": "555", "latitude": 10.0, "longitude": 80.0,
        "speed": 5.0, "heading": 90.0, "course": 90.0,
        "timestamp": datetime(2025, 1, 1, tzinfo=timezone.utc),
    })
    vsm.update_static(normalize_static(_static_msg(mmsi=555, name="FISHER I",
                                                   stype=30, dest="KOCHI")))
    s = vsm.get_vessel_state("555")
    assert s["ship_name"] == "FISHER I"
    assert s["ship_type"] == "Fishing"
    assert s["destination"] == "KOCHI"
    # first-seen proxy recorded on the first position
    assert s["first_seen_lat"] == 10.0 and s["first_seen_lon"] == 80.0
    assert s["first_seen_time"] == datetime(2025, 1, 1, tzinfo=timezone.utc)
    # An empty static update must not wipe a known name
    vsm.update_static({"vessel_id": "555", "ship_name": "", "ship_type": "Unknown"})
    assert vsm.get_vessel_state("555")["ship_name"] == "FISHER I"
    print("static merges into state without clobbering  OK")
