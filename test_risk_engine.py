"""
test_risk_engine.py
Tests for the M7 unified vessel risk engine (risk_engine.py).

Run:
    python test_risk_engine.py
or:
    python -m pytest test_risk_engine.py -q
"""

from risk_engine import (
    compute_risk, generate_alerts,
    W_GEOFENCE, W_ISOLATION, W_TEMPORAL, W_BEHAVIOUR, W_AIS_GAP,
    HIGH_THRESHOLD, MEDIUM_THRESHOLD,
)

_OPEN = {"inside": False, "near": False, "zone_name": "", "nearest_zone": "",
         "zone_status": "Open Waters"}
_INSIDE = {"inside": True, "near": False, "zone_name": "Zone A", "nearest_zone": "Zone A",
           "zone_status": "INSIDE Zone A"}
_NORMAL_ANOM = {"anomaly_score": 0.0, "is_anomalous": False}


def test_weights_sum_to_one():
    total = W_GEOFENCE + W_ISOLATION + W_TEMPORAL + W_BEHAVIOUR + W_AIS_GAP
    assert abs(total - 1.0) < 1e-9, total
    print("weights sum to 1.0  OK")


def test_all_components_low():
    r = compute_risk("A", _OPEN, {}, _NORMAL_ANOM)
    assert r["overall_score"] == 0
    assert r["risk_level"] == "LOW"
    assert r["risk_factors"] == ["No elevated risk indicators"]
    print("all components low -> 0 / LOW  OK")


def test_geofence_only():
    r = compute_risk("G", _INSIDE, {}, _NORMAL_ANOM)
    assert r["geofence_component"] == 1.0
    assert r["overall_score"] == int(round(W_GEOFENCE * 100))  # 30
    print(f"geofence-only -> {r['overall_score']}  OK")


def test_isolation_only():
    r = compute_risk("I", _OPEN, {}, {"anomaly_score": 1.0, "is_anomalous": True})
    assert r["isolation_component"] == 1.0
    assert r["overall_score"] == int(round(W_ISOLATION * 100))  # 25
    print(f"isolation-only -> {r['overall_score']}  OK")


def test_temporal_only():
    r = compute_risk("T", _OPEN, {}, _NORMAL_ANOM,
                     temporal={"temporal_anomaly_score": 1.0, "temporal_model_status": "scored"})
    assert r["temporal_component"] == 1.0
    assert r["overall_score"] == int(round(W_TEMPORAL * 100))  # 15
    print(f"temporal-only -> {r['overall_score']}  OK")


def test_behaviour_only():
    r = compute_risk("B", _OPEN, {"loitering_score": 1.0}, _NORMAL_ANOM)
    assert r["behaviour_component"] == 1.0
    assert r["overall_score"] == int(round(W_BEHAVIOUR * 100))  # 20
    print(f"behaviour-only -> {r['overall_score']}  OK")


def test_ais_gap_only():
    r = compute_risk("GAP", _OPEN, {}, _NORMAL_ANOM, ais_gap_minutes=60)
    assert r["ais_gap_component"] == 1.0
    assert r["overall_score"] == int(round(W_AIS_GAP * 100))  # 10
    print(f"ais-gap-only -> {r['overall_score']}  OK")


def test_missing_temporal_safe():
    # No temporal provided, or unavailable status → temporal component 0 (not fabricated).
    r1 = compute_risk("X", _OPEN, {}, _NORMAL_ANOM, temporal=None)
    r2 = compute_risk("Y", _OPEN, {}, _NORMAL_ANOM,
                      temporal={"temporal_anomaly_score": 0.9, "temporal_model_status": "insufficient_data"})
    assert r1["temporal_component"] == 0.0
    assert r2["temporal_component"] == 0.0, "insufficient_data must not contribute"
    print("missing/insufficient temporal -> 0 (not fabricated)  OK")


def test_no_ais_gap_double_count():
    # Both a simulated gap (47) and a trajectory gap (55) → ONE unified gap at 55.
    r = compute_risk("D", _OPEN, {}, _NORMAL_ANOM,
                     ais_gap_minutes=47, behavior={"max_ais_gap_minutes": 55})
    gap_factors = [f for f in r["factors"] if "ais gap" in f["factor"].lower()]
    assert len(gap_factors) == 1, f"expected 1 gap factor, got {len(gap_factors)}"
    assert r["ais_gap_minutes"] == 55
    # The component equals a single 55/60 contribution, not 47+55 double.
    assert abs(r["ais_gap_component"] - min(55 / 60.0, 1.0)) < 1e-3
    print("no AIS-gap double count (unified to max=55)  OK")


def test_risk_level_thresholds():
    # Construct a vessel just over HIGH: geo(1.0)=30 + isolation(1.0)=25 = 55.
    r = compute_risk("H", _INSIDE, {}, {"anomaly_score": 1.0, "is_anomalous": True})
    assert r["overall_score"] >= HIGH_THRESHOLD and r["risk_level"] == "HIGH"
    # Just MEDIUM: geofence only = 30 (>=20, <55).
    rm = compute_risk("M", _INSIDE, {}, _NORMAL_ANOM)
    assert rm["risk_level"] == "MEDIUM"
    print("risk-level thresholds  OK")


def test_risk_factors_and_breakdown():
    r = compute_risk("F", _INSIDE, {"loitering_score": 0.8},
                     {"anomaly_score": 0.9, "is_anomalous": True},
                     temporal={"temporal_anomaly_score": 0.8, "temporal_model_status": "scored"})
    # structured breakdown present
    for k in ("geofence_component", "isolation_component", "temporal_component",
              "behaviour_component", "ais_gap_component", "explanation"):
        assert k in r
    assert any("monitoring zone" in f.lower() for f in r["risk_factors"])
    assert "illegal fishing" not in r["explanation"].lower()
    print("risk factors + structured breakdown + honest wording  OK")


def test_alert_wording_honest():
    risk = {"V": compute_risk("V", _INSIDE, {"loitering_score": 0.9},
                              {"anomaly_score": 1.0, "is_anomalous": True})}
    alerts = generate_alerts(risk)
    assert alerts and alerts[0]["level"] == "HIGH"
    for a in alerts:
        assert "illegal fishing" not in a["message"].lower()
    print("alert wording honest (no illegal-fishing claim)  OK")


def test_simulated_regression():
    """V102/V087/V215 remain HIGH under the unified engine."""
    import warnings; warnings.filterwarnings("ignore")
    import os, tempfile
    from zones import get_zones
    from geofencing import run_geofencing
    from feature_engineering import build_feature_matrix
    from anomaly_detector import run_anomaly_detection
    from behavior_analysis import analyze_all
    from risk_engine import run_risk_engine
    from data_generator import generate_vessel_dataframe, build_all_trajectories

    tmp_if = os.path.join(tempfile.mkdtemp(), "if.joblib")
    zones = get_zones("Simulated")
    base = generate_vessel_dataframe()
    trajs = build_all_trajectories(base)
    beh = analyze_all(trajs)
    gm = {g["vessel_id"]: g for g in run_geofencing(base, zones)}
    feats = build_feature_matrix(base, trajs, gm, behavior_map=beh)
    anom = run_anomaly_detection(feats, model_path=tmp_if)
    risk = run_risk_engine(base["vessel_id"].tolist(), gm, feats, anom, behavior_map=beh)
    for vid in ["V102", "V087", "V215"]:
        assert risk[vid]["risk_level"] == "HIGH", f"{vid}={risk[vid]['risk_level']}"
    print("simulated regression V102/V087/V215 HIGH  OK")


if __name__ == "__main__":
    tests = [
        test_weights_sum_to_one, test_all_components_low, test_geofence_only,
        test_isolation_only, test_temporal_only, test_behaviour_only,
        test_ais_gap_only, test_missing_temporal_safe, test_no_ais_gap_double_count,
        test_risk_level_thresholds, test_risk_factors_and_breakdown,
        test_alert_wording_honest, test_simulated_regression,
    ]
    p = 0
    for t in tests:
        t(); p += 1
    print(f"\n=== {p}/{len(tests)} TESTS PASSED ===")
