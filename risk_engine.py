"""
risk_engine.py
Unified vessel risk scoring engine for the Illegal Fishing Detection System.

Round 3 — Milestone M7
----------------------
This is the SINGLE canonical risk decision layer. It combines five normalised
[0, 1] components, each counted EXACTLY ONCE, using documented weights:

    overall = 100 * ( w_geofence  * geofence_component
                    + w_isolation * isolation_component
                    + w_temporal  * temporal_component
                    + w_behaviour * behaviour_component
                    + w_ais_gap   * ais_gap_component )

Component definitions (all normalised to [0, 1])
------------------------------------------------
geofence   : 1.0 inside a zone; proximity fraction when near; else 0.0
isolation  : Isolation Forest normalised anomaly score (M5)
temporal   : temporal sequence anomaly score (M6); 0.0 when unavailable
behaviour  : max of loitering / speed-deviation / erratic / turning indicators
ais_gap    : a SINGLE unified AIS-gap signal — max of the simulated per-vessel
             gap and the trajectory-derived gap, scaled by 60 min. This removes
             the earlier triple-counting of AIS gaps.

Honesty note
------------
Unsupervised anomaly / suspicious-behaviour scoring. Factors are phrased as
"suspicious fishing-related behaviour indicators" — never as confirmed illegal
fishing. No labelled ground truth; no accuracy is claimed.
"""

from typing import Dict, List

# ---------------------------------------------------------------------------
# Component weights (sum to 1.0 — documented, each signal counted once)
# ---------------------------------------------------------------------------

W_GEOFENCE   = 0.30
W_ISOLATION  = 0.25
W_TEMPORAL   = 0.15
W_BEHAVIOUR  = 0.20
W_AIS_GAP    = 0.10
# Sum = 1.00

# ---------------------------------------------------------------------------
# Thresholds
# ---------------------------------------------------------------------------

HIGH_THRESHOLD   = 55
MEDIUM_THRESHOLD = 20

# AIS-gap normalisation reference (minutes → full component at 60 min)
AIS_GAP_FULL_MINUTES = 60.0
AIS_GAP_MIN_MINUTES  = 20.0   # below this, no AIS-gap contribution


# ---------------------------------------------------------------------------
# Component computation (each returns a value in [0, 1])
# ---------------------------------------------------------------------------

def _geofence_component(geo: Dict, feat: Dict) -> float:
    if geo.get("inside"):
        return 1.0
    if geo.get("near"):
        return float(min(max(feat.get("zone_proximity_norm", 0.0), 0.0), 1.0))
    return 0.0


def _isolation_component(anomaly: Dict) -> float:
    return float(min(max(anomaly.get("anomaly_score", 0.0), 0.0), 1.0))


def _temporal_component(temporal: Dict) -> float:
    if not temporal or temporal.get("temporal_model_status") != "scored":
        return 0.0
    return float(min(max(temporal.get("temporal_anomaly_score", 0.0), 0.0), 1.0))


def _behaviour_component(feat: Dict, behavior: Dict) -> float:
    """Max of the behavioural indicators (loitering, speed dev, erratic, turning)."""
    loiter   = float(feat.get("loitering_score", 0.0))
    spd_dev  = float(feat.get("speed_deviation", 0.0))
    erratic  = float(feat.get("erratic_score", 0.0))
    turn     = float(behavior.get("mean_heading_change", 0.0)) / 120.0  # 120° → 1.0
    return float(min(max(max(loiter, spd_dev, erratic, turn), 0.0), 1.0))


def _ais_gap_minutes_unified(ais_gap_minutes: float, behavior: Dict) -> float:
    """Single unified gap in minutes = max(simulated gap, trajectory gap)."""
    traj_gap = float(behavior.get("max_ais_gap_minutes", 0.0))
    return max(float(ais_gap_minutes or 0.0), traj_gap)


def _ais_gap_component(gap_minutes: float) -> float:
    if gap_minutes < AIS_GAP_MIN_MINUTES:
        return 0.0
    return float(min(gap_minutes / AIS_GAP_FULL_MINUTES, 1.0))


# ---------------------------------------------------------------------------
# Behaviour label (kept for the dashboard's primary-behaviour text)
# ---------------------------------------------------------------------------

def _derive_behavior(geo: Dict, feat: Dict, anomaly: Dict) -> str:
    if geo.get("inside"):
        return "Loitering / Zone Violation" if feat.get("loitering_score", 0) > 0.50 else "Zone Violation"
    if feat.get("loitering_score", 0) > 0.60:
        return "Suspicious Loitering"
    if feat.get("erratic_score", 0) > 0.50:
        return "Erratic Movement"
    if feat.get("speed_deviation", 0) > 0.40:
        return "Speed Anomaly"
    if geo.get("near"):
        return "Proximity to Restricted Zone"
    if anomaly.get("is_anomalous"):
        return "Anomalous Behaviour"
    return "Normal Fishing" if feat.get("speed", 0) <= 9.0 else "Normal Transit"


# ---------------------------------------------------------------------------
# Core unified scoring
# ---------------------------------------------------------------------------

def compute_risk(vessel_id: str,
                 geo: Dict,
                 feat: Dict,
                 anomaly: Dict,
                 ais_gap_minutes: float = 0,
                 behavior: Dict = None,
                 temporal: Dict = None) -> Dict:
    """
    Compute the unified risk for one vessel.

    Returns a structured breakdown:
      vessel_id, risk_score (0-100 int, == overall_score), overall_score,
      risk_level, behavior, zone_status,
      geofence_component, isolation_component, temporal_component,
      behaviour_component, ais_gap_component (each 0-1),
      factors (weighted breakdown for the panel),
      risk_factors (plain-language list), explanation,
      ais_gap_minutes, behavior_summary, behavior_labels.
    """
    behavior = behavior or {}
    temporal = temporal or {}
    feat = feat or {}

    # ── Components (each in [0,1], each counted once) ─────────────────────
    c_geo  = _geofence_component(geo, feat)
    c_iso  = _isolation_component(anomaly)
    c_temp = _temporal_component(temporal)
    c_beh  = _behaviour_component(feat, behavior)
    gap_minutes = _ais_gap_minutes_unified(ais_gap_minutes, behavior)
    c_gap  = _ais_gap_component(gap_minutes)

    # ── Weighted overall score ───────────────────────────────────────────
    overall = (W_GEOFENCE * c_geo + W_ISOLATION * c_iso + W_TEMPORAL * c_temp
               + W_BEHAVIOUR * c_beh + W_AIS_GAP * c_gap)
    overall_score = int(round(min(overall, 1.0) * 100))

    risk_level = ("HIGH" if overall_score >= HIGH_THRESHOLD
                  else "MEDIUM" if overall_score >= MEDIUM_THRESHOLD
                  else "LOW")

    # ── Weighted factor breakdown (for the dashboard panel) ──────────────
    factors: List[Dict] = []

    def _add(name: str, comp: float, weight: float):
        if comp > 0.01:
            factors.append({
                "factor":       name,
                "contribution": int(round(weight * comp * 100)),
                "max_weight":   int(round(weight * 100)),
                "component":    round(comp, 4),
            })

    if geo.get("inside"):
        _add(f"Inside monitoring zone ({geo.get('zone_name','')})", c_geo, W_GEOFENCE)
    elif geo.get("near"):
        _add(f"Near monitoring zone ({geo.get('nearest_zone','')})", c_geo, W_GEOFENCE)
    _add("Isolation Forest anomaly", c_iso, W_ISOLATION)
    _add("Temporal sequence anomaly", c_temp, W_TEMPORAL)
    _add("Behavioural anomaly (loitering/turning/speed)", c_beh, W_BEHAVIOUR)
    if c_gap > 0.01:
        _add(f"Significant AIS gap ({int(gap_minutes)} min)", c_gap, W_AIS_GAP)

    # ── Explainable, plain-language risk factors ─────────────────────────
    risk_factors: List[str] = []
    if geo.get("inside"):
        risk_factors.append(f"Inside demonstration monitoring zone ({geo.get('zone_name','')})")
    elif geo.get("near"):
        risk_factors.append(f"Operating near monitoring zone ({geo.get('nearest_zone','')})")
    if c_iso >= 0.55:
        risk_factors.append("Elevated Isolation Forest anomaly")
    if c_temp >= 0.55:
        risk_factors.append("Elevated temporal sequence anomaly")
    if float(feat.get("loitering_score", 0)) >= 0.6:
        risk_factors.append("Loitering behaviour")
    if float(behavior.get("mean_heading_change", 0)) >= 45.0:
        risk_factors.append("High turning activity")
    if float(feat.get("speed_deviation", 0)) >= 0.4:
        risk_factors.append("Speed anomaly")
    if c_gap > 0.01:
        risk_factors.append(f"Significant AIS gap ({int(gap_minutes)} min)")
    if not risk_factors:
        risk_factors = ["No elevated risk indicators"]

    behavior_label = _derive_behavior(geo, feat, anomaly)

    if risk_level == "HIGH":
        explanation = ("Multiple or strong suspicious fishing-related behaviour "
                       "indicators detected.")
    elif risk_level == "MEDIUM":
        explanation = "Some suspicious behaviour indicators present; monitor."
    else:
        explanation = "No significant suspicious behaviour indicators."

    return {
        "vessel_id":           vessel_id,
        "risk_score":          overall_score,     # back-compat alias
        "overall_score":       overall_score,
        "risk_level":          risk_level,
        "behavior":            behavior_label,
        "zone_status":         geo.get("zone_status", "Open Waters"),
        "geofence_component":  round(c_geo, 4),
        "isolation_component": round(c_iso, 4),
        "temporal_component":  round(c_temp, 4),
        "behaviour_component": round(c_beh, 4),
        "ais_gap_component":   round(c_gap, 4),
        "factors":             factors,
        "risk_factors":        risk_factors,
        "explanation":         explanation,
        "ais_gap_minutes":     int(round(gap_minutes)),
        "behavior_summary":    behavior.get("behavior_summary", ""),
        "behavior_labels":     behavior.get("behavior_labels", []),
    }


# ---------------------------------------------------------------------------
# Batch risk computation
# ---------------------------------------------------------------------------

def run_risk_engine(vessel_ids: List[str],
                    geo_map: Dict[str, Dict],
                    feature_df,
                    anomaly_map: Dict[str, Dict],
                    behavior_map: Dict[str, Dict] = None,
                    temporal_map: Dict[str, Dict] = None) -> Dict[str, Dict]:
    """
    Run the unified risk engine for all vessels.

    behavior_map : optional {vessel_id: behaviour_dict} (M4)
    temporal_map : optional {vessel_id: {temporal_anomaly_score, ...}} (M6)

    Returns {vessel_id: unified_risk_result}
    """
    from data_generator import get_ais_gap
    behavior_map = behavior_map or {}
    temporal_map = temporal_map or {}
    results: Dict[str, Dict] = {}
    for vid in vessel_ids:
        geo     = geo_map.get(vid, {"inside": False, "near": False,
                                    "zone_name": "", "nearest_zone": "",
                                    "min_dist_deg": 0.30,
                                    "zone_status": "Open Waters"})
        feat    = feature_df.loc[vid].to_dict() if vid in feature_df.index else {}
        anomaly = anomaly_map.get(vid, {"anomaly_score": 0.0, "is_anomalous": False})
        gap     = get_ais_gap(vid)
        results[vid] = compute_risk(vid, geo, feat, anomaly,
                                    ais_gap_minutes=gap,
                                    behavior=behavior_map.get(vid),
                                    temporal=temporal_map.get(vid))
    return results


# ---------------------------------------------------------------------------
# Alert generation (unchanged interface; honest wording)
# ---------------------------------------------------------------------------

def generate_alerts(risk_results: Dict[str, Dict]) -> List[Dict]:
    """
    Dynamically generate alerts from unified risk results.

    Returns alert dicts sorted HIGH-first then by descending score.
    Each alert: {level, icon, vessel, message, reason}
    """
    alerts: List[Dict] = []
    level_order = {"HIGH": 0, "MEDIUM": 1, "LOW": 2}

    for vid, result in risk_results.items():
        level     = result["risk_level"]
        score     = result["overall_score"]
        behavior  = result["behavior"]
        zone_stat = result["zone_status"]
        factors   = result["factors"]

        if level not in ("HIGH", "MEDIUM"):
            continue

        icon = "🔴" if level == "HIGH" else "🟠"
        reason = (max(factors, key=lambda f: f["contribution"])["factor"]
                  if factors else behavior)

        if "Zone Violation" in behavior or "INSIDE" in zone_stat:
            message = (f"{vid} detected inside monitoring zone — {zone_stat}. "
                       f"Suspicious fishing-related behaviour indicators detected.")
        elif "Loitering" in behavior:
            message = (f"{vid} sustained loitering behaviour (risk {score}/100). "
                       f"Suspicious fishing-related behaviour indicators detected.")
        elif "AIS Gap" in reason or result.get("ais_gap_minutes", 0) >= 20:
            gap = result.get("ais_gap_minutes", 0)
            message = (f"{vid} AIS signal gap of {gap} minutes "
                       f"— suspicious reporting-gap behaviour indicator.")
        elif "Temporal" in reason:
            message = (f"{vid} elevated temporal sequence anomaly (risk {score}/100).")
        elif "Speed Anomaly" in behavior:
            message = (f"{vid} speed anomaly — movement inconsistent with normal fishing.")
        elif "Erratic" in behavior:
            message = (f"{vid} erratic course changes — behaviour inconsistent with transit.")
        elif "Proximity" in behavior or "Near" in zone_stat:
            message = (f"{vid} operating close to monitoring zone ({zone_stat}) — monitor.")
        elif "Anomalous" in behavior:
            message = (f"{vid} flagged by anomaly detection (risk {score}/100).")
        else:
            message = f"{vid} risk {score}/100 — {reason}."

        alerts.append({"level": level, "icon": icon, "vessel": vid,
                       "message": message, "reason": reason})

    alerts.sort(key=lambda a: (level_order.get(a["level"], 9),
                               -risk_results[a["vessel"]]["overall_score"]))
    return alerts
