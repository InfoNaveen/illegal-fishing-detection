"""
risk_engine.py
Dynamic risk scoring engine for the Illegal Fishing Detection System.

Combines:
  - Geofencing result       (zone violation / proximity)
  - Behavioural features    (loitering, speed anomaly, erratic movement)
  - Isolation Forest score  (overall anomaly)

Produces per vessel:
  risk_score   : int 0–100
  risk_level   : "LOW" | "MEDIUM" | "HIGH"
  behavior     : human-readable primary behaviour label
  factors      : list of {factor, score, weight} dicts for the breakdown panel
  zone_status  : human-readable zone status string

Weights are intentionally transparent and explainable.
"""

import math
from typing import Dict, List, Tuple

# ---------------------------------------------------------------------------
# Component weights  (must sum ≤ 100 — remainder is baseline)
# ---------------------------------------------------------------------------

W_ZONE_VIOLATION  = 40   # hard inside a restricted zone
W_ZONE_PROXIMITY  = 15   # within PROXIMITY_THRESHOLD of a zone
W_LOITERING       = 20   # loitering_score contribution
W_SPEED_ANOMALY   = 12   # speed_deviation contribution
W_ERRATIC         = 10   # erratic_score contribution
W_ML_ANOMALY      = 25   # Isolation Forest normalised anomaly_score
W_COMBO_BOOST     = 8    # bonus when loitering+erratic both elevated (suspicious combo)
W_AIS_GAP         = 10   # AIS signal gap detected
# Maximum possible raw sum = 40+15+20+12+10+25+8+10 = 140 → normalise to 100

RAW_MAX = W_ZONE_VIOLATION + W_ZONE_PROXIMITY + W_LOITERING + \
          W_SPEED_ANOMALY + W_ERRATIC + W_ML_ANOMALY + W_COMBO_BOOST + W_AIS_GAP

# ---------------------------------------------------------------------------
# Risk thresholds
# ---------------------------------------------------------------------------

HIGH_THRESHOLD   = 55
MEDIUM_THRESHOLD = 20


# ---------------------------------------------------------------------------
# Behaviour label derivation
# ---------------------------------------------------------------------------

def _derive_behavior(geo: Dict, feat: Dict, anomaly: Dict) -> str:
    """Return a concise human-readable primary behaviour description."""
    if geo["inside"]:
        if feat["loitering_score"] > 0.50:
            return "Loitering / Zone Violation"
        return "Zone Violation"
    if feat["loitering_score"] > 0.60:
        return "Suspicious Loitering"
    if feat["erratic_score"] > 0.50:
        return "Erratic Movement"
    if feat["speed_deviation"] > 0.40:
        return "Speed Anomaly"
    if geo["near"]:
        return "Proximity to Restricted Zone"
    if anomaly["is_anomalous"]:
        return "Anomalous Behaviour"
    return "Normal Fishing" if feat["speed"] <= 9.0 else "Normal Transit"


# ---------------------------------------------------------------------------
# Core scoring
# ---------------------------------------------------------------------------

def compute_risk(vessel_id: str,
                 geo: Dict,
                 feat: Dict,
                 anomaly: Dict,
                 ais_gap_minutes: int = 0) -> Dict:
    """
    Compute risk for a single vessel.

    Parameters
    ----------
    vessel_id : str
    geo       : result dict from geofencing.check_vessel_zones()
    feat      : dict of feature values (keyed by feature name)
    anomaly   : {"anomaly_score": float, "is_anomalous": bool}

    Returns
    -------
    {
      "vessel_id"   : str,
      "risk_score"  : int 0-100,
      "risk_level"  : str,
      "behavior"    : str,
      "zone_status" : str,
      "factors"     : list of {factor, contribution, max_weight}
    }
    """
    factors: List[Dict] = []
    raw_total: float = 0.0

    # ── Zone violation ─────────────────────────────────────────────────────
    if geo["inside"]:
        contrib = float(W_ZONE_VIOLATION)
        factors.append({
            "factor":      f"Restricted Zone Violation ({geo['zone_name']})",
            "contribution": int(contrib),
            "max_weight":  W_ZONE_VIOLATION,
        })
        raw_total += contrib

    # ── Zone proximity ─────────────────────────────────────────────────────
    elif geo["near"]:
        # Scale by how close: 1.0 at threshold boundary → 0.0 at PROXIMITY_MAX
        prox_frac = feat.get("zone_proximity_norm", 0.0)
        contrib = W_ZONE_PROXIMITY * prox_frac
        if contrib > 0.5:
            factors.append({
                "factor":      f"Near {geo['nearest_zone']}",
                "contribution": int(round(contrib)),
                "max_weight":  W_ZONE_PROXIMITY,
            })
            raw_total += contrib

    # ── Loitering ──────────────────────────────────────────────────────────
    loiter = feat.get("loitering_score", 0.0)
    if loiter > 0.10:
        contrib = W_LOITERING * loiter
        factors.append({
            "factor":      "Loitering Detected",
            "contribution": int(round(contrib)),
            "max_weight":  W_LOITERING,
        })
        raw_total += contrib

    # ── Speed anomaly ──────────────────────────────────────────────────────
    spd_dev = feat.get("speed_deviation", 0.0)
    if spd_dev > 0.05:
        contrib = W_SPEED_ANOMALY * spd_dev
        factors.append({
            "factor":      "Speed Anomaly",
            "contribution": int(round(contrib)),
            "max_weight":  W_SPEED_ANOMALY,
        })
        raw_total += contrib

    # ── Erratic movement ───────────────────────────────────────────────────
    erratic = feat.get("erratic_score", 0.0)
    if erratic > 0.10:
        contrib = W_ERRATIC * erratic
        factors.append({
            "factor":      "Erratic / Irregular Movement",
            "contribution": int(round(contrib)),
            "max_weight":  W_ERRATIC,
        })
        raw_total += contrib

    # ── ML anomaly score ───────────────────────────────────────────────────
    ml_score = anomaly.get("anomaly_score", 0.0)
    if ml_score > 0.10:
        contrib = W_ML_ANOMALY * ml_score
        factors.append({
            "factor":      "ML Anomaly (Isolation Forest)",
            "contribution": int(round(contrib)),
            "max_weight":  W_ML_ANOMALY,
        })
        raw_total += contrib

    # ── AIS signal gap ────────────────────────────────────────────────────
    from data_generator import AIS_GAP_THRESHOLD_MINUTES
    if ais_gap_minutes >= AIS_GAP_THRESHOLD_MINUTES:
        # Scale contribution: 20 min → ~50% weight, 60 min → 100%
        gap_frac = min(ais_gap_minutes / 60.0, 1.0)
        contrib  = W_AIS_GAP * gap_frac
        factors.append({
            "factor":      f"AIS Signal Gap ({ais_gap_minutes} min)",
            "contribution": int(round(contrib)),
            "max_weight":  W_AIS_GAP,
        })
        raw_total += contrib

    # ── Combo boost: loitering + erratic together is more suspicious ──────
    if loiter > 0.40 and erratic > 0.50:
        combo = W_COMBO_BOOST * min((loiter + erratic) / 2.0, 1.0)
        factors.append({
            "factor":      "Combined Suspicious Indicators",
            "contribution": int(round(combo)),
            "max_weight":  W_COMBO_BOOST,
        })
        raw_total += combo

    # ── Normalise to 0–100 ────────────────────────────────────────────────
    risk_score = int(round(min(raw_total / RAW_MAX * 100, 100)))

    # ── Risk level ────────────────────────────────────────────────────────
    if risk_score >= HIGH_THRESHOLD:
        risk_level = "HIGH"
    elif risk_score >= MEDIUM_THRESHOLD:
        risk_level = "MEDIUM"
    else:
        risk_level = "LOW"

    behavior   = _derive_behavior(geo, feat, anomaly)
    zone_status = geo["zone_status"]

    return {
        "vessel_id":       vessel_id,
        "risk_score":      risk_score,
        "risk_level":      risk_level,
        "behavior":        behavior,
        "zone_status":     zone_status,
        "factors":         factors,
        "ais_gap_minutes": ais_gap_minutes,
    }


# ---------------------------------------------------------------------------
# Batch risk computation
# ---------------------------------------------------------------------------

def run_risk_engine(vessel_ids: List[str],
                    geo_map: Dict[str, Dict],
                    feature_df,
                    anomaly_map: Dict[str, Dict]) -> Dict[str, Dict]:
    """
    Run the risk engine for all vessels.

    Returns {vessel_id: risk_result_dict}
    """
    from data_generator import get_ais_gap
    results: Dict[str, Dict] = {}
    for vid in vessel_ids:
        geo     = geo_map.get(vid, {"inside": False, "near": False,
                                    "zone_name": "", "nearest_zone": "",
                                    "min_dist_deg": 0.30,
                                    "zone_status": "Open Waters"})
        feat    = feature_df.loc[vid].to_dict() if vid in feature_df.index else {}
        anomaly = anomaly_map.get(vid, {"anomaly_score": 0.0, "is_anomalous": False})
        gap     = get_ais_gap(vid)
        results[vid] = compute_risk(vid, geo, feat, anomaly, ais_gap_minutes=gap)
    return results


# ---------------------------------------------------------------------------
# Alert generation
# ---------------------------------------------------------------------------

def generate_alerts(risk_results: Dict[str, Dict]) -> List[Dict]:
    """
    Dynamically generate alerts from computed risk results.

    Returns a list of alert dicts sorted by severity then risk_score desc.
    Each alert: {level, icon, vessel, message, reason}
    """
    alerts: List[Dict] = []
    level_order = {"HIGH": 0, "MEDIUM": 1, "LOW": 2}

    for vid, result in risk_results.items():
        level      = result["risk_level"]
        score      = result["risk_score"]
        behavior   = result["behavior"]
        zone_stat  = result["zone_status"]
        factors    = result["factors"]

        if level not in ("HIGH", "MEDIUM"):
            continue

        icon = "🔴" if level == "HIGH" else "🟠"

        # Build a concise message from top factor
        if factors:
            top_factor = max(factors, key=lambda f: f["contribution"])
            reason     = top_factor["factor"]
        else:
            reason = behavior

        if "Zone Violation" in behavior or "INSIDE" in zone_stat:
            message = (f"{vid} detected inside restricted zone "
                       f"— {zone_stat}. Immediate investigation required.")
        elif "Loitering" in behavior:
            message = (f"{vid} showing sustained loitering behaviour "
                       f"(score: {score}/100). Possible illegal fishing activity.")
        elif "AIS Gap" in reason or result.get("ais_gap_minutes", 0) >= 20:
            gap = result.get("ais_gap_minutes", 0)
            message = (f"{vid} AIS signal gap of {gap} minutes recorded "
                       f"— vessel went dark. Suspicious blackout period.")
        elif "Speed Anomaly" in behavior:
            message = (f"{vid} speed anomaly detected "
                       f"— movement pattern inconsistent with normal fishing.")
        elif "Erratic" in behavior:
            message = (f"{vid} erratic course changes detected "
                       f"— behaviour inconsistent with legitimate transit.")
        elif "Proximity" in behavior or "Near" in zone_stat:
            message = (f"{vid} operating close to restricted zone "
                       f"({zone_stat}) — monitor closely.")
        elif "Anomalous" in behavior:
            message = (f"{vid} flagged by anomaly detection model "
                       f"(score: {score}/100). Pattern deviates from fleet norms.")
        else:
            message = f"{vid} risk score {score}/100 — {reason}."

        alerts.append({
            "level":   level,
            "icon":    icon,
            "vessel":  vid,
            "message": message,
            "reason":  reason,
        })

    # Sort: HIGH first, then by descending risk_score
    alerts.sort(key=lambda a: (
        level_order.get(a["level"], 9),
        -risk_results[a["vessel"]]["risk_score"],
    ))
    return alerts
