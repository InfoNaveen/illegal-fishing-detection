"""
alert_engine.py
Alert generation with deduplication / cooldown (Round 3 — Milestone M8).

Turns per-vessel unified risk results (risk_engine) into alerts during the AIS
replay simulation, while preventing alert spam: the same (vessel, alert_type)
cannot re-fire within a configurable cooldown (measured in replay ticks).

Alert types
-----------
    HIGH_RISK          — overall risk level is HIGH
    ZONE_ENTRY         — vessel is inside a monitoring zone
    AIS_GAP            — significant AIS reporting gap
    BEHAVIOUR_ANOMALY  — strong behavioural anomaly component
    TEMPORAL_ANOMALY   — elevated temporal sequence anomaly component
    COMBINED           — multiple strong signals at once

Honesty: alert messages describe "suspicious fishing-related behaviour
indicators" — never confirmed illegal fishing.
"""

from __future__ import annotations

from typing import Dict, List, Optional

DEFAULT_COOLDOWN_TICKS: int = 10   # same vessel+type cannot refire within N ticks

# Component thresholds for triggering (align with risk_engine semantics).
_ZONE_MIN      = 0.99     # inside zone → geofence component == 1.0
_GAP_MIN       = 0.34     # ~20 min / 60 → component ≈ 0.33+
_BEHAVIOUR_MIN = 0.60
_TEMPORAL_MIN  = 0.55


class AlertEngine:
    """
    Stateful alert generator with per-(vessel, type) cooldown.

    The engine is tick-driven: call evaluate(risk_results, tick) once per replay
    tick. It returns only NEW alerts (those not suppressed by cooldown).
    """

    def __init__(self, cooldown_ticks: int = DEFAULT_COOLDOWN_TICKS):
        self.cooldown_ticks = int(cooldown_ticks)
        # {(vessel_id, alert_type): last_fired_tick}
        self._last_fired: Dict[tuple, int] = {}

    # ------------------------------------------------------------------
    def _due(self, vid: str, atype: str, tick: int) -> bool:
        key = (vid, atype)
        last = self._last_fired.get(key)
        if last is None or (tick - last) >= self.cooldown_ticks:
            self._last_fired[key] = tick
            return True
        return False

    @staticmethod
    def _types_for(result: Dict) -> List[tuple]:
        """Return [(alert_type, level, message, reason), ...] a result warrants."""
        out = []
        vid = result["vessel_id"]
        score = result.get("overall_score", result.get("risk_score", 0))
        level = result["risk_level"]
        zone = result.get("zone_status", "")

        strong_signals = 0

        if result.get("geofence_component", 0.0) >= _ZONE_MIN:
            strong_signals += 1
            out.append(("ZONE_ENTRY", "HIGH" if level == "HIGH" else "MEDIUM",
                        f"{vid} inside monitoring zone ({zone}). "
                        f"Suspicious fishing-related behaviour indicators.",
                        "Inside monitoring zone"))
        if result.get("ais_gap_component", 0.0) >= _GAP_MIN:
            strong_signals += 1
            gap = result.get("ais_gap_minutes", 0)
            out.append(("AIS_GAP", "MEDIUM",
                        f"{vid} significant AIS gap ({gap} min) — "
                        f"suspicious reporting-gap behaviour indicator.",
                        "Significant AIS gap"))
        if result.get("behaviour_component", 0.0) >= _BEHAVIOUR_MIN:
            strong_signals += 1
            out.append(("BEHAVIOUR_ANOMALY", "MEDIUM",
                        f"{vid} strong behavioural anomaly "
                        f"(loitering/turning/speed).",
                        "Behavioural anomaly"))
        if result.get("temporal_component", 0.0) >= _TEMPORAL_MIN:
            strong_signals += 1
            out.append(("TEMPORAL_ANOMALY", "MEDIUM",
                        f"{vid} elevated temporal sequence anomaly.",
                        "Temporal anomaly"))

        if level == "HIGH":
            out.append(("HIGH_RISK", "HIGH",
                        f"{vid} HIGH risk ({score}/100). "
                        f"Suspicious fishing-related behaviour indicators detected.",
                        "HIGH overall risk"))
        if strong_signals >= 2:
            out.append(("COMBINED", "HIGH",
                        f"{vid} multiple suspicious indicators detected "
                        f"({strong_signals} signals).",
                        "Multiple suspicious indicators"))
        return out

    # ------------------------------------------------------------------
    def evaluate(self, risk_results: Dict[str, Dict], tick: int) -> List[Dict]:
        """
        Evaluate risk results for one tick and return NEW alerts (post-cooldown).

        Each alert: {vessel_id, level, message, reason, alert_type, tick}.
        """
        new_alerts: List[Dict] = []
        # Deterministic order: by descending risk then vessel id.
        ordered = sorted(
            risk_results.values(),
            key=lambda r: (-(r.get("overall_score", r.get("risk_score", 0))),
                           r["vessel_id"]),
        )
        for result in ordered:
            vid = result["vessel_id"]
            for atype, level, message, reason in self._types_for(result):
                if self._due(vid, atype, tick):
                    new_alerts.append({
                        "vessel_id":  vid,
                        "level":      level,
                        "message":    message,
                        "reason":     reason,
                        "alert_type": atype,
                        "tick":       tick,
                    })
        return new_alerts

    def reset(self) -> None:
        """Clear cooldown history (e.g. when a new replay starts)."""
        self._last_fired.clear()
