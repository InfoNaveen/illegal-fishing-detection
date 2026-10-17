"""
live_ais.py
Live AIS integration via the AISStream.io WebSocket API.

IMPORTANT TRUTH
---------------
This connects to a real third-party AIS streaming provider. Coverage is not
guaranteed, messages may be delayed or missing, and absence of AIS traffic
does not indicate anything. Anomaly/risk output is a suspicious-behaviour
indicator, not confirmed illegal fishing.

Design
------
- The AISStream WebSocket client runs in a background thread (its own asyncio
  event loop) so it never blocks the Streamlit script thread.
- The API key is read from the environment (AISSTREAM_API_KEY, optionally via
  a local .env file). It is never hardcoded, never logged, and never exposed
  to the UI or the browser.
- Incoming messages are normalized to the SAME internal schema used elsewhere
  in IFDS (vessel_id, latitude, longitude, speed, heading, timestamp) and fed
  into a bounded, thread-safe VesselStateManager — no unlimited memory growth.
- Reconnection uses exponential backoff capped at a maximum delay.
- All networking lives here; app.py only reads state and renders it.
"""

from __future__ import annotations

import asyncio
import json
import os
import threading
import time
from collections import deque
from datetime import datetime, timezone
from typing import Deque, Dict, List, Optional

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass  # python-dotenv is optional at runtime; env vars may be set directly

AISSTREAM_URL = "wss://stream.aisstream.io/v0/stream"

# ---------------------------------------------------------------------------
# Configuration (environment-driven, sensible defaults, no code edits needed)
# ---------------------------------------------------------------------------

def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


def get_api_key() -> Optional[str]:
    """
    Read the AISStream API key securely. Never hardcoded, never logged.

    Precedence (documented):
      1. AISSTREAM_API_KEY environment variable (local .env or shell export)
      2. Streamlit Secrets (st.secrets["AISSTREAM_API_KEY"]) when deployed

    The placeholder value from .env.example is treated as "not configured".
    """
    key = os.environ.get("AISSTREAM_API_KEY", "").strip()
    if not key:
        try:
            import streamlit as st  # optional; only present in the app runtime
            key = str(st.secrets.get("AISSTREAM_API_KEY", "")).strip()
        except Exception:
            key = ""
    if not key or key == "your_aisstream_api_key_here":
        return None
    return key


# "Live AIS Monitoring Region" — a configurable demonstration bounding box,
# NOT a claimed legal fishing-restriction zone. Defaults roughly cover the
# Danish-waters area already used by the historical/replay modes, purely for
# geographic continuity with the rest of the project's demonstration.
LIVE_AIS_BBOX: Dict[str, float] = {
    "min_lat": _env_float("LIVE_AIS_MIN_LAT", 53.5),
    "min_lon": _env_float("LIVE_AIS_MIN_LON", 3.0),
    "max_lat": _env_float("LIVE_AIS_MAX_LAT", 60.0),
    "max_lon": _env_float("LIVE_AIS_MAX_LON", 16.5),
}

LIVE_AIS_REFRESH_SECONDS      = _env_int("LIVE_AIS_REFRESH_SECONDS", 3)
LIVE_AIS_STALE_SECONDS        = _env_int("LIVE_AIS_STALE_SECONDS", 300)
LIVE_AIS_MAX_TRAJECTORY_POINTS = _env_int("LIVE_AIS_MAX_TRAJECTORY_POINTS", 50)

_MAX_BACKOFF_SECONDS = 60.0

# ---------------------------------------------------------------------------
# Selectable India-focused monitoring regions (AISStream subscription boxes).
# Each is {min_lat, min_lon, max_lat, max_lon, center:(lat,lon), zoom}.
# These are DEMONSTRATION monitoring regions, not official legal boundaries,
# and selecting them does not imply national AIS coverage. Coordinates were
# chosen to cover the stated coastal stretch; verify before operational use.
# ---------------------------------------------------------------------------
LIVE_AIS_REGIONS: Dict[str, Dict] = {
    "India (All Coasts — widest coverage)": {
        "min_lat": 6.0, "min_lon": 66.0, "max_lat": 24.0, "max_lon": 93.0,
        "center": (15.0, 80.0), "zoom": 4,
    },
    "Tamil Nadu / Coromandel Coast": {
        "min_lat": 8.0, "min_lon": 78.0, "max_lat": 13.5, "max_lon": 82.0,
        "center": (10.8, 80.0), "zoom": 7,
    },
    "Bay of Bengal (India East)": {
        "min_lat": 8.0, "min_lon": 80.0, "max_lat": 21.0, "max_lon": 92.0,
        "center": (14.5, 86.0), "zoom": 5,
    },
    "Andhra Pradesh Coast": {
        "min_lat": 13.0, "min_lon": 80.0, "max_lat": 19.5, "max_lon": 85.5,
        "center": (16.3, 82.5), "zoom": 6,
    },
    "Kerala Coast": {
        "min_lat": 8.0, "min_lon": 74.0, "max_lat": 13.0, "max_lon": 77.5,
        "center": (10.5, 75.8), "zoom": 7,
    },
    "Karnataka Coast": {
        "min_lat": 12.5, "min_lon": 73.5, "max_lat": 15.5, "max_lon": 75.5,
        "center": (14.0, 74.3), "zoom": 7,
    },
    "Gujarat Coast": {
        "min_lat": 20.0, "min_lon": 68.0, "max_lat": 24.0, "max_lon": 73.0,
        "center": (22.0, 70.0), "zoom": 6,
    },
    "Maharashtra / Goa Coast": {
        "min_lat": 14.5, "min_lon": 71.5, "max_lat": 20.5, "max_lon": 73.5,
        "center": (17.5, 72.6), "zoom": 6,
    },
    "Danish Waters (project historical region)": {
        "min_lat": 53.5, "min_lon": 3.0, "max_lat": 60.0, "max_lon": 16.5,
        "center": (56.5, 10.5), "zoom": 6,
    },
}

DEFAULT_REGION_NAME = "India (All Coasts — widest coverage)"


def get_region_bbox(region_name: str) -> Dict[str, float]:
    """Return the {min_lat,min_lon,max_lat,max_lon} box for a named region."""
    r = LIVE_AIS_REGIONS.get(region_name, LIVE_AIS_REGIONS[DEFAULT_REGION_NAME])
    return {k: r[k] for k in ("min_lat", "min_lon", "max_lat", "max_lon")}


def get_region_center_zoom(region_name: str):
    """Return (center (lat,lon), zoom) for a named region, for the map view."""
    r = LIVE_AIS_REGIONS.get(region_name, LIVE_AIS_REGIONS[DEFAULT_REGION_NAME])
    return r["center"], r["zoom"]


def build_subscription_message(api_key: str, bbox: Dict[str, float]) -> dict:
    """
    Build the AISStream subscription message.

    AISStream expects a bounding box as [[minLat,minLon],[maxLat,maxLon]] pairs
    inside "BoundingBoxes" (a list of boxes). We send a single box built from
    the configured LIVE_AIS_BBOX.
    """
    return {
        "APIKey": api_key,
        "BoundingBoxes": [[
            [bbox["min_lat"], bbox["min_lon"]],
            [bbox["max_lat"], bbox["max_lon"]],
        ]],
        # PositionReport gives us live movement; ShipStaticData (AIS msg type 5)
        # gives us vessel type, name, destination and ETA so the dashboard can
        # show "vessel details", not just coordinates.
        "FilterMessageTypes": ["PositionReport", "ShipStaticData"],
    }


# ---------------------------------------------------------------------------
# AIS ship-type decoding (ITU-R M.1371 "Type of ship and cargo" first digit).
# AIS has NO "departure time/location" field — it carries DESTINATION + ETA in
# the static/voyage message (type 5). We expose those honestly and add a
# monitor-side "first seen" time/position as a practical proxy for when the
# vessel entered our live monitoring window.
# ---------------------------------------------------------------------------

def decode_ship_type(code) -> str:
    """Map an AIS ship-type code (0-99) to a human-readable category.

    Only the first digit (the broad category) is interpreted; the second digit
    refines cargo/hazard and is not needed for a readable label. Returns
    "Unknown" when the code is missing or 0.
    """
    try:
        c = int(code)
    except (TypeError, ValueError):
        return "Unknown"
    if c <= 0:
        return "Unknown"
    tens = c // 10
    specific = {
        50: "Pilot Vessel", 51: "Search & Rescue", 52: "Tug",
        53: "Port Tender", 54: "Anti-pollution", 55: "Law Enforcement",
        58: "Medical Transport", 59: "Non-combatant Ship",
    }
    if c in specific:
        return specific[c]
    return {
        2: "Wing-in-Ground",
        3: "Fishing / Special",  # 30 = Fishing; refined below
        4: "High-Speed Craft",
        6: "Passenger",
        7: "Cargo",
        8: "Tanker",
        9: "Other",
    }.get(tens, "Other")


def ship_type_label(code) -> str:
    """Readable ship-type label with the fishing case called out explicitly."""
    try:
        c = int(code)
    except (TypeError, ValueError):
        return "Unknown"
    if c == 30:
        return "Fishing"
    if 31 <= c <= 32:
        return "Towing"
    if c == 33:
        return "Dredging"
    if c == 34:
        return "Diving Ops"
    if c == 35:
        return "Military Ops"
    if c == 36:
        return "Sailing"
    if c == 37:
        return "Pleasure Craft"
    return decode_ship_type(code)


def _decode_eta(msg: dict) -> Optional[str]:
    """Build a readable ETA string from AISStream ShipStaticData ETA fields.

    AIS ETA is month/day/hour/minute (no year). Returns None when unset
    (month 0 means "not available"). This is the vessel-reported ETA, not a
    guaranteed arrival.
    """
    eta = msg.get("Eta") or msg.get("ETA") or {}
    if not isinstance(eta, dict):
        return None
    mo = eta.get("Month", 0) or 0
    day = eta.get("Day", 0) or 0
    hr = eta.get("Hour", 24)
    mn = eta.get("Minute", 60)
    if not mo or not day:
        return None
    hr = 0 if hr in (24, None) else hr
    mn = 0 if mn in (60, None) else mn
    try:
        return f"{int(mo):02d}-{int(day):02d} {int(hr):02d}:{int(mn):02d} UTC"
    except (TypeError, ValueError):
        return None


def normalize_static(raw: dict) -> Optional[Dict]:
    """Normalize one AISStream ShipStaticData message into vessel metadata.

    Returns None for non-static messages or when the vessel id is missing.
    Values are vessel-reported and may be stale, blank, or inaccurate.
    """
    if not isinstance(raw, dict) or raw.get("MessageType") != "ShipStaticData":
        return None
    meta = raw.get("MetaData", {}) or {}
    sd = raw.get("Message", {}).get("ShipStaticData", {}) or {}
    mmsi = meta.get("MMSI", sd.get("UserID"))
    if mmsi is None:
        return None
    try:
        vessel_id = str(int(mmsi))
    except (TypeError, ValueError):
        vessel_id = str(mmsi).strip()
    if not vessel_id or vessel_id.lower() == "none":
        return None

    name = (sd.get("Name") or meta.get("ShipName") or "").strip()
    dest = (sd.get("Destination") or "").strip()
    try:
        draught = float(sd.get("MaximumStaticDraught")) or None
    except (TypeError, ValueError):
        draught = None

    return {
        "vessel_id":   vessel_id,
        "ship_name":   name or None,
        "ship_type":   ship_type_label(sd.get("Type")),
        "destination": dest or None,
        "eta":         _decode_eta(sd),
        "draught":     draught,
    }


# ---------------------------------------------------------------------------
# Message normalization
# ---------------------------------------------------------------------------

def normalize_message(raw: dict) -> Optional[Dict]:
    """
    Normalize one AISStream message into the internal IFDS schema:
    vessel_id, latitude, longitude, speed, heading, timestamp (+ extras).

    Returns None for non-position messages or messages missing required
    fields. Never fabricates values — missing optional fields become None so
    the existing cleaning layer can handle them (NaN semantics).
    """
    if not isinstance(raw, dict):
        return None
    if raw.get("MessageType") != "PositionReport":
        return None

    meta = raw.get("MetaData", {}) or {}
    pr = raw.get("Message", {}).get("PositionReport", {}) or {}

    mmsi = meta.get("MMSI", pr.get("UserID"))
    lat = pr.get("Latitude", meta.get("latitude"))
    lon = pr.get("Longitude", meta.get("longitude"))
    if mmsi is None or lat is None or lon is None:
        return None

    try:
        lat = float(lat)
        lon = float(lon)
    except (TypeError, ValueError):
        return None
    if not (-90.0 <= lat <= 90.0) or not (-180.0 <= lon <= 180.0):
        return None

    try:
        vessel_id = str(int(mmsi))
    except (TypeError, ValueError):
        vessel_id = str(mmsi).strip()
    if not vessel_id or vessel_id.lower() == "none":
        return None

    ts_raw = meta.get("time_utc")
    timestamp = None
    if ts_raw:
        try:
            timestamp = pd_to_timestamp(ts_raw)
        except Exception:
            timestamp = None
    if timestamp is None:
        timestamp = datetime.now(timezone.utc)

    def _num(v):
        try:
            f = float(v)
            return f if f == f else None  # filter NaN
        except (TypeError, ValueError):
            return None

    return {
        "vessel_id":  vessel_id,
        "latitude":   lat,
        "longitude":  lon,
        "speed":      _num(pr.get("Sog")),
        "heading":    _num(pr.get("TrueHeading", pr.get("Cog"))),
        "course":     _num(pr.get("Cog")),
        "nav_status": pr.get("NavigationalStatus"),
        "timestamp":  timestamp,
    }


def pd_to_timestamp(ts_raw: str):
    """Parse an AISStream ISO-ish timestamp string into a UTC datetime."""
    import pandas as pd
    parsed = pd.to_datetime(ts_raw, errors="coerce", utc=True)
    if parsed is None or parsed is pd.NaT:
        return None
    return parsed.to_pydatetime()


# ---------------------------------------------------------------------------
# Bounded, thread-safe vessel state manager
# ---------------------------------------------------------------------------

class VesselStateManager:
    """
    In-memory live vessel state, bounded per vessel (no unlimited growth).

    Thread-safe: update_vessel() is called from the background WebSocket
    thread; get_* methods are called from the Streamlit script thread.
    """

    def __init__(self, max_trajectory_points: int = LIVE_AIS_MAX_TRAJECTORY_POINTS):
        self._lock = threading.Lock()
        self._max_points = max_trajectory_points
        self._state: Dict[str, Dict] = {}
        self._tracks: Dict[str, Deque[Dict]] = {}
        self._static: Dict[str, Dict] = {}  # vessel metadata (type/name/dest/eta)
        self.message_count = 0

    def update_vessel(self, obs: Dict) -> None:
        """Update one vessel's current state + bounded trajectory from a
        normalized observation dict."""
        vid = obs["vessel_id"]
        with self._lock:
            prev = self._state.get(vid)
            # "first_seen_*" = when/where this vessel first entered OUR live
            # monitoring window. AIS has no departure field; this is an honest
            # monitor-side proxy, labelled as such in the UI.
            self._state[vid] = {
                "vessel_id":        vid,
                "latitude":         obs["latitude"],
                "longitude":        obs["longitude"],
                "speed":            obs.get("speed"),
                "heading":          obs.get("heading"),
                "course":           obs.get("course"),
                "timestamp":        obs["timestamp"],
                "prev_latitude":    prev["latitude"] if prev else None,
                "prev_longitude":   prev["longitude"] if prev else None,
                "prev_speed":       prev["speed"] if prev else None,
                "prev_heading":     prev["heading"] if prev else None,
                "last_ais_timestamp": obs["timestamp"],
                "message_count":    (prev["message_count"] + 1) if prev else 1,
                "first_seen_time":  prev["first_seen_time"] if prev else obs["timestamp"],
                "first_seen_lat":   prev["first_seen_lat"] if prev else obs["latitude"],
                "first_seen_lon":   prev["first_seen_lon"] if prev else obs["longitude"],
            }
            dq = self._tracks.setdefault(vid, deque(maxlen=self._max_points))
            dq.append({
                "latitude":  obs["latitude"],
                "longitude": obs["longitude"],
                "speed":     obs.get("speed"),
                "heading":   obs.get("heading"),
                "timestamp": obs["timestamp"],
            })
            self.message_count += 1

    def update_static(self, meta: Dict) -> None:
        """Store/merge vessel static metadata (type, name, destination, ETA).

        Non-empty values overwrite; empty values never clobber a known value.
        """
        vid = meta.get("vessel_id")
        if not vid:
            return
        with self._lock:
            cur = self._static.get(vid, {})
            for k in ("ship_name", "ship_type", "destination", "eta", "draught"):
                v = meta.get(k)
                if v not in (None, "", "Unknown"):
                    cur[k] = v
            cur.setdefault("ship_type", meta.get("ship_type", "Unknown"))
            self._static[vid] = cur

    def _with_static(self, vid: str, s: Dict) -> Dict:
        """Return a copy of state merged with any known static metadata."""
        out = dict(s)
        meta = self._static.get(vid, {})
        out["ship_name"]   = meta.get("ship_name")
        out["ship_type"]   = meta.get("ship_type", "Unknown")
        out["destination"] = meta.get("destination")
        out["eta"]         = meta.get("eta")
        out["draught"]     = meta.get("draught")
        return out

    def get_vessel_state(self, vessel_id: str) -> Optional[Dict]:
        with self._lock:
            s = self._state.get(str(vessel_id))
            return self._with_static(str(vessel_id), s) if s else None

    def get_all_active_vessels(self) -> Dict[str, Dict]:
        with self._lock:
            return {vid: self._with_static(vid, s) for vid, s in self._state.items()}

    def get_recent_track(self, vessel_id: str):
        import pandas as pd
        with self._lock:
            dq = self._tracks.get(str(vessel_id))
            points = list(dq) if dq else []
        if not points:
            return pd.DataFrame(columns=["latitude", "longitude", "speed",
                                         "heading", "timestamp"])
        return pd.DataFrame(points)

    def get_all_recent_tracks(self) -> Dict[str, "object"]:
        return {vid: self.get_recent_track(vid) for vid in list(self._state.keys())}

    def remove_stale_vessels(self, stale_seconds: int = LIVE_AIS_STALE_SECONDS) -> int:
        """Drop vessels not heard from in stale_seconds. Returns count removed."""
        now = datetime.now(timezone.utc)
        removed = 0
        with self._lock:
            stale_ids = []
            for vid, s in self._state.items():
                ts = s.get("timestamp")
                try:
                    age = (now - ts.replace(tzinfo=timezone.utc) if ts.tzinfo is None
                           else now - ts).total_seconds()
                except Exception:
                    age = 0
                if age > stale_seconds:
                    stale_ids.append(vid)
            for vid in stale_ids:
                self._state.pop(vid, None)
                self._tracks.pop(vid, None)
                self._static.pop(vid, None)
                removed += 1
        return removed

    def vessel_count(self) -> int:
        with self._lock:
            return len(self._state)


# ---------------------------------------------------------------------------
# WebSocket client (background thread)
# ---------------------------------------------------------------------------

class LiveAISClient:
    """
    Background AISStream WebSocket client.

    States: "disconnected" | "connecting" | "connected" | "error"
    """

    def __init__(self, api_key: Optional[str] = None,
                bbox: Optional[Dict[str, float]] = None,
                state_manager: Optional[VesselStateManager] = None):
        self.api_key = api_key or get_api_key()
        self.bbox = bbox or LIVE_AIS_BBOX
        self.state = state_manager or VesselStateManager()

        self.status = "disconnected"
        self.last_error: str = ""
        self.last_message_at: Optional[datetime] = None
        self.last_update_at: Optional[datetime] = None
        self.connected_since: Optional[datetime] = None
        self.reconnect_attempts = 0

        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()

    # ------------------------------------------------------------------
    def start(self) -> None:
        """Start the background connection thread (idempotent)."""
        if self._thread and self._thread.is_alive():
            return
        if not self.api_key:
            self.status = "error"
            self.last_error = "AISSTREAM_API_KEY not configured"
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """Signal a graceful shutdown of the background thread."""
        self._stop_event.set()

    def is_running(self) -> bool:
        return bool(self._thread and self._thread.is_alive())

    # ------------------------------------------------------------------
    def _run(self) -> None:
        import asyncio
        try:
            asyncio.run(self._connect_loop())
        except Exception as exc:  # defensive: never crash the host process
            self.status = "error"
            self.last_error = f"Live AIS client terminated: {exc}"

    async def _connect_loop(self) -> None:
        import websockets

        delay = 1.0
        while not self._stop_event.is_set():
            self.status = "connecting"
            try:
                async with websockets.connect(AISSTREAM_URL,
                                              open_timeout=10) as ws:
                    sub = build_subscription_message(self.api_key, self.bbox)
                    await ws.send(json.dumps(sub))
                    self.status = "connected"
                    self.connected_since = datetime.now(timezone.utc)
                    self.reconnect_attempts = 0
                    delay = 1.0  # reset backoff on a successful connection

                    while not self._stop_event.is_set():
                        try:
                            raw = await asyncio.wait_for(ws.recv(), timeout=5)
                        except asyncio.TimeoutError:
                            continue
                        self.last_message_at = datetime.now(timezone.utc)
                        try:
                            msg = json.loads(raw)
                        except json.JSONDecodeError:
                            continue
                        if _looks_like_auth_error(msg):
                            self.status = "error"
                            self.last_error = ("Unable to authenticate with the "
                                              "AIS streaming provider. Please "
                                              "verify the configured API key.")
                            return
                        mtype = msg.get("MessageType")
                        if mtype == "ShipStaticData":
                            stat = normalize_static(msg)
                            if stat is not None:
                                self.state.update_static(stat)
                            continue
                        obs = normalize_message(msg)
                        if obs is not None:
                            self.state.update_vessel(obs)
                            self.last_update_at = datetime.now(timezone.utc)
            except Exception as exc:
                if self._stop_event.is_set():
                    break
                self.status = "error"
                self.last_error = _safe_error_text(exc)

            if self._stop_event.is_set():
                break
            self.reconnect_attempts += 1
            await asyncio.sleep(delay)
            delay = min(delay * 2, _MAX_BACKOFF_SECONDS)

        self.status = "disconnected"

    # ------------------------------------------------------------------
    def stats(self) -> Dict:
        """Lightweight runtime statistics for the dashboard (no secrets)."""
        return {
            "status":             self.status,
            "last_error":         self.last_error,
            "vessel_count":       self.state.vessel_count(),
            "message_count":      self.state.message_count,
            "last_message_at":    self.last_message_at,
            "last_update_at":     self.last_update_at,
            "connected_since":    self.connected_since,
            "reconnect_attempts": self.reconnect_attempts,
            "bbox":               self.bbox,
        }


def _looks_like_auth_error(msg: dict) -> bool:
    if not isinstance(msg, dict):
        return False
    err = str(msg.get("error", "")).lower()
    return "key" in err and ("invalid" in err or "unauthor" in err or "auth" in err)


def _safe_error_text(exc: Exception) -> str:
    """Return a short, non-sensitive error description (never leaks the key)."""
    text = str(exc)
    return text[:200] if text else exc.__class__.__name__
