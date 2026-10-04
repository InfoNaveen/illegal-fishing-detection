"""
database.py
Lightweight SQLite persistence layer (Round 3 — Milestone M3).

Role
----
This module is a HISTORY / SINK layer. The in-memory detection pipeline runs
completely independently of it. If the database is unavailable, the pipeline
and dashboard must still work — callers use persist_pipeline_run() inside a
focused try/except and surface a warning on failure.

Design
------
- Standard-library sqlite3 only (no external dependency).
- Configurable database path (default: data/ifds.db).
- Parameterised SQL everywhere (never string-interpolate data values).
- Idempotent schema creation (CREATE TABLE IF NOT EXISTS), safe to call twice.
- Bulk inserts via executemany() inside a single transaction for speed.
- Duplicate protection:
    * vessels   : UNIQUE(vessel_id) + UPSERT (first_seen kept, last_seen bumped)
    * positions : UNIQUE(vessel_id, timestamp, latitude, longitude, source)
                  with INSERT OR IGNORE
    * features / anomaly_scores / risk_scores / alerts : append rows stamped
      with a computed_at run timestamp (each run is a distinct record).
"""

from __future__ import annotations

import json
import os
import sqlite3
from datetime import datetime, timezone
from typing import Dict, List, Optional

import pandas as pd

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DEFAULT_DB_PATH: str = os.path.join("data", "ifds.db")


class DatabaseError(Exception):
    """Raised when a persistence operation fails. Callers treat it as non-fatal."""


def _utc_now() -> str:
    """ISO-8601 UTC timestamp string used for computed_at / created_at."""
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# ---------------------------------------------------------------------------
# Connection + schema
# ---------------------------------------------------------------------------

def get_connection(db_path: str = DEFAULT_DB_PATH) -> sqlite3.Connection:
    """
    Open (and create if needed) a SQLite connection.

    Creates the parent directory if it does not exist. Enables foreign keys
    softly (schema does not depend on them being enforced).
    """
    parent = os.path.dirname(os.path.abspath(db_path))
    if parent and not os.path.exists(parent):
        os.makedirs(parent, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        conn.execute("PRAGMA foreign_keys = ON;")
    except sqlite3.Error:
        pass  # non-fatal
    return conn


_SCHEMA = """
CREATE TABLE IF NOT EXISTS vessels (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    vessel_id  TEXT NOT NULL UNIQUE,
    first_seen TEXT,
    last_seen  TEXT,
    source     TEXT,
    region     TEXT
);

CREATE TABLE IF NOT EXISTS positions (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    vessel_id  TEXT NOT NULL,
    timestamp  TEXT,
    latitude   REAL NOT NULL,
    longitude  REAL NOT NULL,
    speed      REAL,
    heading    REAL,
    source     TEXT,
    UNIQUE (vessel_id, timestamp, latitude, longitude, source)
);

CREATE TABLE IF NOT EXISTS features (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    vessel_id    TEXT NOT NULL,
    computed_at  TEXT NOT NULL,
    feature_json TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS anomaly_scores (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    vessel_id     TEXT NOT NULL,
    computed_at   TEXT NOT NULL,
    anomaly_score REAL,
    is_anomalous  INTEGER
);

CREATE TABLE IF NOT EXISTS risk_scores (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    vessel_id       TEXT NOT NULL,
    computed_at     TEXT NOT NULL,
    risk_score      REAL,
    risk_level      TEXT,
    zone_status     TEXT,
    behavior        TEXT,
    ais_gap_minutes REAL
);

CREATE TABLE IF NOT EXISTS alerts (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    vessel_id  TEXT NOT NULL,
    created_at TEXT NOT NULL,
    level      TEXT,
    message    TEXT,
    reason     TEXT
);

CREATE INDEX IF NOT EXISTS idx_positions_vessel ON positions (vessel_id);
CREATE INDEX IF NOT EXISTS idx_risk_vessel      ON risk_scores (vessel_id, computed_at);
CREATE INDEX IF NOT EXISTS idx_alerts_created   ON alerts (created_at);
"""


def initialize_database(db_path: str = DEFAULT_DB_PATH) -> None:
    """
    Create the database directory and all tables if they do not exist.

    Safe to call repeatedly. Never deletes existing data.
    """
    try:
        conn = get_connection(db_path)
        try:
            conn.executescript(_SCHEMA)
            conn.commit()
        finally:
            conn.close()
    except sqlite3.Error as exc:
        raise DatabaseError(f"Failed to initialize database at {db_path}: {exc}") from exc


# ---------------------------------------------------------------------------
# Persistence (single transaction per run)
# ---------------------------------------------------------------------------

def persist_pipeline_run(
    enriched_df: pd.DataFrame,
    feature_df: pd.DataFrame,
    anomaly_map: Dict[str, Dict],
    risk_map: Dict[str, Dict],
    alerts: List[Dict],
    source: str,
    region: str,
    positions_df: Optional[pd.DataFrame] = None,
    db_path: str = DEFAULT_DB_PATH,
) -> Dict[str, int]:
    """
    Persist one completed pipeline run.

    Parameters
    ----------
    enriched_df  : vessel DataFrame with vessel_id and (for position fallback)
                   latitude/longitude/speed/heading.
    feature_df   : feature matrix indexed by vessel_id.
    anomaly_map  : {vessel_id: {anomaly_score, is_anomalous}}
    risk_map     : {vessel_id: {risk_score, risk_level, zone_status, behavior,
                                ais_gap_minutes}}
    alerts       : list of {vessel, level, message, reason}
    source       : "Simulated" | "Historical AIS"
    region       : region label from zones.get_region_meta()
    positions_df : optional cleaned positions to store (Historical AIS). If
                   None, one position per vessel is derived from enriched_df.
    db_path      : SQLite path.

    Returns
    -------
    dict of row counts actually written (deduplicated), e.g.
    {"vessels": n, "positions": n, "features": n, "anomaly_scores": n,
     "risk_scores": n, "alerts": n}

    Raises
    ------
    DatabaseError on any failure (caller treats it as non-fatal).
    """
    now = _utc_now()
    counts = {"vessels": 0, "positions": 0, "features": 0,
              "anomaly_scores": 0, "risk_scores": 0, "alerts": 0}

    try:
        conn = get_connection(db_path)
        conn.executescript(_SCHEMA)  # ensure tables exist (idempotent)
        try:
            cur = conn.cursor()

            vessel_ids = [str(v) for v in enriched_df["vessel_id"].tolist()]

            # ── vessels (upsert) ─────────────────────────────────────────
            vessel_rows = [(vid, now, now, source, region) for vid in vessel_ids]
            cur.executemany(
                """
                INSERT INTO vessels (vessel_id, first_seen, last_seen, source, region)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(vessel_id) DO UPDATE SET
                    last_seen = excluded.last_seen,
                    source    = excluded.source,
                    region    = excluded.region
                """,
                vessel_rows,
            )
            counts["vessels"] = len(vessel_rows)

            # ── positions (INSERT OR IGNORE for dedup) ───────────────────
            if positions_df is not None and not positions_df.empty:
                pos_rows = [
                    (
                        str(r["vessel_id"]),
                        str(r["timestamp"]) if "timestamp" in r and pd.notna(r["timestamp"]) else None,
                        float(r["latitude"]),
                        float(r["longitude"]),
                        float(r["speed"]) if "speed" in r and pd.notna(r["speed"]) else None,
                        float(r["heading"]) if "heading" in r and pd.notna(r["heading"]) else None,
                        source,
                    )
                    for _, r in positions_df.iterrows()
                ]
            else:
                # Fallback: one current position per vessel from enriched_df.
                pos_rows = [
                    (
                        str(r["vessel_id"]),
                        None,
                        float(r["latitude"]),
                        float(r["longitude"]),
                        float(r["speed"]) if "speed" in r and pd.notna(r["speed"]) else None,
                        float(r["heading"]) if "heading" in r and pd.notna(r["heading"]) else None,
                        source,
                    )
                    for _, r in enriched_df.iterrows()
                ]
            before = conn.total_changes
            cur.executemany(
                """
                INSERT OR IGNORE INTO positions
                    (vessel_id, timestamp, latitude, longitude, speed, heading, source)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                pos_rows,
            )
            counts["positions"] = conn.total_changes - before

            # ── features (JSON per vessel) ───────────────────────────────
            feat_rows = []
            for vid in vessel_ids:
                if vid in feature_df.index:
                    feat_dict = {k: float(v) for k, v in feature_df.loc[vid].items()}
                    feat_rows.append((vid, now, json.dumps(feat_dict)))
            cur.executemany(
                "INSERT INTO features (vessel_id, computed_at, feature_json) VALUES (?, ?, ?)",
                feat_rows,
            )
            counts["features"] = len(feat_rows)

            # ── anomaly scores ───────────────────────────────────────────
            anom_rows = [
                (vid, now,
                 float(anomaly_map[vid]["anomaly_score"]),
                 1 if anomaly_map[vid]["is_anomalous"] else 0)
                for vid in vessel_ids if vid in anomaly_map
            ]
            cur.executemany(
                """INSERT INTO anomaly_scores
                   (vessel_id, computed_at, anomaly_score, is_anomalous)
                   VALUES (?, ?, ?, ?)""",
                anom_rows,
            )
            counts["anomaly_scores"] = len(anom_rows)

            # ── risk scores ──────────────────────────────────────────────
            risk_rows = [
                (vid, now,
                 float(risk_map[vid]["risk_score"]),
                 risk_map[vid]["risk_level"],
                 risk_map[vid].get("zone_status"),
                 risk_map[vid].get("behavior"),
                 float(risk_map[vid].get("ais_gap_minutes", 0) or 0))
                for vid in vessel_ids if vid in risk_map
            ]
            cur.executemany(
                """INSERT INTO risk_scores
                   (vessel_id, computed_at, risk_score, risk_level,
                    zone_status, behavior, ais_gap_minutes)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                risk_rows,
            )
            counts["risk_scores"] = len(risk_rows)

            # ── alerts ───────────────────────────────────────────────────
            alert_rows = [
                (str(a["vessel"]), now, a.get("level"),
                 a.get("message"), a.get("reason"))
                for a in alerts
            ]
            cur.executemany(
                """INSERT INTO alerts (vessel_id, created_at, level, message, reason)
                   VALUES (?, ?, ?, ?, ?)""",
                alert_rows,
            )
            counts["alerts"] = len(alert_rows)

            conn.commit()
        finally:
            conn.close()
    except sqlite3.Error as exc:
        raise DatabaseError(f"Persistence failed: {exc}") from exc

    return counts


# ---------------------------------------------------------------------------
# Query helpers (for the dashboard History section)
# ---------------------------------------------------------------------------

def get_history_summary(db_path: str = DEFAULT_DB_PATH) -> Dict[str, int]:
    """Return aggregate counts for the History panel."""
    try:
        conn = get_connection(db_path)
        conn.executescript(_SCHEMA)
        try:
            cur = conn.cursor()
            def _count(sql: str) -> int:
                row = cur.execute(sql).fetchone()
                return int(row[0]) if row and row[0] is not None else 0
            return {
                "total_vessels":    _count("SELECT COUNT(*) FROM vessels"),
                "total_positions":  _count("SELECT COUNT(*) FROM positions"),
                "total_alerts":     _count("SELECT COUNT(*) FROM alerts"),
                "high_alerts":      _count("SELECT COUNT(*) FROM alerts WHERE level = 'HIGH'"),
                "total_risk_rows":  _count("SELECT COUNT(*) FROM risk_scores"),
            }
        finally:
            conn.close()
    except sqlite3.Error as exc:
        raise DatabaseError(f"History summary query failed: {exc}") from exc


def get_recent_alerts(limit: int = 10, db_path: str = DEFAULT_DB_PATH) -> List[Dict]:
    """Return the most recent alerts (newest first)."""
    try:
        conn = get_connection(db_path)
        conn.executescript(_SCHEMA)
        try:
            rows = conn.execute(
                """SELECT vessel_id, created_at, level, message, reason
                   FROM alerts ORDER BY id DESC LIMIT ?""",
                (int(limit),),
            ).fetchall()
            return [dict(r) for r in rows]
        finally:
            conn.close()
    except sqlite3.Error as exc:
        raise DatabaseError(f"Recent alerts query failed: {exc}") from exc


def get_recent_risk(limit: int = 10, db_path: str = DEFAULT_DB_PATH) -> List[Dict]:
    """Return the most recent risk assessments (newest first)."""
    try:
        conn = get_connection(db_path)
        conn.executescript(_SCHEMA)
        try:
            rows = conn.execute(
                """SELECT vessel_id, computed_at, risk_score, risk_level,
                          zone_status, behavior
                   FROM risk_scores ORDER BY id DESC LIMIT ?""",
                (int(limit),),
            ).fetchall()
            return [dict(r) for r in rows]
        finally:
            conn.close()
    except sqlite3.Error as exc:
        raise DatabaseError(f"Recent risk query failed: {exc}") from exc


def get_vessel_recent(vessel_id: str, limit: int = 5,
                      db_path: str = DEFAULT_DB_PATH) -> Dict[str, List[Dict]]:
    """
    Return recent risk assessments and alerts for a single vessel.

    Returns {"risk": [...], "alerts": [...]}.
    """
    try:
        conn = get_connection(db_path)
        conn.executescript(_SCHEMA)
        try:
            risk = conn.execute(
                """SELECT computed_at, risk_score, risk_level, zone_status, behavior
                   FROM risk_scores WHERE vessel_id = ?
                   ORDER BY id DESC LIMIT ?""",
                (str(vessel_id), int(limit)),
            ).fetchall()
            al = conn.execute(
                """SELECT created_at, level, message, reason
                   FROM alerts WHERE vessel_id = ?
                   ORDER BY id DESC LIMIT ?""",
                (str(vessel_id), int(limit)),
            ).fetchall()
            return {"risk": [dict(r) for r in risk],
                    "alerts": [dict(r) for r in al]}
        finally:
            conn.close()
    except sqlite3.Error as exc:
        raise DatabaseError(f"Vessel history query failed: {exc}") from exc
