"""
ais_loader.py
Historical AIS data ingestion for the Illegal Fishing Detection System.

Round 3 — Milestone M1.

Responsibility
--------------
Load historical AIS vessel-movement data from a local CSV file and convert it
into the raw data contract already used by the detection pipeline:

    load_vessel_dataframe(csv_path)      -> normalized pandas DataFrame
    build_trajectories_from_ais(df)      -> {vessel_id: [(lat, lon), ...]}

This module ONLY handles ingestion + light validation. It does not perform
geofencing, feature engineering, anomaly detection, or risk scoring — those
remain in their existing modules and consume this module's output unchanged.

Data honesty
------------
AIS data describes vessel *movement*. It does not, by itself, prove illegal
fishing. This loader deliberately does NOT fabricate behaviour labels such as
"Fishing" or "Illegal Fishing". Suspicious-behaviour detection is left entirely
to the downstream geofencing / behavioural / Isolation Forest / risk pipeline.

No live AIS feed, no API key, and no network access are involved. The loader
reads a historical CSV supplied locally.
"""

from __future__ import annotations

import os
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Repository-relative default path. The application handles the file not
# existing — see app.py integration. No machine-specific absolute paths.
#
# This points at the historical AIS sample (Danish Maritime Authority AIS data,
# 2025-02-27). The tiny synthetic data/sample_ais.csv is retained ONLY for the
# automated test suite (test_ais_loader.py references it directly).
DEFAULT_AIS_CSV_PATH: str = os.path.join("data", "ifds_ais_sample.csv")

# Normalized internal column names the rest of the application expects.
NORMALIZED_COLUMNS: List[str] = [
    "vessel_id",
    "latitude",
    "longitude",
    "speed",
    "heading",
    "timestamp",
]

# Columns that MUST be resolvable from the source CSV.
REQUIRED_NORMALIZED: List[str] = ["vessel_id", "latitude", "longitude"]

# Column-mapping layer: maps common AIS source column names (lower-cased) to
# our normalized names. AIS datasets vary widely, so we accept several aliases.
# Matching is case-insensitive (keys here are lower-case).
COLUMN_ALIASES: Dict[str, str] = {
    # vessel identifier
    "mmsi":          "vessel_id",
    "vessel_id":     "vessel_id",
    "vesselid":      "vessel_id",
    "id":            "vessel_id",
    "ship_id":       "vessel_id",
    # latitude
    "lat":           "latitude",
    "latitude":      "latitude",
    "y":             "latitude",
    # longitude
    "lon":           "longitude",
    "long":          "longitude",
    "lng":           "longitude",
    "longitude":     "longitude",
    "x":             "longitude",
    # speed over ground
    "sog":           "speed",
    "speed":         "speed",
    "speed_over_ground": "speed",
    # course / heading over ground
    "cog":           "heading",
    "heading":       "heading",
    "course":        "heading",
    "course_over_ground": "heading",
    # timestamp
    "basedatetime":  "timestamp",
    "base_date_time": "timestamp",
    "timestamp":     "timestamp",
    "time":          "timestamp",
    "datetime":      "timestamp",
    "t":             "timestamp",
}


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------

class AISColumnError(ValueError):
    """Raised when a required AIS column cannot be resolved from the CSV."""


class AISFileError(FileNotFoundError):
    """Raised when the AIS CSV file does not exist."""


# ---------------------------------------------------------------------------
# Column normalization
# ---------------------------------------------------------------------------

def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Rename source columns to normalized names using COLUMN_ALIASES.

    Matching is case-insensitive and whitespace-trimmed. Columns that do not
    match any alias are preserved under their original name (so useful extra
    source columns are not lost), unless they collide with a normalized name.

    Raises
    ------
    AISColumnError
        If any REQUIRED_NORMALIZED column cannot be produced.
    """
    rename_map: Dict[str, str] = {}
    taken: set = set()

    for original in df.columns:
        key = str(original).strip().lower()
        if key in COLUMN_ALIASES:
            target = COLUMN_ALIASES[key]
            # First alias wins if two source columns map to the same target.
            if target not in taken:
                rename_map[original] = target
                taken.add(target)

    out = df.rename(columns=rename_map)

    # Verify all required columns are present after renaming.
    missing = [c for c in REQUIRED_NORMALIZED if c not in out.columns]
    if missing:
        available = ", ".join(str(c) for c in df.columns)
        raise AISColumnError(
            f"AIS CSV is missing required column(s): {', '.join(missing)}. "
            f"Expected one of the recognised aliases for each. "
            f"Columns found in file: [{available}]. "
            f"Recognised aliases → vessel_id: MMSI/vessel_id/id; "
            f"latitude: LAT/latitude/y; longitude: LON/longitude/x; "
            f"speed: SOG/speed; heading: COG/heading/course; "
            f"timestamp: BaseDateTime/timestamp/time."
        )
    return out


# ---------------------------------------------------------------------------
# Validation / coercion
# ---------------------------------------------------------------------------

def _coerce_and_validate(df: pd.DataFrame) -> pd.DataFrame:
    """
    Coerce normalized columns to the correct dtypes and drop invalid rows.

    Validation rules (basic M1 ingestion validation only):
      - vessel_id must be non-null and non-empty
      - latitude  must be numeric and within [-90, 90]
      - longitude must be numeric and within [-180, 180]
      - speed     coerced to numeric; invalid → NaN then filled with 0.0
      - heading   coerced to numeric; invalid → NaN then filled with 0.0
      - timestamp parsed when present; unparseable timestamps → row dropped
        ONLY if a timestamp column exists (if there is no timestamp column at
        all, a synthetic sequential order is used instead — see below).

    Invalid rows are DROPPED (never fabricated). The number of dropped rows is
    recorded on the returned DataFrame's .attrs["dropped_rows"].

    Returns
    -------
    pandas.DataFrame
        Cleaned DataFrame containing NORMALIZED_COLUMNS (timestamp may be a
        synthetic ordering column if the source had no timestamp).
    """
    original_len = len(df)
    work = df.copy()

    # ── vessel_id ──────────────────────────────────────────────────────────
    work["vessel_id"] = work["vessel_id"].astype(str).str.strip()
    work = work[work["vessel_id"].notna()]
    work = work[work["vessel_id"] != ""]
    work = work[work["vessel_id"].str.lower() != "nan"]

    # ── latitude / longitude ────────────────────────────────────────────────
    work["latitude"]  = pd.to_numeric(work["latitude"], errors="coerce")
    work["longitude"] = pd.to_numeric(work["longitude"], errors="coerce")
    work = work[work["latitude"].notna() & work["longitude"].notna()]
    work = work[(work["latitude"] >= -90) & (work["latitude"] <= 90)]
    work = work[(work["longitude"] >= -180) & (work["longitude"] <= 180)]

    # ── speed ────────────────────────────────────────────────────────────────
    if "speed" in work.columns:
        work["speed"] = pd.to_numeric(work["speed"], errors="coerce").fillna(0.0)
    else:
        # No speed column — provide a neutral 0.0 so downstream never crashes.
        # Documented: absence of speed is not fabricated movement.
        work["speed"] = 0.0

    # ── heading ──────────────────────────────────────────────────────────────
    if "heading" in work.columns:
        work["heading"] = pd.to_numeric(work["heading"], errors="coerce").fillna(0.0)
    else:
        work["heading"] = 0.0

    # ── timestamp ─────────────────────────────────────────────────────────────
    if "timestamp" in work.columns:
        # dayfirst=True handles DD/MM/YYYY sources (e.g. Danish AIS exports)
        # while still parsing ISO timestamps correctly.
        work["timestamp"] = pd.to_datetime(
            work["timestamp"], errors="coerce", dayfirst=True
        )
        # Drop rows whose timestamp could not be parsed.
        work = work[work["timestamp"].notna()]
    else:
        # No timestamp column at all: create a stable sequential order so
        # trajectories still follow source row order deterministically.
        work = work.reset_index(drop=True)
        work["timestamp"] = pd.RangeIndex(start=0, stop=len(work), step=1)

    cleaned_len = len(work)
    work.attrs["dropped_rows"] = original_len - cleaned_len
    return work


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_vessel_dataframe(csv_path: str = DEFAULT_AIS_CSV_PATH) -> pd.DataFrame:
    """
    Load and normalize a historical AIS CSV into the pipeline's raw contract.

    Parameters
    ----------
    csv_path : str
        Path to a local historical AIS CSV file. Defaults to the
        repository-relative ``data/sample_ais.csv``.

    Returns
    -------
    pandas.DataFrame
        Normalized DataFrame sorted by (vessel_id, timestamp) with at least the
        columns:
            vessel_id, latitude, longitude, speed, heading, timestamp
        Any additional source columns that did not collide with normalized
        names are preserved. The DataFrame's ``.attrs["dropped_rows"]`` records
        how many invalid rows were filtered out.

    Raises
    ------
    AISFileError
        If ``csv_path`` does not exist.
    AISColumnError
        If a required column (vessel_id / latitude / longitude) is missing.

    Notes
    -----
    This function does NOT assign any behaviour label. AIS data describes
    movement only; suspicious-behaviour detection is done downstream.
    """
    if not os.path.exists(csv_path):
        raise AISFileError(
            f"Historical AIS dataset not found at '{csv_path}'. "
            f"Add a CSV file to the data/ directory or configure the AIS "
            f"dataset path."
        )

    raw = pd.read_csv(csv_path)
    normalized = _normalize_columns(raw)
    cleaned = _coerce_and_validate(normalized)

    # Deterministic ordering so each vessel's history is chronological.
    cleaned = cleaned.sort_values(["vessel_id", "timestamp"]).reset_index(drop=True)

    return cleaned


def build_trajectories_from_ais(
    df: pd.DataFrame,
) -> Dict[str, List[Tuple[float, float]]]:
    """
    Build per-vessel chronological trajectories from a normalized AIS DataFrame.

    Parameters
    ----------
    df : pandas.DataFrame
        A DataFrame produced by :func:`load_vessel_dataframe` (must contain
        vessel_id, latitude, longitude, and timestamp).

    Returns
    -------
    dict
        ``{vessel_id: [(latitude, longitude), ...]}`` with each vessel's points
        ordered chronologically by timestamp. Rows with invalid coordinates are
        ignored (they are already filtered by load_vessel_dataframe, but this is
        defensive).

    Notes
    -----
    Matches the output contract of
    ``data_generator.build_all_trajectories`` so the downstream pipeline is
    unchanged.
    """
    trajectories: Dict[str, List[Tuple[float, float]]] = {}

    if df.empty:
        return trajectories

    # Ensure chronological order per vessel.
    ordered = df.sort_values(["vessel_id", "timestamp"])

    for vid, group in ordered.groupby("vessel_id", sort=False):
        points: List[Tuple[float, float]] = []
        for lat, lon in zip(group["latitude"], group["longitude"]):
            if pd.isna(lat) or pd.isna(lon):
                continue
            if not (-90 <= lat <= 90 and -180 <= lon <= 180):
                continue
            points.append((float(lat), float(lon)))
        if points:
            trajectories[str(vid)] = points

    return trajectories


def summarize_latest_positions(df: pd.DataFrame) -> pd.DataFrame:
    """
    Reduce a multi-row-per-vessel AIS DataFrame to one row per vessel using the
    most recent (last chronological) position.

    This is useful because the downstream pipeline (geofencing, feature matrix,
    risk engine) expects ONE current position per vessel, while AIS data has
    many timestamped rows per vessel.

    Parameters
    ----------
    df : pandas.DataFrame
        Normalized AIS DataFrame from :func:`load_vessel_dataframe`.

    Returns
    -------
    pandas.DataFrame
        One row per vessel_id with columns:
            vessel_id, latitude, longitude, speed, heading, behavior
        ``behavior`` is set to the neutral value ``"AIS Track"`` purely so the
        existing map/trail code (which reads a ``behavior`` column) does not
        break. This is NOT a ground-truth fishing label — it only indicates the
        record originated from an AIS track. Real behaviour is derived
        downstream by the feature/risk pipeline.
    """
    if df.empty:
        return pd.DataFrame(
            columns=["vessel_id", "latitude", "longitude",
                     "speed", "heading", "behavior"]
        )

    ordered = df.sort_values(["vessel_id", "timestamp"])
    latest = ordered.groupby("vessel_id", sort=False).tail(1).copy()

    result = latest[["vessel_id", "latitude", "longitude",
                     "speed", "heading"]].reset_index(drop=True)

    # Neutral compatibility value — explicitly NOT a fishing/illegal label.
    result["behavior"] = "AIS Track"

    # Round for display consistency with the simulated path.
    result["latitude"]  = result["latitude"].round(4)
    result["longitude"] = result["longitude"].round(4)
    result["speed"]     = result["speed"].round(1)

    return result
