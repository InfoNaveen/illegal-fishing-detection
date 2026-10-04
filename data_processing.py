"""
data_processing.py
AIS data cleaning and validation (Round 3 — Milestone M2).

Sits between historical AIS ingestion and the existing detection pipeline:

    historical CSV
          ↓
    ais_loader.load_vessel_dataframe()
          ↓
    data_processing.clean_ais_dataframe()   ← this module
          ↓
    trajectory construction + existing detection pipeline

Scope (deliberately conservative)
---------------------------------
This module performs *basic, deterministic* cleaning only. It does NOT
interpolate, resample, reconstruct, or infer anything. It never fabricates
movement values and never classifies vessels. Those concerns belong to later
milestones.

The simulated (Bay of Bengal) path does not use this module at all.
"""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Expected normalized columns (produced by ais_loader).
# ---------------------------------------------------------------------------
_REQUIRED = ["vessel_id", "latitude", "longitude"]


# ---------------------------------------------------------------------------
# Individual cleaning steps (each returns df + a count of rows removed).
# ---------------------------------------------------------------------------

def _normalize_timestamp(df: pd.DataFrame) -> Tuple[pd.DataFrame, int]:
    """
    Ensure a datetime ``timestamp`` column. The historical dataset uses
    ``DD/MM/YYYY HH:MM:SS`` so day-first parsing is used explicitly to avoid
    ambiguous interpretation. Rows with an unparseable timestamp are removed.

    If no timestamp column exists, a stable sequential ordering column is used
    (ais_loader already does this, but we are defensive here).
    """
    before = len(df)
    if "timestamp" not in df.columns:
        out = df.reset_index(drop=True).copy()
        out["timestamp"] = pd.RangeIndex(start=0, stop=len(out), step=1)
        return out, 0

    out = df.copy()
    if not pd.api.types.is_datetime64_any_dtype(out["timestamp"]):
        # The historical dataset is uniformly "DD/MM/YYYY HH:MM:SS". Try that
        # explicit format first (fast, unambiguous, no dateutil warning); fall
        # back to day-first coercion for any rows that don't match.
        parsed = pd.to_datetime(
            out["timestamp"], format="%d/%m/%Y %H:%M:%S", errors="coerce"
        )
        unmatched = parsed.isna() & out["timestamp"].notna()
        if unmatched.any():
            parsed.loc[unmatched] = pd.to_datetime(
                out.loc[unmatched, "timestamp"], errors="coerce", dayfirst=True
            )
        out["timestamp"] = parsed
    out = out[out["timestamp"].notna()]
    return out, before - len(out)


def _coerce_numeric(df: pd.DataFrame) -> pd.DataFrame:
    """
    Coerce latitude/longitude/speed/heading to numeric.

    - latitude / longitude: coerced to numeric (invalid → NaN, dropped later).
    - speed / heading: coerced to numeric; genuinely missing AIS values become
      NaN. We do NOT fabricate movement values here. Safe defaults for the
      downstream pipeline (which currently expects numeric speed/heading) are
      applied only at the very end, and that substitution is documented.
    """
    out = df.copy()
    for col in ("latitude", "longitude"):
        out[col] = pd.to_numeric(out[col], errors="coerce")
    for col in ("speed", "heading"):
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
        else:
            out[col] = np.nan
    return out


def _validate_coordinates(df: pd.DataFrame) -> Tuple[pd.DataFrame, int, int]:
    """
    Remove rows with missing vessel ID, missing coordinates, or out-of-range
    coordinates (lat ∉ [-90, 90], lon ∉ [-180, 180]).

    Invalid coordinates are NEVER replaced with fabricated values — the row is
    dropped.

    Returns
    -------
    (cleaned_df, missing_coord_rows_removed, out_of_range_rows_removed)
    """
    out = df.copy()

    # Missing vessel id
    out["vessel_id"] = out["vessel_id"].astype(str).str.strip()
    out = out[(out["vessel_id"] != "") & (out["vessel_id"].str.lower() != "nan")]

    # Missing coordinates
    before_missing = len(out)
    out = out[out["latitude"].notna() & out["longitude"].notna()]
    missing_removed = before_missing - len(out)

    # Out-of-range coordinates
    before_range = len(out)
    out = out[(out["latitude"] >= -90) & (out["latitude"] <= 90)]
    out = out[(out["longitude"] >= -180) & (out["longitude"] <= 180)]
    range_removed = before_range - len(out)

    return out, missing_removed, range_removed


def _remove_negative_speed(df: pd.DataFrame) -> Tuple[pd.DataFrame, int]:
    """
    Remove rows with negative SOG (speed over ground), which is physically
    invalid in AIS. High-but-positive speeds are kept — anomaly detection, not
    an arbitrary threshold here, decides whether they are suspicious. NaN speeds
    are kept at this stage (they are not "negative").
    """
    before = len(df)
    out = df[~(df["speed"] < 0)].copy()   # NaN < 0 is False, so NaN rows kept
    return out, before - len(out)


def _normalize_heading(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalise heading/course to the circular range [0, 360).

    Heading 0 is a VALID direction and is never treated as missing. Values
    outside the range are wrapped modulo 360. NaN headings are left as NaN.
    AIS commonly uses 511 to mean "not available"; such sentinel values are
    converted to NaN (not fabricated to a real direction).
    """
    out = df.copy()
    # AIS "heading not available" sentinel.
    out.loc[out["heading"] == 511, "heading"] = np.nan
    mask = out["heading"].notna()
    out.loc[mask, "heading"] = out.loc[mask, "heading"].mod(360)
    return out


def _remove_exact_duplicates(df: pd.DataFrame) -> Tuple[pd.DataFrame, int]:
    """
    Remove exact duplicate records per vessel: identical
    (vessel_id, timestamp, latitude, longitude).

    Legitimate repeated positions where a vessel is stationary across DIFFERENT
    timestamps are preserved — only rows that duplicate the same timestamp AND
    position are dropped.
    """
    before = len(df)
    out = df.drop_duplicates(
        subset=["vessel_id", "timestamp", "latitude", "longitude"],
        keep="first",
    )
    return out, before - len(out)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def clean_ais_dataframe(df: pd.DataFrame,
                        fill_defaults: bool = True) -> Tuple[pd.DataFrame, Dict]:
    """
    Clean a normalized AIS DataFrame (from ais_loader) deterministically.

    Steps, in order:
      1. timestamp normalisation (day-first; drop unparseable)
      2. numeric coercion (lat/lon/speed/heading)
      3. coordinate validation (drop missing id / missing / out-of-range coords)
      4. negative-speed removal
      5. heading normalisation (wrap to [0,360); 511 sentinel → NaN)
      6. exact-duplicate removal
      7. chronological ordering by (vessel_id, timestamp)

    Parameters
    ----------
    df : pandas.DataFrame
        Normalized AIS data (needs at least vessel_id, latitude, longitude).
    fill_defaults : bool
        If True (default), speed/heading NaN are filled with 0.0 at the END so
        the existing downstream pipeline (which expects numeric values) does not
        break. This substitution is a compatibility measure, not fabricated
        movement data, and is reflected in the quality report
        (``speed_defaults_filled`` / ``heading_defaults_filled``).

    Returns
    -------
    (cleaned_df, quality_report)
        cleaned_df   : cleaned, chronologically ordered DataFrame
        quality_report : deterministic dict of cleaning statistics
    """
    if df is None or df.empty:
        empty = pd.DataFrame(
            columns=["vessel_id", "latitude", "longitude",
                     "speed", "heading", "timestamp"]
        )
        return empty, {
            "input_rows": 0, "output_rows": 0, "rows_removed": 0,
            "missing_timestamp_rows": 0, "invalid_coordinates": 0,
            "missing_coordinate_rows": 0, "invalid_speed_rows": 0,
            "duplicate_rows": 0, "unique_vessels": 0,
            "speed_defaults_filled": 0, "heading_defaults_filled": 0,
        }

    missing = [c for c in _REQUIRED if c not in df.columns]
    if missing:
        raise ValueError(
            f"clean_ais_dataframe: input is missing required column(s): "
            f"{', '.join(missing)}"
        )

    input_rows = len(df)

    # 1. timestamp
    work, ts_removed = _normalize_timestamp(df)
    # 2. numeric
    work = _coerce_numeric(work)
    # 3. coordinates
    work, missing_coord_removed, range_removed = _validate_coordinates(work)
    # 4. negative speed
    work, neg_speed_removed = _remove_negative_speed(work)
    # 5. heading
    work = _normalize_heading(work)
    # 6. exact duplicates
    work, dup_removed = _remove_exact_duplicates(work)
    # 7. chronological ordering
    work = work.sort_values(["vessel_id", "timestamp"]).reset_index(drop=True)

    # Count NaNs BEFORE optional default fill (for honest reporting).
    speed_nan = int(work["speed"].isna().sum())
    heading_nan = int(work["heading"].isna().sum())

    if fill_defaults:
        work["speed"] = work["speed"].fillna(0.0)
        work["heading"] = work["heading"].fillna(0.0)

    report: Dict = {
        "input_rows":              int(input_rows),
        "output_rows":             int(len(work)),
        "rows_removed":            int(input_rows - len(work)),
        "missing_timestamp_rows":  int(ts_removed),
        "missing_coordinate_rows": int(missing_coord_removed),
        "invalid_coordinates":     int(range_removed),
        "invalid_speed_rows":      int(neg_speed_removed),
        "duplicate_rows":          int(dup_removed),
        "unique_vessels":          int(work["vessel_id"].nunique()),
        "speed_defaults_filled":   speed_nan if fill_defaults else 0,
        "heading_defaults_filled": heading_nan if fill_defaults else 0,
    }

    return work, report
