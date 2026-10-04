"""
realtime_simulator.py
AIS REPLAY SIMULATION (Round 3 — Milestone M8).

IMPORTANT TRUTH
---------------
This is NOT a live AIS feed. It is a controlled, deterministic REPLAY of the
cleaned historical AIS sample. Timestamps are played back in order; the replay
speed is configurable and does NOT wait for real-world time by default.

Design
------
- Load the cleaned historical AIS positions once (via the existing M1/M2
  ingestion + cleaning), sort globally by timestamp.
- Partition the ordered stream into N "ticks" (batches of chronological
  observations). Each advance() consumes one tick.
- Maintain current per-vessel state (latest position, speed, heading,
  timestamp, recent trajectory window). Only vessels seen in a tick are
  updated — the whole dataset is NOT recomputed every tick.
- The simulator holds no Streamlit state; the app drives it via session_state.
"""

from __future__ import annotations

from collections import deque
from typing import Deque, Dict, List, Optional

import pandas as pd

from ais_loader import load_vessel_dataframe, DEFAULT_AIS_CSV_PATH
from data_processing import clean_ais_dataframe

# Recent trajectory window kept per vessel (enough for behaviour/temporal).
RECENT_WINDOW: int = 20
DEFAULT_TICKS: int = 60          # number of replay steps across the dataset
DEFAULT_REPLAY_INTERVAL: float = 0.2   # seconds between ticks (UI hint only)


class AISReplaySimulator:
    """
    Deterministic replay of cleaned historical AIS observations.

    Usage
    -----
        sim = AISReplaySimulator.from_csv()
        batch = sim.advance()          # returns the observations in this tick
        state = sim.current_state()    # {vessel_id: latest obs dict}
        recent = sim.recent_track(vid) # DataFrame of recent positions
    """

    def __init__(self, ordered_df: pd.DataFrame, n_ticks: int = DEFAULT_TICKS):
        # ordered_df must be sorted by timestamp ascending.
        self._df = ordered_df.reset_index(drop=True)
        self._n = len(self._df)
        self.n_ticks = max(1, int(n_ticks))
        # Precompute tick boundaries (row index ranges) once — O(n) partition.
        self._bounds = self._compute_bounds()
        self.tick_index = 0
        # Live state
        self._state: Dict[str, Dict] = {}
        self._recent: Dict[str, Deque] = {}

    # ------------------------------------------------------------------
    @classmethod
    def from_csv(cls, csv_path: str = DEFAULT_AIS_CSV_PATH,
                 n_ticks: int = DEFAULT_TICKS) -> "AISReplaySimulator":
        """Load + clean the historical AIS sample and build a time-ordered replay."""
        raw = load_vessel_dataframe(csv_path)
        clean, _report = clean_ais_dataframe(raw)
        ordered = clean.sort_values("timestamp").reset_index(drop=True)
        return cls(ordered, n_ticks=n_ticks)

    @classmethod
    def from_dataframe(cls, clean_df: pd.DataFrame,
                       n_ticks: int = DEFAULT_TICKS) -> "AISReplaySimulator":
        """Build a replay from an already-cleaned DataFrame (used by tests)."""
        ordered = clean_df.sort_values("timestamp").reset_index(drop=True)
        return cls(ordered, n_ticks=n_ticks)

    # ------------------------------------------------------------------
    def _compute_bounds(self) -> List[int]:
        """Return n_ticks+1 row-index boundaries splitting the stream evenly."""
        if self._n == 0:
            return [0]
        step = max(1, self._n // self.n_ticks)
        bounds = list(range(0, self._n, step))
        if bounds[-1] != self._n:
            bounds.append(self._n)
        return bounds

    @property
    def total_ticks(self) -> int:
        return max(0, len(self._bounds) - 1)

    @property
    def finished(self) -> bool:
        return self.tick_index >= self.total_ticks

    @property
    def progress(self) -> float:
        return 0.0 if self.total_ticks == 0 else self.tick_index / self.total_ticks

    def current_sim_time(self) -> Optional[pd.Timestamp]:
        """Timestamp of the last observation consumed so far."""
        for vid in self._state:
            pass
        if not self._state:
            return None
        return max((s.get("timestamp") for s in self._state.values()
                    if s.get("timestamp") is not None), default=None)

    # ------------------------------------------------------------------
    def advance(self) -> List[Dict]:
        """
        Consume the next tick. Returns the list of observation dicts emitted,
        and updates per-vessel current state + recent trajectory windows.
        """
        if self.finished:
            return []
        lo = self._bounds[self.tick_index]
        hi = self._bounds[self.tick_index + 1]
        batch_df = self._df.iloc[lo:hi]
        emitted: List[Dict] = []
        for _, row in batch_df.iterrows():
            vid = str(row["vessel_id"])
            obs = {
                "vessel_id": vid,
                "timestamp": row["timestamp"],
                "latitude":  float(row["latitude"]),
                "longitude": float(row["longitude"]),
                "speed":     float(row["speed"]) if pd.notna(row["speed"]) else 0.0,
                "heading":   float(row["heading"]) if pd.notna(row["heading"]) else 0.0,
            }
            self._state[vid] = obs
            dq = self._recent.setdefault(vid, deque(maxlen=RECENT_WINDOW))
            dq.append(obs)
            emitted.append(obs)
        self.tick_index += 1
        return emitted

    def current_state(self) -> Dict[str, Dict]:
        """{vessel_id: latest observation dict} for all vessels seen so far."""
        return dict(self._state)

    def active_vessel_ids(self) -> List[str]:
        return list(self._state.keys())

    def recent_track(self, vessel_id: str) -> pd.DataFrame:
        """Recent trajectory window for a vessel as a DataFrame (chronological)."""
        dq = self._recent.get(str(vessel_id))
        if not dq:
            return pd.DataFrame(columns=["latitude", "longitude", "speed",
                                         "heading", "timestamp"])
        return pd.DataFrame(list(dq))[["latitude", "longitude", "speed",
                                       "heading", "timestamp"]]

    def state_dataframe(self) -> pd.DataFrame:
        """Current state as a one-row-per-vessel DataFrame (for geofencing etc.)."""
        if not self._state:
            return pd.DataFrame(columns=["vessel_id", "latitude", "longitude",
                                         "speed", "heading", "behavior"])
        rows = []
        for vid, obs in self._state.items():
            rows.append({
                "vessel_id": vid,
                "latitude":  round(obs["latitude"], 4),
                "longitude": round(obs["longitude"], 4),
                "speed":     round(obs["speed"], 1),
                "heading":   obs["heading"],
                "behavior":  "AIS Track",
            })
        return pd.DataFrame(rows)
