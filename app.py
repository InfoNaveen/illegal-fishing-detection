"""
app.py
Main Streamlit dashboard for the Illegal Fishing Detection System.

Pipeline (all computed once and cached):
  1. generate_vessel_dataframe()     → raw vessel fields
  2. build_all_trajectories()        → {vessel_id: [(lat,lon),...]}
  3. run_geofencing()                → {vessel_id: geo_result}
  4. build_feature_matrix()          → feature DataFrame
  5. run_anomaly_detection()         → {vessel_id: {anomaly_score, is_anomalous}}
  6. run_risk_engine()               → {vessel_id: risk_result}
  7. generate_alerts()               → [alert_dict, ...]

The resulting enriched DataFrame (with risk_score, risk_level, behavior,
zone_status columns added) is passed to the unchanged map_builder.build_map().
"""

import streamlit as st
from streamlit_folium import st_folium
import pandas as pd

# ── Pipeline imports ────────────────────────────────────────────────────────
from data_generator import (
    generate_vessel_dataframe,
    build_all_trajectories,
    get_restricted_zones,
)
from geofencing import run_geofencing
from feature_engineering import build_feature_matrix
from anomaly_detector import run_anomaly_detection, run_anomaly_detection_with_status
from risk_engine import run_risk_engine, generate_alerts
from map_builder import build_map

# ── Round 3 M1: historical AIS ingestion (additive, optional path) ──────────
from ais_loader import (
    load_vessel_dataframe,
    build_trajectories_from_ais,
    summarize_latest_positions,
    DEFAULT_AIS_CSV_PATH,
    AISFileError,
    AISColumnError,
)
# ── Round 3 M2: AIS cleaning + regional zone configuration ──────────────────
from data_processing import clean_ais_dataframe
from zones import get_zones, get_region_meta
# ── Round 3 M3: SQLite persistence (history sink — never a hard dependency) ──
import database
from database import DatabaseError, DEFAULT_DB_PATH
# ── Round 3 M4: trajectory & behaviour analysis ─────────────────────────────
from behavior_analysis import analyze_all
# ── Round 3 M6: temporal sequence anomaly detection ─────────────────────────
from temporal_model_manager import run_temporal_detection

# ---------------------------------------------------------------------------
# Page config — must be the first Streamlit call
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Illegal Fishing Detection System",
    page_icon="🛥️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Global CSS — dark maritime theme (unchanged from Round 1)
# ---------------------------------------------------------------------------
st.markdown("""
<style>
/* ── Base ────────────────────────────────────────────────── */
html, body, [data-testid="stApp"] {
    background-color: #080c1a;
    color: #d0d8e8;
    font-family: 'Segoe UI', monospace, sans-serif;
}
/* ── Sidebar ─────────────────────────────────────────────── */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0d1226 0%, #0a0f1e 100%);
    border-right: 1px solid #1e2d50;
}
[data-testid="stSidebar"] * { color: #c8d6f0 !important; }
/* ── Metric cards ─────────────────────────────────────────── */
[data-testid="metric-container"] {
    background: linear-gradient(135deg, #0d1b35 0%, #0a1525 100%);
    border: 1px solid #1e3a6e;
    border-radius: 10px;
    padding: 12px 16px;
    box-shadow: 0 4px 15px rgba(0,100,255,0.08);
}
[data-testid="stMetricValue"] { color: #00d4ff !important; font-size: 2rem !important; }
[data-testid="stMetricLabel"] { color: #8099bb !important; font-size: 0.75rem !important; letter-spacing: 0.05em; }
/* ── Section headers ──────────────────────────────────────── */
.section-header {
    background: linear-gradient(90deg, #0d2550 0%, #080c1a 100%);
    border-left: 4px solid #00d4ff;
    padding: 8px 14px;
    border-radius: 0 6px 6px 0;
    margin-bottom: 12px;
    font-size: 0.9rem;
    font-weight: 600;
    letter-spacing: 0.08em;
    color: #00d4ff;
    text-transform: uppercase;
}
/* ── Info cards ───────────────────────────────────────────── */
.info-card {
    background: #0d1b35;
    border: 1px solid #1e3a6e;
    border-radius: 8px;
    padding: 14px;
    margin-bottom: 10px;
}
.info-row {
    display: flex;
    justify-content: space-between;
    padding: 4px 0;
    border-bottom: 1px solid #1a2840;
    font-size: 0.82rem;
}
.info-label { color: #7a94b8; }
.info-value { color: #e0eaff; font-weight: 600; }
/* ── Alert cards ──────────────────────────────────────────── */
.alert-high {
    background: linear-gradient(90deg, #2a0808 0%, #1a0505 100%);
    border-left: 4px solid #ff3333;
    border-radius: 0 6px 6px 0;
    padding: 10px 14px;
    margin-bottom: 8px;
    font-size: 0.82rem;
}
.alert-medium {
    background: linear-gradient(90deg, #2a1500 0%, #1a0e00 100%);
    border-left: 4px solid #ff8c00;
    border-radius: 0 6px 6px 0;
    padding: 10px 14px;
    margin-bottom: 8px;
    font-size: 0.82rem;
}
/* ── Risk bar ─────────────────────────────────────────────── */
.risk-bar-wrap {
    background: #0a1525;
    border-radius: 4px;
    height: 8px;
    width: 100%;
    margin-top: 4px;
}
.risk-bar-fill { height: 8px; border-radius: 4px; }
/* ── Selectbox / button ───────────────────────────────────── */
[data-testid="stSelectbox"] > div {
    background: #0d1b35 !important;
    border: 1px solid #1e3a6e !important;
    border-radius: 6px !important;
    color: #d0d8e8 !important;
}
.stButton > button {
    background: linear-gradient(90deg, #0066cc, #0044aa);
    color: white; border: none; border-radius: 6px;
    font-weight: 600; letter-spacing: 0.05em;
}
/* ── Scrollbar ────────────────────────────────────────────── */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: #080c1a; }
::-webkit-scrollbar-thumb { background: #1e3a6e; border-radius: 3px; }
/* ── Hide Streamlit branding ──────────────────────────────── */
#MainMenu, footer, header { visibility: hidden; }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Full detection pipeline  (cached — runs once per session)
# ---------------------------------------------------------------------------

@st.cache_data
def run_pipeline(source: str = "Simulated",
                 csv_path: str = DEFAULT_AIS_CSV_PATH) -> tuple:
    """
    Execute the complete detection pipeline.

    Parameters
    ----------
    source : str
        "Simulated" (default) uses the built-in synthetic vessel generator.
        "Historical AIS" loads vessel movement data from a local CSV via
        ais_loader and feeds it through the SAME downstream pipeline.
    csv_path : str
        Path to the historical AIS CSV (only used when source == "Historical AIS").

    Returns
    -------
    df          : enriched vessel DataFrame (all columns the UI needs)
    alerts      : list of dynamic alert dicts
    anomaly_map : {vessel_id: {anomaly_score, is_anomalous}}
    risk_map    : {vessel_id: risk_result_dict}
    feature_df  : feature DataFrame (used by ML details panel)
    meta        : dict of source metadata (zones, region, data-quality report)

    Notes
    -----
    Only the raw-data + trajectory stage and the zone configuration differ
    between the two sources. Everything from geofencing onward is identical and
    unchanged. The simulated path is untouched.
    """
    # Zone configuration is source-aware (zones.py). Geofencing algorithm
    # itself is unchanged.
    zones = get_zones(source)
    quality_report: dict = {}
    positions_df = None   # cleaned positions to persist (Historical AIS only)

    # map_trajectories: {vessel_id: [(lat,lon),...]} — real historical tracks
    # (AIS) or synthetic trails (simulated), used by the map for the selected
    # vessel. behavior_trajectories feed the M4 behaviour analysis.
    map_trajectories: dict = {}

    # 1 + 2. Raw vessel data + trajectories (source-dependent).
    if source == "Historical AIS":
        ais_df = load_vessel_dataframe(csv_path)             # may raise AISFileError/AISColumnError
        # M2: deterministic cleaning between ingestion and the pipeline.
        clean_df, quality_report = clean_ais_dataframe(ais_df)
        trajectories = build_trajectories_from_ais(clean_df) # full multi-point tracks (cleaned)
        base_df      = summarize_latest_positions(clean_df)  # one current row per vessel
        # Keep the cleaned positions for persistence (M3). Convert timestamp to
        # string here so the sink layer stores a stable value.
        positions_df = clean_df[["vessel_id", "timestamp",
                                 "latitude", "longitude",
                                 "speed", "heading"]].copy()
        positions_df["timestamp"] = positions_df["timestamp"].astype(str)
        # M4: behaviour analysis consumes time-aware DataFrame slices. Group the
        # cleaned frame ONCE (O(n)) into per-vessel slices to avoid O(n^2).
        behavior_trajectories = {
            str(vid): grp[["latitude", "longitude", "speed", "heading", "timestamp"]]
            for vid, grp in clean_df.groupby("vessel_id", sort=False)
        }
        map_trajectories = trajectories  # real chronological (lat,lon) tracks
    else:
        # Default simulated path — unchanged from Round 2.
        base_df      = generate_vessel_dataframe()
        trajectories = build_all_trajectories(base_df)
        # Simulated behaviour uses the (lat,lon) tuple trajectories directly.
        behavior_trajectories = trajectories
        map_trajectories = trajectories

    # 2b. Behaviour analysis (M4) — per-vessel, operates on ordered tracks.
    behavior_map = analyze_all(behavior_trajectories)

    # 2c. Temporal sequence tracks (M6). Historical AIS already has time-aware
    # DataFrame slices; simulated tuple trajectories become minimal DataFrames.
    if source == "Historical AIS":
        temporal_tracks = behavior_trajectories
    else:
        temporal_tracks = {
            str(vid): pd.DataFrame(pts, columns=["latitude", "longitude"])
            for vid, pts in behavior_trajectories.items()
        }

    # 3. Geofencing (same algorithm, source-appropriate zones)
    geo_list = run_geofencing(base_df, zones)
    geo_map  = {g["vessel_id"]: g for g in geo_list}

    # 4. Feature extraction (now includes numeric behavioural features — M4)
    feature_df = build_feature_matrix(base_df, trajectories, geo_map,
                                      behavior_map=behavior_map)

    # 5. Isolation Forest anomaly detection (persistent model — M5)
    anomaly_map, model_status, model_meta = run_anomaly_detection_with_status(
        feature_df)

    # 5b. Temporal sequence anomaly detection (persistent model — M6)
    temporal_map, temporal_status, temporal_meta = run_temporal_detection(
        temporal_tracks)

    # 6. Unified risk engine (geofence + isolation + temporal + behaviour +
    #    AIS gap, each counted once — M7)
    vessel_ids = base_df["vessel_id"].tolist()
    risk_map   = run_risk_engine(vessel_ids, geo_map, feature_df, anomaly_map,
                                 behavior_map=behavior_map,
                                 temporal_map=temporal_map)

    # 7. Enrich base DataFrame with pipeline outputs
    df = base_df.copy()
    df["risk_score"]  = df["vessel_id"].map(lambda v: risk_map[v]["risk_score"])
    df["risk_level"]  = df["vessel_id"].map(lambda v: risk_map[v]["risk_level"])
    df["behavior"]    = df["vessel_id"].map(lambda v: risk_map[v]["behavior"])
    df["zone_status"] = df["vessel_id"].map(lambda v: risk_map[v]["zone_status"])

    # 8. Dynamic alerts
    alerts = generate_alerts(risk_map)

    region_meta = get_region_meta(source)
    meta = {
        "source":         source,
        "zones":          zones,
        "region":         region_meta["region"],
        "map_center":     region_meta["map_center"],
        "map_zoom":       region_meta["map_zoom"],
        "zone_label":     region_meta["zone_label"],
        "quality_report": quality_report,
        "positions_df":   positions_df,
        "behavior_map":   behavior_map,
        "map_trajectories": map_trajectories,
        "model_status":   model_status,
        "model_meta":     model_meta,
        "temporal_map":   temporal_map,
        "temporal_status": temporal_status,
        "temporal_meta":  temporal_meta,
    }

    return df, alerts, anomaly_map, risk_map, feature_df, meta


# ---------------------------------------------------------------------------
# Data source selection (Round 3 M1) — chosen in the sidebar below, but read
# here so the pipeline runs with the correct source. Default is "Simulated".
# ---------------------------------------------------------------------------
if "data_source" not in st.session_state:
    st.session_state["data_source"] = "Simulated"

_active_source = st.session_state["data_source"]
_ais_error_message = ""

try:
    df, alerts, anomaly_map, risk_map, feature_df, pipeline_meta = run_pipeline(
        source=_active_source,
        csv_path=DEFAULT_AIS_CSV_PATH,
    )
except (AISFileError, AISColumnError) as exc:
    # Historical AIS unavailable/invalid — fall back to simulated so the
    # dashboard never crashes, and surface a clear message in the sidebar.
    _ais_error_message = str(exc)
    _active_source = "Simulated"
    df, alerts, anomaly_map, risk_map, feature_df, pipeline_meta = run_pipeline(
        source="Simulated",
        csv_path=DEFAULT_AIS_CSV_PATH,
    )


# ---------------------------------------------------------------------------
# M3 persistence boundary — focused try/except. SQLite is a history SINK:
# if it fails, the pipeline results above are untouched and the dashboard
# still renders. The write is cached per source so repeated Streamlit reruns
# do not re-insert on every widget interaction.
# ---------------------------------------------------------------------------
_persist_warning = ""


@st.cache_data(show_spinner=False)
def _persist_once(source: str, _df, _feat, _anom, _risk, _alerts, region, _positions):
    """Persist one pipeline run. Cached per source so it writes once per run.
    Returns the row-count dict. Underscored args are excluded from the cache
    key by Streamlit; `source` is the cache key."""
    database.initialize_database(DEFAULT_DB_PATH)
    return database.persist_pipeline_run(
        enriched_df=_df,
        feature_df=_feat,
        anomaly_map=_anom,
        risk_map=_risk,
        alerts=_alerts,
        source=source,
        region=region,
        positions_df=_positions,
        db_path=DEFAULT_DB_PATH,
    )


try:
    _persist_counts = _persist_once(
        _active_source, df, feature_df, anomaly_map, risk_map, alerts,
        pipeline_meta["region"], pipeline_meta.get("positions_df"),
    )
except DatabaseError as exc:
    _persist_warning = f"Persistence unavailable (results still shown): {exc}"
    _persist_counts = {}
except Exception as exc:  # defensive: persistence must never kill the app
    _persist_warning = f"Persistence skipped (results still shown): {exc}"
    _persist_counts = {}


# ---------------------------------------------------------------------------
# Dashboard statistics
# ---------------------------------------------------------------------------

def _compute_stats(df: pd.DataFrame, alerts: list) -> dict:
    high   = int((df["risk_level"] == "HIGH").sum())
    medium = int((df["risk_level"] == "MEDIUM").sum())
    low    = int((df["risk_level"] == "LOW").sum())
    active = sum(1 for a in alerts if a["level"] == "HIGH")
    return {
        "total_vessels": len(df),
        "high_risk":     high,
        "medium_risk":   medium,
        "low_risk":      low,
        "active_alerts": active,
    }

stats = _compute_stats(df, alerts)

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
_RISK_COLOURS = {"HIGH": "#FF3333", "MEDIUM": "#FF8C00", "LOW": "#2ECC71"}

with st.sidebar:
    st.markdown("""
    <div style="text-align:center;padding:16px 0 8px;">
      <div style="font-size:2.2rem;">🛥️</div>
      <div style="font-size:1.1rem;font-weight:700;color:#00d4ff;letter-spacing:0.08em;">
        IFDS
      </div>
      <div style="font-size:0.7rem;color:#4a6a9a;letter-spacing:0.12em;margin-top:2px;">
        ILLEGAL FISHING DETECTION SYSTEM
      </div>
    </div>
    <hr style="border-color:#1e2d50;margin:8px 0 16px;">
    """, unsafe_allow_html=True)

    # ── Data source selector (Round 3 M1) ───────────────────────────────────
    st.markdown('<div class="section-header">🛰️ Data Source</div>',
                unsafe_allow_html=True)
    st.radio(
        "Data Source",
        options=["Simulated", "Historical AIS"],
        key="data_source",
        label_visibility="collapsed",
    )
    if _active_source == "Historical AIS" and not _ais_error_message:
        _qr = pipeline_meta.get("quality_report", {})
        st.markdown(
            '<div style="font-size:0.68rem;color:#2ECC71;margin:-4px 0 6px;">'
            '● Historical AIS loaded</div>',
            unsafe_allow_html=True,
        )
        st.markdown(f"""
        <div class="info-card" style="padding:10px 12px;margin-bottom:10px;">
          <div class="info-row">
            <span class="info-label">Region</span>
            <span class="info-value">{pipeline_meta.get('region', 'Danish Waters')}</span>
          </div>
          <div class="info-row">
            <span class="info-label">Dataset</span>
            <span class="info-value" style="font-size:0.72rem;">2025-02-27 Historical AIS</span>
          </div>
          <div class="info-row">
            <span class="info-label">Records (raw)</span>
            <span class="info-value">{_qr.get('input_rows', 0):,}</span>
          </div>
          <div class="info-row">
            <span class="info-label">Records (clean)</span>
            <span class="info-value">{_qr.get('output_rows', 0):,}</span>
          </div>
          <div class="info-row">
            <span class="info-label">Rows removed</span>
            <span class="info-value">{_qr.get('rows_removed', 0):,}</span>
          </div>
          <div class="info-row" style="border-bottom:none;">
            <span class="info-label">Vessels</span>
            <span class="info-value">{_qr.get('unique_vessels', 0):,}</span>
          </div>
        </div>
        <div style="font-size:0.62rem;color:#5a7aa5;margin:-4px 0 10px;line-height:1.5;">
          Demonstration monitoring zones over Danish waters — not legally
          restricted areas. Movement data only; not proof of illegal fishing.
        </div>
        """, unsafe_allow_html=True)
    elif _ais_error_message:
        st.warning(
            "Historical AIS dataset not found or invalid. "
            "Add a CSV file to data/ or configure the AIS dataset path. "
            "Showing simulated data instead."
        )
    else:
        st.markdown(
            '<div style="font-size:0.68rem;color:#4a6a9a;margin:-4px 0 10px;">'
            '● Using built-in simulated fleet</div>',
            unsafe_allow_html=True,
        )

    # ── ML model status (M5) ─────────────────────────────────────────────
    _ms = pipeline_meta.get("model_status", "ready")
    _ms_label = {
        "loaded":    ("● Loaded existing model", "#2ECC71"),
        "trained":   ("● Trained new model",      "#00d4ff"),
        "retrained": ("● Retrained model (schema changed)", "#FF8C00"),
        "in-memory": ("● In-memory model",         "#8899bb"),
    }.get(_ms, ("● Model ready", "#8899bb"))
    _ts = pipeline_meta.get("temporal_status", "ready")
    _ts_label = {
        "loaded":    ("● Temporal model loaded", "#2ECC71"),
        "trained":   ("● Temporal model trained", "#00d4ff"),
        "retrained": ("● Temporal model retrained", "#FF8C00"),
    }.get(_ts, ("● Temporal model ready", "#8899bb"))
    st.markdown('<div class="section-header">🤖 ML Models</div>',
                unsafe_allow_html=True)
    st.markdown(
        f'<div style="font-size:0.7rem;color:{_ms_label[1]};margin:-4px 0 2px;">'
        f'{_ms_label[0]}</div>'
        f'<div style="font-size:0.64rem;color:#4a6a9a;margin-bottom:4px;">'
        f'Isolation Forest · point anomaly</div>'
        f'<div style="font-size:0.7rem;color:{_ts_label[1]};margin:0 0 2px;">'
        f'{_ts_label[0]}</div>'
        f'<div style="font-size:0.64rem;color:#4a6a9a;margin-bottom:10px;">'
        f'PCA sequence autoencoder · temporal anomaly</div>',
        unsafe_allow_html=True,
    )

    st.markdown('<div class="section-header">🔍 Vessel Inspector</div>',
                unsafe_allow_html=True)

    vessel_ids_sorted = ["— Select a vessel —"] + sorted(df["vessel_id"].tolist())
    selected_vessel   = st.selectbox("Vessel ID", vessel_ids_sorted,
                                     label_visibility="collapsed")

    if selected_vessel != "— Select a vessel —":
        row         = df[df["vessel_id"] == selected_vessel].iloc[0]
        risk_colour = _RISK_COLOURS.get(row["risk_level"], "#888")
        risk_result = risk_map[selected_vessel]
        anom        = anomaly_map[selected_vessel]
        _temporal   = pipeline_meta.get("temporal_map", {}).get(
            selected_vessel, {"temporal_anomaly_score": 0.0,
                              "temporal_model_status": "insufficient_data"})

        st.markdown(f"""
        <div class="info-card">
          <div style="color:#00d4ff;font-weight:700;font-size:1rem;margin-bottom:10px;">
            ⚓ {row['vessel_id']}
          </div>
          <div class="info-row">
            <span class="info-label">Speed</span>
            <span class="info-value">{row['speed']} kn</span>
          </div>
          <div class="info-row">
            <span class="info-label">Heading</span>
            <span class="info-value">{row['heading']}°</span>
          </div>
          <div class="info-row">
            <span class="info-label">Latitude</span>
            <span class="info-value">{row['latitude']}°N</span>
          </div>
          <div class="info-row">
            <span class="info-label">Longitude</span>
            <span class="info-value">{row['longitude']}°E</span>
          </div>
          <div class="info-row">
            <span class="info-label">Risk Score</span>
            <span class="info-value" style="color:{risk_colour};">
              {row['risk_score']}/100
            </span>
          </div>
          <div class="info-row">
            <span class="info-label">Risk Level</span>
            <span class="info-value" style="color:{risk_colour};">
              {row['risk_level']}
            </span>
          </div>
          <div class="info-row">
            <span class="info-label">Zone Status</span>
            <span class="info-value" style="color:#ffd700;font-size:0.75rem;">
              {row['zone_status']}
            </span>
          </div>
          <div class="info-row">
            <span class="info-label">Behavior</span>
            <span class="info-value" style="font-size:0.75rem;">{row['behavior']}</span>
          </div>
          <div class="info-row" style="border-bottom:none;">
            <span class="info-label">IF Anomaly Score</span>
            <span class="info-value" style="color:#a78bfa;">
              {anom['anomaly_score']:.3f}
              {"  🔺" if anom['is_anomalous'] else ""}
            </span>
          </div>
          <div class="info-row" style="border-bottom:none;">
            <span class="info-label">Temporal Anomaly</span>
            <span class="info-value" style="color:#7fd4ff;">
              {_temporal['temporal_anomaly_score']:.3f}
              {"" if _temporal['temporal_model_status'] == "scored" else "  (n/a)"}
            </span>
          </div>
          {f'''<div class="info-row" style="border-bottom:none;margin-top:4px;">
            <span class="info-label">AIS Gap</span>
            <span class="info-value" style="color:#ffcc00;font-size:0.75rem;">
              ⚠️ {risk_result["ais_gap_minutes"]} min blackout
            </span>
          </div>''' if risk_result.get("ais_gap_minutes", 0) >= 20 else ""}
          <div style="margin-top:10px;">
            <div style="font-size:0.7rem;color:#4a6a9a;margin-bottom:4px;">
              RISK SCORE INDICATOR
            </div>
            <div class="risk-bar-wrap">
              <div class="risk-bar-fill"
                   style="width:{row['risk_score']}%;background:{risk_colour};">
              </div>
            </div>
          </div>
        </div>
        """, unsafe_allow_html=True)

        # ── Dynamic risk breakdown from risk_engine ──────────────────────
        factors = risk_result.get("factors", [])
        if factors:
            st.markdown('<div class="section-header">📊 Risk Breakdown</div>',
                        unsafe_allow_html=True)
            total = sum(f["contribution"] for f in factors)
            for factor in factors:
                pct = min(int(factor["contribution"] / factor["max_weight"] * 100), 100)
                st.markdown(f"""
                <div style="margin-bottom:8px;">
                  <div style="display:flex;justify-content:space-between;
                              font-size:0.78rem;margin-bottom:3px;">
                    <span style="color:#a0b8d8;">{factor['factor']}</span>
                    <span style="color:#ff6666;font-weight:700;">
                      +{factor['contribution']}
                    </span>
                  </div>
                  <div class="risk-bar-wrap">
                    <div class="risk-bar-fill"
                         style="width:{pct}%;background:#ff4444;opacity:0.7;">
                    </div>
                  </div>
                </div>
                """, unsafe_allow_html=True)
            st.markdown(f"""
            <div style="background:#1a0505;border:1px solid #ff3333;border-radius:6px;
                        padding:8px 12px;margin-top:6px;text-align:center;">
              <span style="color:#aaa;font-size:0.75rem;">TOTAL RISK SCORE &nbsp;</span>
              <span style="color:#ff3333;font-size:1.3rem;font-weight:700;">
                {row['risk_score']}/100
              </span>
            </div>
            """, unsafe_allow_html=True)

        # ── Feature values for the selected vessel ───────────────────────
        if selected_vessel in feature_df.index:
            feat_row = feature_df.loc[selected_vessel]
            st.markdown('<div class="section-header" style="margin-top:14px;">🧮 Feature Values</div>',
                        unsafe_allow_html=True)
            feat_labels = {
                "speed":               "Speed (kn)",
                "speed_deviation":     "Speed Deviation",
                "loitering_score":     "Loitering Score",
                "displacement_ratio":  "Displacement Ratio",
                "erratic_score":       "Erratic Score",
                "zone_proximity_norm": "Zone Proximity",
                "bearing_change_rate": "Bearing Change Rate",
            }
            for key, label in feat_labels.items():
                val = feat_row[key]
                st.markdown(f"""
                <div class="info-row">
                  <span class="info-label">{label}</span>
                  <span class="info-value">{val:.4f}</span>
                </div>""", unsafe_allow_html=True)

        # ── Behaviour Analysis for the selected vessel (M4) ──────────────
        _beh = pipeline_meta.get("behavior_map", {}).get(selected_vessel)
        if _beh:
            st.markdown('<div class="section-header" style="margin-top:14px;">🧭 Behaviour Analysis</div>',
                        unsafe_allow_html=True)
            _beh_rows = [
                ("Trajectory Duration", f"{_beh['trajectory_duration_minutes']:.0f} min"),
                ("Position Count",      f"{_beh['position_count']}"),
                ("Distance Travelled",  f"{_beh['distance_travelled_km']:.2f} km"),
                ("Avg Speed",           f"{_beh['avg_speed']:.1f} kn"),
                ("Max Speed",           f"{_beh['max_speed']:.1f} kn"),
                ("Speed Variability",   f"{_beh['speed_std']:.2f}"),
                ("Stationary Ratio",    f"{_beh['stationary_ratio']*100:.0f}%"),
                ("Loitering Score",     f"{_beh['loitering_score']:.3f}"),
                ("Mean Heading Change", f"{_beh['mean_heading_change']:.1f}°"),
                ("Max AIS Gap",         f"{_beh['max_ais_gap_minutes']:.0f} min"),
                ("Significant Gaps",    f"{_beh['significant_gap_count']}"),
            ]
            for label, val in _beh_rows:
                st.markdown(f"""
                <div class="info-row">
                  <span class="info-label">{label}</span>
                  <span class="info-value">{val}</span>
                </div>""", unsafe_allow_html=True)
            # Behaviour classification chips + plain-language summary
            _chips = "".join(
                f'<span style="display:inline-block;background:#13284a;'
                f'border:1px solid #1e3a6e;color:#7fd4ff;border-radius:4px;'
                f'padding:1px 6px;margin:2px 3px 0 0;font-size:0.66rem;">{lbl}</span>'
                for lbl in _beh.get("behavior_labels", [])
            )
            st.markdown(f"""
            <div style="margin-top:6px;">{_chips}</div>
            <div style="font-size:0.72rem;color:#a0b8d8;margin-top:6px;line-height:1.5;">
              {_beh.get('behavior_summary', '')}
            </div>
            """, unsafe_allow_html=True)

        # ── Risk Factors (explainable, M4) ───────────────────────────────
        _rfactors = risk_result.get("risk_factors", [])
        if _rfactors:
            st.markdown('<div class="section-header" style="margin-top:14px;">🧾 Risk Factors</div>',
                        unsafe_allow_html=True)
            _rf_html = "".join(
                f'<div style="font-size:0.75rem;color:#c8d6f0;padding:3px 0;'
                f'border-bottom:1px solid #1a2840;">▸ {rf}</div>'
                for rf in _rfactors
            )
            st.markdown(f'<div class="info-card" style="padding:10px 12px;">{_rf_html}</div>',
                        unsafe_allow_html=True)

        # ── Recent history for this vessel (from SQLite, M3) ─────────────
        try:
            _vh = database.get_vessel_recent(selected_vessel, limit=5,
                                             db_path=DEFAULT_DB_PATH)
        except Exception:
            _vh = {"risk": [], "alerts": []}
        if _vh.get("risk"):
            st.markdown('<div class="section-header" style="margin-top:14px;">🕑 Recent History</div>',
                        unsafe_allow_html=True)
            for r in _vh["risk"]:
                rl = r.get("risk_level", "")
                rc = {"HIGH": "#FF3333", "MEDIUM": "#FF8C00", "LOW": "#2ECC71"}.get(rl, "#8899bb")
                ts = str(r.get("computed_at", ""))[:16].replace("T", " ")
                st.markdown(f"""
                <div class="info-row">
                  <span class="info-label" style="font-size:0.7rem;">{ts}</span>
                  <span class="info-value" style="color:{rc};">
                    {int(r.get('risk_score', 0))}/100 · {rl}
                  </span>
                </div>""", unsafe_allow_html=True)

    st.markdown('<hr style="border-color:#1e2d50;margin:20px 0 12px;">', unsafe_allow_html=True)
    st.markdown(f"""
    <div style="font-size:0.68rem;color:#2a4060;text-align:center;line-height:1.6;">
      Live Detection Pipeline · Isolation Forest<br>
      {pipeline_meta.get('region', 'Bay of Bengal')} · Maritime Surveillance
    </div>
    """, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Main content
# ---------------------------------------------------------------------------

# ── Page header ─────────────────────────────────────────────────────────────
st.markdown(f"""
<div style="display:flex;align-items:center;gap:16px;
            padding:18px 24px;margin-bottom:20px;
            background:linear-gradient(90deg,#0a1830 0%,#080c1a 100%);
            border-bottom:1px solid #1e2d50;border-radius:0 0 8px 8px;">
  <div style="font-size:2.5rem;">🛥️</div>
  <div>
    <div style="font-size:1.5rem;font-weight:700;color:#00d4ff;letter-spacing:0.06em;">
      Illegal Fishing Detection System
    </div>
    <div style="font-size:0.78rem;color:#4a6a9a;letter-spacing:0.1em;margin-top:2px;">
      MARITIME VESSEL MONITORING · {pipeline_meta.get('region', 'Bay of Bengal').upper()} · ANOMALY DETECTION PROTOTYPE
    </div>
  </div>
  <div style="margin-left:auto;text-align:right;">
    <div style="font-size:0.68rem;color:#2a5080;">SYSTEM STATUS</div>
    <div style="color:#2ECC71;font-weight:700;font-size:0.9rem;">● OPERATIONAL</div>
    <div style="font-size:0.65rem;color:#2a5080;margin-top:2px;">
      ISOLATION FOREST · MODEL {pipeline_meta.get('model_status', 'ready').upper()}
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

# ── Metrics row ──────────────────────────────────────────────────────────────
c1, c2, c3, c4, c5 = st.columns(5)
with c1:
    st.metric("🚢 Total Vessels",  stats["total_vessels"])
with c2:
    st.metric("🔴 High Risk",      stats["high_risk"],
              delta=f"+{stats['high_risk']} flagged", delta_color="inverse")
with c3:
    st.metric("🟠 Medium Risk",    stats["medium_risk"])
with c4:
    st.metric("🟢 Low Risk",       stats["low_risk"])
with c5:
    st.metric("⚠️ Active Alerts",  stats["active_alerts"],
              delta="Requires attention", delta_color="inverse")

st.markdown("<div style='margin-bottom:16px;'></div>", unsafe_allow_html=True)

# ── Map + right panel ────────────────────────────────────────────────────────
map_col, panel_col = st.columns([3, 1], gap="medium")
sel = selected_vessel if selected_vessel != "— Select a vessel —" else ""

with map_col:
    st.markdown('<div class="section-header">🗺️ Live Vessel Tracking Map</div>',
                unsafe_allow_html=True)
    # In Historical AIS mode, overlay the selected vessel's REAL AIS track (M4).
    _sel_traj = None
    if sel and _active_source == "Historical AIS":
        _sel_traj = pipeline_meta.get("map_trajectories", {}).get(sel)
    folium_map = build_map(
        df,
        selected_vessel=sel,
        zones=pipeline_meta["zones"],
        center=pipeline_meta["map_center"],
        zoom=pipeline_meta["map_zoom"],
        selected_trajectory=_sel_traj,
    )
    st_folium(folium_map, width=None, height=560, returned_objects=[])

with panel_col:
    # ── Dynamic alerts ───────────────────────────────────────────────────
    st.markdown('<div class="section-header">🚨 Active Alerts</div>',
                unsafe_allow_html=True)

    displayed = 0
    for alert in alerts:
        if displayed >= 8:   # cap at 8 to avoid overflow
            break
        css_class = "alert-high" if alert["level"] == "HIGH" else "alert-medium"
        st.markdown(f"""
        <div class="{css_class}">
          <div style="display:flex;justify-content:space-between;margin-bottom:4px;">
            <span style="font-weight:700;font-size:0.8rem;">
              {alert['icon']} {alert['vessel']}
            </span>
            <span style="font-size:0.68rem;color:#4a6a9a;">{alert['level']}</span>
          </div>
          <div style="color:#b0c0d8;font-size:0.76rem;line-height:1.4;">
            {alert['message']}
          </div>
        </div>
        """, unsafe_allow_html=True)
        displayed += 1

    # ── Fleet status table ───────────────────────────────────────────────
    st.markdown('<div class="section-header" style="margin-top:16px;">📋 Fleet Status</div>',
                unsafe_allow_html=True)

    display_df = (
        df[["vessel_id", "risk_level", "risk_score", "speed"]]
        .sort_values("risk_score", ascending=False)
        .copy()
    )
    display_df.columns = ["Vessel", "Risk", "Score", "Spd(kn)"]
    st.dataframe(display_df, hide_index=True, height=280, width="stretch")

st.markdown("<div style='margin-bottom:16px;'></div>", unsafe_allow_html=True)

# ── Pipeline section ─────────────────────────────────────────────────────────
st.markdown('<div class="section-header">⚙️ Detection Pipeline</div>',
            unsafe_allow_html=True)

pipeline_steps = [
    ("📡", "Vessel Data"),
    ("🗺️", "Geofencing"),
    ("🧮", "Feature\nExtraction"),
    ("🤖", "Isolation\nForest"),
    ("📊", "Risk\nEngine"),
    ("🚨", "Dynamic\nAlerts"),
]

cols = st.columns(len(pipeline_steps) * 2 - 1)
for i, (icon, label) in enumerate(pipeline_steps):
    with cols[i * 2]:
        st.markdown(f"""
        <div style="background:linear-gradient(135deg,#0d1b35,#0a1525);
                    border:1px solid #1e3a6e;border-radius:8px;
                    padding:12px 8px;text-align:center;">
          <div style="font-size:1.4rem;">{icon}</div>
          <div style="font-size:0.72rem;font-weight:700;color:#00d4ff;
                      letter-spacing:0.06em;margin-top:4px;line-height:1.4;">
            {label}
          </div>
        </div>
        """, unsafe_allow_html=True)
    if i < len(pipeline_steps) - 1:
        with cols[i * 2 + 1]:
            st.markdown("""
            <div style="text-align:center;padding-top:22px;
                        color:#1e5080;font-size:1.6rem;">▶</div>
            """, unsafe_allow_html=True)

st.markdown("<div style='margin-bottom:16px;'></div>", unsafe_allow_html=True)

# ── Bottom: HIGH risk analysis + anomaly scores ──────────────────────────────
bottom_left, bottom_right = st.columns(2, gap="medium")

with bottom_left:
    st.markdown('<div class="section-header">🔬 High Risk Vessel Analysis</div>',
                unsafe_allow_html=True)

    high_df = df[df["risk_level"] == "HIGH"].sort_values("risk_score", ascending=False)
    if high_df.empty:
        st.markdown('<p style="color:#4a6a9a;font-size:0.82rem;">No HIGH risk vessels detected.</p>',
                    unsafe_allow_html=True)
    else:
        for _, hrow in high_df.iterrows():
            rc = "#FF3333"
            factors_html = ""
            for f in risk_map[hrow["vessel_id"]].get("factors", [])[:3]:
                factors_html += (f'<div style="font-size:0.7rem;color:#ff9999;'
                                 f'margin-top:3px;">▸ {f["factor"]} '
                                 f'(+{f["contribution"]})</div>')
            st.markdown(f"""
            <div class="info-card" style="border-left:3px solid {rc};">
              <div style="display:flex;justify-content:space-between;
                          align-items:center;margin-bottom:6px;">
                <span style="color:#00d4ff;font-weight:700;">⚓ {hrow['vessel_id']}</span>
                <span style="color:{rc};font-weight:700;font-size:1rem;">
                  {hrow['risk_score']}/100
                </span>
              </div>
              <div style="font-size:0.75rem;color:#a0b8d8;margin-bottom:4px;">
                {hrow['behavior']}
              </div>
              {factors_html}
              <div class="risk-bar-wrap" style="margin-top:6px;">
                <div class="risk-bar-fill"
                     style="width:{hrow['risk_score']}%;background:{rc};">
                </div>
              </div>
              <div style="font-size:0.7rem;color:#ffd700;margin-top:6px;">
                📍 {hrow['zone_status']}
              </div>
            </div>
            """, unsafe_allow_html=True)

with bottom_right:
    st.markdown('<div class="section-header">📈 Risk Score Distribution</div>',
                unsafe_allow_html=True)

    chart_df = (
        df[["vessel_id", "risk_score"]]
        .sort_values("risk_score", ascending=False)
        .set_index("vessel_id")
    )
    st.bar_chart(chart_df["risk_score"], height=210, width="stretch")

    # ── ML Anomaly scores mini-table ─────────────────────────────────────
    st.markdown('<div class="section-header" style="margin-top:12px;">🤖 Isolation Forest Scores</div>',
                unsafe_allow_html=True)

    ml_rows = [
        {
            "Vessel": vid,
            "IF Score": f"{v['anomaly_score']:.3f}",
            "Anomalous": "🔺 Yes" if v["is_anomalous"] else "✓ No",
        }
        for vid, v in sorted(anomaly_map.items(),
                              key=lambda x: x[1]["anomaly_score"], reverse=True)
    ]
    ml_df = pd.DataFrame(ml_rows)
    st.dataframe(ml_df, hide_index=True, height=200, width="stretch")

st.markdown("<div style='margin-bottom:16px;'></div>", unsafe_allow_html=True)

# ── History section (M3 — SQLite-backed) ─────────────────────────────────────
st.markdown('<div class="section-header">🗄️ Stored History (SQLite)</div>',
            unsafe_allow_html=True)

if _persist_warning:
    st.warning(_persist_warning)

try:
    _hist = database.get_history_summary(DEFAULT_DB_PATH)
    _recent_alerts = database.get_recent_alerts(limit=8, db_path=DEFAULT_DB_PATH)
    _recent_risk = database.get_recent_risk(limit=8, db_path=DEFAULT_DB_PATH)
    _history_ok = True
except Exception as exc:
    _history_ok = False
    st.info(f"History is temporarily unavailable: {exc}")

if _history_ok:
    hc1, hc2, hc3, hc4 = st.columns(4)
    with hc1:
        st.metric("🚢 Vessels Stored", f"{_hist['total_vessels']:,}")
    with hc2:
        st.metric("📍 Positions Stored", f"{_hist['total_positions']:,}")
    with hc3:
        st.metric("🚨 Alerts Stored", f"{_hist['total_alerts']:,}")
    with hc4:
        st.metric("🔴 HIGH Alerts", f"{_hist['high_alerts']:,}")

    hleft, hright = st.columns(2, gap="medium")

    with hleft:
        st.markdown('<div class="section-header" style="margin-top:8px;">Recent Alerts</div>',
                    unsafe_allow_html=True)
        if _recent_alerts:
            _ra_df = pd.DataFrame([
                {
                    "Vessel": a["vessel_id"],
                    "Level":  a["level"],
                    "Message": (a["message"] or "")[:60],
                    "Time":   str(a["created_at"])[:16].replace("T", " "),
                }
                for a in _recent_alerts
            ])
            st.dataframe(_ra_df, hide_index=True, height=240, width="stretch")
        else:
            st.markdown('<p style="color:#4a6a9a;font-size:0.82rem;">No alerts stored yet.</p>',
                        unsafe_allow_html=True)

    with hright:
        st.markdown('<div class="section-header" style="margin-top:8px;">Recent Risk Assessments</div>',
                    unsafe_allow_html=True)
        if _recent_risk:
            _rr_df = pd.DataFrame([
                {
                    "Vessel": r["vessel_id"],
                    "Score":  int(r["risk_score"]) if r["risk_score"] is not None else 0,
                    "Level":  r["risk_level"],
                    "Time":   str(r["computed_at"])[:16].replace("T", " "),
                }
                for r in _recent_risk
            ])
            st.dataframe(_rr_df, hide_index=True, height=240, width="stretch")
        else:
            st.markdown('<p style="color:#4a6a9a;font-size:0.82rem;">No risk records stored yet.</p>',
                        unsafe_allow_html=True)

st.markdown("<div style='margin-bottom:24px;'></div>", unsafe_allow_html=True)

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown(f"""
<div style="text-align:center;padding:16px;border-top:1px solid #1e2d50;
            color:#2a4060;font-size:0.7rem;letter-spacing:0.06em;">
  ILLEGAL FISHING DETECTION SYSTEM · ISOLATION FOREST ANOMALY DETECTION
  &nbsp;|&nbsp; {pipeline_meta.get('region', 'Bay of Bengal')} Maritime Surveillance
</div>
""", unsafe_allow_html=True)
