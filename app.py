"""
app.py
Main Streamlit dashboard for the Illegal Fishing Detection System — Round 2.

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
from anomaly_detector import run_anomaly_detection
from risk_engine import run_risk_engine, generate_alerts
from map_builder import build_map

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
def run_pipeline() -> tuple:
    """
    Execute the complete Round 2 detection pipeline.

    Returns
    -------
    df          : enriched vessel DataFrame (all columns the UI needs)
    alerts      : list of dynamic alert dicts
    anomaly_map : {vessel_id: {anomaly_score, is_anomalous}}
    risk_map    : {vessel_id: risk_result_dict}
    feature_df  : feature DataFrame (used by ML details panel)
    """
    zones = get_restricted_zones()

    # 1. Raw vessel data
    base_df = generate_vessel_dataframe()

    # 2. Trajectories
    trajectories = build_all_trajectories(base_df)

    # 3. Geofencing
    geo_list = run_geofencing(base_df, zones)
    geo_map  = {g["vessel_id"]: g for g in geo_list}

    # 4. Feature extraction
    feature_df = build_feature_matrix(base_df, trajectories, geo_map)

    # 5. Isolation Forest anomaly detection
    anomaly_map = run_anomaly_detection(feature_df)

    # 6. Risk engine
    vessel_ids = base_df["vessel_id"].tolist()
    risk_map   = run_risk_engine(vessel_ids, geo_map, feature_df, anomaly_map)

    # 7. Enrich base DataFrame with pipeline outputs
    df = base_df.copy()
    df["risk_score"]  = df["vessel_id"].map(lambda v: risk_map[v]["risk_score"])
    df["risk_level"]  = df["vessel_id"].map(lambda v: risk_map[v]["risk_level"])
    df["behavior"]    = df["vessel_id"].map(lambda v: risk_map[v]["behavior"])
    df["zone_status"] = df["vessel_id"].map(lambda v: risk_map[v]["zone_status"])

    # 8. Dynamic alerts
    alerts = generate_alerts(risk_map)

    return df, alerts, anomaly_map, risk_map, feature_df


df, alerts, anomaly_map, risk_map, feature_df = run_pipeline()


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

    st.markdown('<hr style="border-color:#1e2d50;margin:20px 0 12px;">', unsafe_allow_html=True)
    st.markdown("""
    <div style="font-size:0.68rem;color:#2a4060;text-align:center;line-height:1.6;">
      Round 2 · Live Pipeline · Isolation Forest<br>
      Bay of Bengal Region<br>
      <span style="color:#1e3a6e;">© 2025 IFDS Project</span>
    </div>
    """, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Main content
# ---------------------------------------------------------------------------

# ── Page header ─────────────────────────────────────────────────────────────
st.markdown("""
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
      MARITIME VESSEL MONITORING · BAY OF BENGAL · ROUND 2 PROTOTYPE
    </div>
  </div>
  <div style="margin-left:auto;text-align:right;">
    <div style="font-size:0.68rem;color:#2a5080;">SYSTEM STATUS</div>
    <div style="color:#2ECC71;font-weight:700;font-size:0.9rem;">● OPERATIONAL</div>
    <div style="font-size:0.65rem;color:#2a5080;margin-top:2px;">
      ISOLATION FOREST · LIVE PIPELINE
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
    folium_map = build_map(df, selected_vessel=sel)
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
st.markdown('<div class="section-header">⚙️ Round 2 Detection Pipeline</div>',
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

st.markdown("<div style='margin-bottom:24px;'></div>", unsafe_allow_html=True)

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("""
<div style="text-align:center;padding:16px;border-top:1px solid #1e2d50;
            color:#2a4060;font-size:0.7rem;letter-spacing:0.06em;">
  ILLEGAL FISHING DETECTION SYSTEM · ROUND 2 · ISOLATION FOREST ANOMALY DETECTION
  &nbsp;|&nbsp; Bay of Bengal Maritime Surveillance · © 2025
</div>
""", unsafe_allow_html=True)
