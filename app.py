"""
app.py
Main Streamlit dashboard for the
Illegal Fishing Detection System — Round 1 Prototype.
"""

import streamlit as st
from streamlit_folium import st_folium
import pandas as pd

from data_generator import (
    generate_vessel_dataframe,
    get_alerts,
    get_risk_breakdown,
    compute_dashboard_stats,
)
from map_builder import build_map

# ---------------------------------------------------------------------------
# Page config — must be first Streamlit call
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Illegal Fishing Detection System",
    page_icon="🛥️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Global CSS — dark maritime theme
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
.risk-bar-fill {
    height: 8px;
    border-radius: 4px;
}

/* ── Pipeline ─────────────────────────────────────────────── */
.pipeline-step {
    display: inline-block;
    background: #0d1b35;
    border: 1px solid #1e3a6e;
    border-radius: 6px;
    padding: 8px 14px;
    font-size: 0.78rem;
    font-weight: 600;
    color: #00d4ff;
    letter-spacing: 0.05em;
    text-align: center;
}
.pipeline-arrow {
    display: inline-block;
    color: #1e3a6e;
    font-size: 1.2rem;
    vertical-align: middle;
    margin: 0 4px;
}

/* ── Selectbox / button ───────────────────────────────────── */
[data-testid="stSelectbox"] > div {
    background: #0d1b35 !important;
    border: 1px solid #1e3a6e !important;
    border-radius: 6px !important;
    color: #d0d8e8 !important;
}
.stButton > button {
    background: linear-gradient(90deg, #0066cc, #0044aa);
    color: white;
    border: none;
    border-radius: 6px;
    font-weight: 600;
    letter-spacing: 0.05em;
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
# Data (cached so it doesn't re-generate on every interaction)
# ---------------------------------------------------------------------------
@st.cache_data
def load_data() -> pd.DataFrame:
    return generate_vessel_dataframe()

df = load_data()
alerts = get_alerts()
stats  = compute_dashboard_stats(df)

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
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

    vessel_ids = ["— Select a vessel —"] + sorted(df["vessel_id"].tolist())
    selected_vessel = st.selectbox("Vessel ID", vessel_ids, label_visibility="collapsed")

    if selected_vessel != "— Select a vessel —":
        row = df[df["vessel_id"] == selected_vessel].iloc[0]
        risk_colour = {"HIGH": "#FF3333", "MEDIUM": "#FF8C00", "LOW": "#2ECC71"}.get(
            row["risk_level"], "#888")

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
          <div class="info-row" style="border-bottom:none;">
            <span class="info-label">Behavior</span>
            <span class="info-value" style="font-size:0.75rem;">{row['behavior']}</span>
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

        # Risk breakdown for vessels that have one
        breakdown = get_risk_breakdown(selected_vessel)
        if breakdown:
            st.markdown('<div class="section-header">📊 Risk Breakdown</div>',
                        unsafe_allow_html=True)
            total = sum(f["score"] for f in breakdown)
            for factor in breakdown:
                pct = int(factor["score"] / 100 * 100)
                st.markdown(f"""
                <div style="margin-bottom:8px;">
                  <div style="display:flex;justify-content:space-between;
                              font-size:0.78rem;margin-bottom:3px;">
                    <span style="color:#a0b8d8;">{factor['factor']}</span>
                    <span style="color:#ff6666;font-weight:700;">+{factor['score']}</span>
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
                {total}/100
              </span>
            </div>
            """, unsafe_allow_html=True)

    st.markdown('<hr style="border-color:#1e2d50;margin:20px 0 12px;">', unsafe_allow_html=True)
    st.markdown("""
    <div style="font-size:0.68rem;color:#2a4060;text-align:center;line-height:1.6;">
      Round 1 Prototype · Demo Data<br>
      Bay of Bengal Region<br>
      <span style="color:#1e3a6e;">© 2025 IFDS Project</span>
    </div>
    """, unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Main content
# ---------------------------------------------------------------------------

# ── Page header ────────────────────────────────────────────────────────────
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
      MARITIME VESSEL MONITORING · BAY OF BENGAL · REAL-TIME PROTOTYPE
    </div>
  </div>
  <div style="margin-left:auto;text-align:right;">
    <div style="font-size:0.68rem;color:#2a5080;">SYSTEM STATUS</div>
    <div style="color:#2ECC71;font-weight:700;font-size:0.9rem;">● OPERATIONAL</div>
    <div style="font-size:0.65rem;color:#2a5080;margin-top:2px;">
      ROUND 1 · DEMO MODE
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

# ── Metrics row ─────────────────────────────────────────────────────────────
c1, c2, c3, c4, c5 = st.columns(5)
with c1:
    st.metric("🚢 Total Vessels",   stats["total_vessels"])
with c2:
    st.metric("🔴 High Risk",       stats["high_risk"],
              delta=f"+{stats['high_risk']} flagged", delta_color="inverse")
with c3:
    st.metric("🟠 Medium Risk",     stats["medium_risk"])
with c4:
    st.metric("🟢 Low Risk",        stats["low_risk"])
with c5:
    st.metric("⚠️ Active Alerts",   stats["active_alerts"],
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
    # ── Alerts ──────────────────────────────────────────────────────────────
    st.markdown('<div class="section-header">🚨 Active Alerts</div>',
                unsafe_allow_html=True)

    for alert in alerts:
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

    # ── Fleet table ─────────────────────────────────────────────────────────
    st.markdown('<div class="section-header" style="margin-top:16px;">📋 Fleet Status</div>',
                unsafe_allow_html=True)

    display_df = df[["vessel_id", "risk_level", "risk_score", "speed"]].copy()
    display_df.columns = ["Vessel", "Risk", "Score", "Spd(kn)"]

    def _color_risk(val):
        colours = {"HIGH": "color: #FF3333", "MEDIUM": "color: #FF8C00",
                   "LOW": "color: #2ECC71"}
        return colours.get(val, "")

    st.dataframe(display_df, hide_index=True, height=280,
                 width="stretch")

st.markdown("<div style='margin-bottom:16px;'></div>", unsafe_allow_html=True)

# ── Pipeline section ─────────────────────────────────────────────────────────
st.markdown('<div class="section-header">⚙️ System Processing Pipeline</div>',
            unsafe_allow_html=True)

pipeline_steps = [
    ("📡", "Vessel Data\nIngestion"),
    ("🗺️", "Geofencing\nAnalysis"),
    ("🔍", "Behavior\nAnalysis"),
    ("📊", "Risk\nAssessment"),
    ("🚨", "Alert\nGeneration"),
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

# ── Bottom two-column section: vessel detail + risk score distribution ───────
bottom_left, bottom_right = st.columns(2, gap="medium")

with bottom_left:
    st.markdown('<div class="section-header">🔬 Vessel Risk Analysis</div>',
                unsafe_allow_html=True)

    # Show detailed risk breakdown for HIGH risk vessels
    high_df = df[df["risk_level"] == "HIGH"].sort_values("risk_score", ascending=False)

    for _, hrow in high_df.iterrows():
        risk_colour = "#FF3333"
        st.markdown(f"""
        <div class="info-card" style="border-left:3px solid {risk_colour};">
          <div style="display:flex;justify-content:space-between;align-items:center;
                      margin-bottom:8px;">
            <span style="color:#00d4ff;font-weight:700;">⚓ {hrow['vessel_id']}</span>
            <span style="color:{risk_colour};font-weight:700;font-size:1rem;">
              {hrow['risk_score']}/100
            </span>
          </div>
          <div style="font-size:0.75rem;color:#a0b8d8;margin-bottom:6px;">
            {hrow['behavior']}
          </div>
          <div class="risk-bar-wrap">
            <div class="risk-bar-fill"
                 style="width:{hrow['risk_score']}%;background:{risk_colour};">
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

    # Simple bar chart using streamlit's native bar_chart
    chart_df = (
        df[["vessel_id", "risk_score", "risk_level"]]
        .sort_values("risk_score", ascending=False)
        .set_index("vessel_id")
    )
    st.bar_chart(chart_df["risk_score"], height=260, width="stretch")

    # Summary counts
    scol1, scol2, scol3 = st.columns(3)
    with scol1:
        st.markdown(f"""
        <div style="background:#1a0505;border:1px solid #ff3333;border-radius:6px;
                    padding:8px;text-align:center;">
          <div style="color:#ff3333;font-size:1.4rem;font-weight:700;">
            {stats['high_risk']}
          </div>
          <div style="color:#aaa;font-size:0.65rem;">HIGH RISK</div>
        </div>""", unsafe_allow_html=True)
    with scol2:
        st.markdown(f"""
        <div style="background:#1a0a00;border:1px solid #ff8c00;border-radius:6px;
                    padding:8px;text-align:center;">
          <div style="color:#ff8c00;font-size:1.4rem;font-weight:700;">
            {stats['medium_risk']}
          </div>
          <div style="color:#aaa;font-size:0.65rem;">MEDIUM RISK</div>
        </div>""", unsafe_allow_html=True)
    with scol3:
        st.markdown(f"""
        <div style="background:#001a08;border:1px solid #2ecc71;border-radius:6px;
                    padding:8px;text-align:center;">
          <div style="color:#2ecc71;font-size:1.4rem;font-weight:700;">
            {stats['low_risk']}
          </div>
          <div style="color:#aaa;font-size:0.65rem;">LOW RISK</div>
        </div>""", unsafe_allow_html=True)

st.markdown("<div style='margin-bottom:24px;'></div>", unsafe_allow_html=True)

# ── Footer ───────────────────────────────────────────────────────────────────
st.markdown("""
<div style="text-align:center;padding:16px;border-top:1px solid #1e2d50;
            color:#2a4060;font-size:0.7rem;letter-spacing:0.06em;">
  ILLEGAL FISHING DETECTION SYSTEM · ROUND 1 PROTOTYPE · DEMO / SIMULATED DATA ONLY
  &nbsp;|&nbsp; Bay of Bengal Maritime Surveillance · © 2025
</div>
""", unsafe_allow_html=True)
