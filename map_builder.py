"""
map_builder.py
Builds the Folium interactive map for the
Illegal Fishing Detection System prototype (Round 1).
"""

import folium
from folium import plugins
import pandas as pd
from typing import List, Dict, Tuple

from data_generator import generate_trajectory, get_restricted_zones

# ---------------------------------------------------------------------------
# Colour / style helpers
# ---------------------------------------------------------------------------

_RISK_COLOURS = {
    "HIGH":   "#FF3333",
    "MEDIUM": "#FF8C00",
    "LOW":    "#2ECC71",
}

_TRAIL_COLOURS = {
    "HIGH":   "#FF6666",
    "MEDIUM": "#FFB347",
    "LOW":    "#7DCEA0",
}

_ICON_COLOURS = {
    "HIGH":   "red",
    "MEDIUM": "orange",
    "LOW":    "green",
}


def _risk_badge(risk_level: str) -> str:
    colour = _RISK_COLOURS.get(risk_level, "#888888")
    return (
        f'<span style="background:{colour};color:#fff;padding:2px 6px;'
        f'border-radius:4px;font-weight:bold;font-size:11px;">'
        f'{risk_level}</span>'
    )


# ---------------------------------------------------------------------------
# Map initialisation
# ---------------------------------------------------------------------------

def _create_base_map(center: Tuple[float, float] = (12.5, 80.2),
                     zoom: int = 8) -> folium.Map:
    """Return a dark-themed Folium map centred on the Bay of Bengal."""
    m = folium.Map(
        location=center,
        zoom_start=zoom,
        tiles="CartoDB dark_matter",
        control_scale=True,
    )
    return m


# ---------------------------------------------------------------------------
# Restricted zones
# ---------------------------------------------------------------------------

def _add_restricted_zones(fmap: folium.Map) -> folium.Map:
    """Draw restricted fishing zone polygons onto the map."""
    zones = get_restricted_zones()
    zone_group = folium.FeatureGroup(name="Restricted Zones", show=True)

    for zone in zones:
        # Polygon fill
        folium.Polygon(
            locations=zone["coords"],
            color=zone["color"],
            weight=2.5,
            fill=True,
            fill_color=zone["color"],
            fill_opacity=0.18,
            tooltip=zone["name"],
            popup=folium.Popup(
                f'<b style="color:{zone["color"]}">{zone["name"]}</b><br>'
                f'<span style="font-size:12px;">⚠️ Fishing Prohibited</span>',
                max_width=200,
            ),
        ).add_to(zone_group)

        # Zone label marker (DivIcon)
        folium.Marker(
            location=zone["center"],
            icon=folium.DivIcon(
                html=(
                    f'<div style="'
                    f'background:rgba(255,50,50,0.85);'
                    f'color:white;'
                    f'padding:4px 8px;'
                    f'border-radius:4px;'
                    f'font-size:11px;'
                    f'font-weight:bold;'
                    f'white-space:nowrap;'
                    f'border:1px solid {zone["color"]};'
                    f'">'
                    f'⛔ {zone["name"]}'
                    f'</div>'
                ),
                icon_size=(160, 28),
                icon_anchor=(80, 14),
            ),
        ).add_to(zone_group)

    zone_group.add_to(fmap)
    return fmap


# ---------------------------------------------------------------------------
# Vessel trails
# ---------------------------------------------------------------------------

def _add_vessel_trails(fmap: folium.Map,
                       df: pd.DataFrame) -> folium.Map:
    """Draw dotted historical trail lines for every vessel."""
    trail_group = folium.FeatureGroup(name="Vessel Trails", show=True)

    for _, row in df.iterrows():
        trail = generate_trajectory(
            lat=row["latitude"],
            lon=row["longitude"],
            behavior=row["behavior"],
        )
        colour = _TRAIL_COLOURS.get(row["risk_level"], "#AAAAAA")

        folium.PolyLine(
            locations=trail,
            color=colour,
            weight=1.8,
            opacity=0.55,
            dash_array="6 4",
            tooltip=f'{row["vessel_id"]} trail',
        ).add_to(trail_group)

    trail_group.add_to(fmap)
    return fmap


# ---------------------------------------------------------------------------
# Vessel markers
# ---------------------------------------------------------------------------

def _vessel_popup_html(row: pd.Series) -> str:
    risk_colour = _RISK_COLOURS.get(row["risk_level"], "#888")
    return f"""
    <div style="font-family:monospace;min-width:210px;background:#1a1a2e;
                color:#e0e0e0;padding:10px;border-radius:6px;
                border-left:4px solid {risk_colour};">
      <b style="font-size:14px;color:#00d4ff;">⚓ {row['vessel_id']}</b><br><br>
      <table style="width:100%;font-size:12px;border-collapse:collapse;">
        <tr><td style="color:#aaa;">Speed</td>
            <td style="color:#fff;text-align:right;"><b>{row['speed']} kn</b></td></tr>
        <tr><td style="color:#aaa;">Heading</td>
            <td style="color:#fff;text-align:right;"><b>{row['heading']}°</b></td></tr>
        <tr><td style="color:#aaa;">Risk Score</td>
            <td style="color:{risk_colour};text-align:right;">
                <b>{row['risk_score']}/100</b></td></tr>
        <tr><td style="color:#aaa;">Risk Level</td>
            <td style="text-align:right;">{_risk_badge(row['risk_level'])}</td></tr>
        <tr><td style="color:#aaa;">Behavior</td>
            <td style="color:#fff;text-align:right;font-size:11px;">
                <b>{row['behavior']}</b></td></tr>
        <tr><td style="color:#aaa;">Zone</td>
            <td style="color:#ffd700;text-align:right;font-size:11px;">
                <b>{row['zone_status']}</b></td></tr>
      </table>
    </div>
    """


def _add_vessel_markers(fmap: folium.Map,
                        df: pd.DataFrame,
                        selected_vessel: str = "") -> folium.Map:
    """Add a CircleMarker for every vessel, with a pulsing ring for HIGH risk."""
    vessel_group = folium.FeatureGroup(name="Vessels", show=True)

    for _, row in df.iterrows():
        lat  = row["latitude"]
        lon  = row["longitude"]
        vid  = row["vessel_id"]
        risk = row["risk_level"]
        colour = _RISK_COLOURS.get(risk, "#888888")
        is_selected = (vid == selected_vessel)

        # Outer glow ring for HIGH risk
        if risk == "HIGH":
            folium.CircleMarker(
                location=[lat, lon],
                radius=18,
                color=colour,
                fill=True,
                fill_color=colour,
                fill_opacity=0.12,
                weight=1,
            ).add_to(vessel_group)

        # Main marker
        folium.CircleMarker(
            location=[lat, lon],
            radius=10 if is_selected else 8,
            color="#ffffff" if is_selected else colour,
            fill=True,
            fill_color=colour,
            fill_opacity=0.85,
            weight=3 if is_selected else 1.5,
            tooltip=folium.Tooltip(
                f"<b>{vid}</b> | {risk} RISK | {row['speed']} kn",
                style="background:#1a1a2e;color:#00d4ff;border:1px solid #00d4ff;"
                      "padding:4px 8px;border-radius:4px;font-family:monospace;",
            ),
            popup=folium.Popup(_vessel_popup_html(row), max_width=240),
        ).add_to(vessel_group)

        # Vessel ID label
        folium.Marker(
            location=[lat + 0.025, lon],
            icon=folium.DivIcon(
                html=(
                    f'<div style="'
                    f'color:{colour};'
                    f'font-size:10px;'
                    f'font-weight:bold;'
                    f'font-family:monospace;'
                    f'text-shadow:0 0 4px #000,0 0 4px #000;'
                    f'white-space:nowrap;">'
                    f'{vid}'
                    f'</div>'
                ),
                icon_size=(50, 14),
                icon_anchor=(25, 7),
            ),
        ).add_to(vessel_group)

    vessel_group.add_to(fmap)
    return fmap


# ---------------------------------------------------------------------------
# Map legend
# ---------------------------------------------------------------------------

def _add_legend(fmap: folium.Map) -> folium.Map:
    legend_html = """
    <div style="position:fixed;bottom:30px;left:30px;z-index:1000;
                background:rgba(10,10,30,0.92);color:#e0e0e0;
                padding:12px 16px;border-radius:8px;
                border:1px solid #00d4ff;font-family:monospace;font-size:12px;">
      <b style="color:#00d4ff;">⚓ VESSEL RISK LEGEND</b><br><br>
      <span style="color:#FF3333;">●</span>&nbsp; HIGH RISK &nbsp;&nbsp;&nbsp;
      <span style="color:#FF8C00;">●</span>&nbsp; MEDIUM RISK &nbsp;&nbsp;&nbsp;
      <span style="color:#2ECC71;">●</span>&nbsp; LOW RISK<br><br>
      <span style="color:#FF4444;">▬</span>&nbsp; Restricted Zone &nbsp;&nbsp;
      <span style="color:#aaa;">- -</span>&nbsp; Vessel Trail
    </div>
    """
    fmap.get_root().html.add_child(folium.Element(legend_html))
    return fmap


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_map(df: pd.DataFrame, selected_vessel: str = "") -> folium.Map:
    """
    Assemble and return the complete Folium map with:
    - Dark CartoDB base tiles
    - Restricted zone polygons + labels
    - Vessel movement trails
    - Vessel markers with popups
    - Layer control
    - Legend
    """
    fmap = _create_base_map()
    fmap = _add_restricted_zones(fmap)
    fmap = _add_vessel_trails(fmap, df)
    fmap = _add_vessel_markers(fmap, df, selected_vessel)
    fmap = _add_legend(fmap)
    folium.LayerControl(collapsed=False).add_to(fmap)
    return fmap
