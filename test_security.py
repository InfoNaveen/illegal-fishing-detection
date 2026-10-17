"""
test_security.py
Security regression tests for the Illegal Fishing Detection System.

These lock in fixes made during the security audit:

1. Map popups/tooltips HTML-escape vessel-supplied text (live AIS feed is
   untrusted → stored-XSS defence).
2. The live vessel-state stale check fails safe (never drops a vessel on a
   bad/missing timestamp) and handles naive + tz-aware timestamps.

All tests are deterministic, offline, and require no API key.
"""

from datetime import datetime, timezone, timedelta

import pandas as pd

import map_builder
from map_builder import _esc, _vessel_popup_html
from live_ais import VesselStateManager


# ---------------------------------------------------------------------------
# 1. HTML escaping in the Folium map (XSS defence)
# ---------------------------------------------------------------------------

def test_esc_neutralises_html():
    payload = '<script>alert(1)</script>'
    out = _esc(payload)
    assert "<script>" not in out
    assert "&lt;script&gt;" in out
    # quotes are escaped too (attribute-context safety)
    assert _esc('"onmouseover="x').count('"') == 0


def test_vessel_popup_escapes_untrusted_fields():
    row = pd.Series({
        "vessel_id": '<img src=x onerror=alert(1)>',
        "speed": 8.0,
        "heading": 90.0,
        "risk_score": 42,
        "risk_level": "LOW",
        "behavior": '<b>evil</b>',
        "zone_status": '"><script>steal()</script>',
    })
    html_out = _vessel_popup_html(row)
    # No raw injected tags survive as live HTML — the opening "<" of every
    # injected tag must be escaped, so no new element/attribute context exists.
    assert "<img" not in html_out
    assert "<script>" not in html_out
    assert "</script>" not in html_out
    # Escaped forms are present instead (payloads rendered as inert text).
    assert "&lt;img" in html_out
    assert "&lt;script&gt;" in html_out


def test_build_map_with_malicious_vessel_id_does_not_embed_raw_html():
    df = pd.DataFrame([{
        "vessel_id": '<script>x</script>',
        "latitude": 10.0, "longitude": 80.0,
        "speed": 5.0, "heading": 90.0, "behavior": "AIS Track",
        "risk_level": "LOW", "risk_score": 10, "zone_status": "Open Waters",
    }])
    fmap = map_builder.build_map(df, zones=[])
    rendered = fmap.get_root().render()
    # The raw script tag from the vessel id must never reach the HTML output.
    assert "<script>x</script>" not in rendered


# ---------------------------------------------------------------------------
# 2. Stale-vessel age check fails safe
# ---------------------------------------------------------------------------

def _obs(vid, ts):
    return {"vessel_id": vid, "latitude": 10.0, "longitude": 80.0,
            "speed": 5.0, "heading": 90.0, "course": 90.0, "timestamp": ts}


def test_stale_removal_handles_naive_and_aware():
    vsm = VesselStateManager()
    now = datetime.now(timezone.utc)
    vsm.update_vessel(_obs("aware_old", now - timedelta(seconds=10_000)))
    vsm.update_vessel(_obs("naive_old",
                           (now - timedelta(seconds=10_000)).replace(tzinfo=None)))
    vsm.update_vessel(_obs("fresh", now))
    removed = vsm.remove_stale_vessels(stale_seconds=300)
    assert removed == 2
    assert vsm.get_vessel_state("fresh") is not None
    assert vsm.get_vessel_state("aware_old") is None
    assert vsm.get_vessel_state("naive_old") is None


def test_stale_removal_fails_safe_on_bad_timestamp():
    vsm = VesselStateManager()
    vsm.update_vessel(_obs("bad", "not-a-timestamp"))
    # Must not raise and must not drop the vessel on an unparseable timestamp.
    assert vsm.remove_stale_vessels(stale_seconds=1) == 0
    assert vsm.get_vessel_state("bad") is not None


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"{name}  OK")
    print("all security tests passed")
