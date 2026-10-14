# IFDS — Client Setup Guide

A beginner-friendly guide to running the Illegal Fishing Detection System and
connecting the **Live AIS** feed with your AISStream.io API key.

You do **not** need to edit any Python code to configure the API key.

---

## What you need

- A Windows, macOS, or Linux computer.
- Python 3.10 or newer.
- Your AISStream.io API key (you already generated one).

> Keep your API key private. Never post it publicly and never upload a `.env`
> file to GitHub.

---

## Section A — Run locally (Windows)

**1. Install Python** (if you don't have it): download from
https://www.python.org/downloads/ and tick "Add Python to PATH" during install.

**2. Get the project**: clone or download this repository, then open it in a
terminal:
```powershell
cd "C:\path\to\illegal-fishing-detection"
```

**3. Create and activate a virtual environment:**
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

**4. Install the requirements:**
```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

**5. Create your `.env` file** from the template:
```powershell
Copy-Item .env.example .env
```

**6. Open `.env`** in any text editor (Notepad is fine) and replace the
placeholder with your real key:
```
AISSTREAM_API_KEY=paste_your_real_key_here
```
Save the file. (You can leave the other `LIVE_AIS_*` lines as they are.)

**7. Start the application:**
```powershell
python -m streamlit run app.py
```

**8. Open the dashboard** — your browser opens automatically, or go to
http://localhost:8501

**9. Select the mode:** in the left sidebar, under **Data Source**, choose
**Live AIS**.

**10. Pick a monitoring region** (e.g. "Tamil Nadu / Coromandel Coast") and
watch the **connection status** and incoming vessels.

### macOS / Linux (short version)
```bash
cd /path/to/illegal-fishing-detection
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
cp .env.example .env         # then edit .env and paste your key
python -m streamlit run app.py
```

---

## Section B — Streamlit Community Cloud deployment

Local `.env` and cloud **Secrets** are **separate** configurations. On
Streamlit Cloud you do not upload `.env`; you use the Secrets box instead.

1. Deploy the app on Streamlit Community Cloud (point it at `app.py`).
2. In the app's **Settings → Secrets**, paste this TOML (replace the value with
   your real key, keep the quotes):
   ```toml
   AISSTREAM_API_KEY = "paste_your_real_key_here"
   ```
3. Save. Streamlit will restart the app automatically; if not, use
   **Reboot app**.
4. Open the app, choose **Live AIS**, and verify the connection status.

> Never put your real key in `README.md`, in the code, or in any committed
> file. Only `.env` (local) or Secrets (cloud) should hold it.

---

## Section C — Troubleshooting

| Symptom | What to do |
|---|---|
| **"Live AIS is not configured"** | The key is missing. Check `.env` has `AISSTREAM_API_KEY=...` (local) or Secrets is set (cloud). Reload the app. |
| **Status shows ERROR / "Unable to authenticate"** | The key was rejected. Re-check it in your AISStream account and paste it again exactly. |
| **Connected but no vessels** | Normal if no vessels are currently transmitting AIS in the selected region. Try a busier region or wait. Coverage is provider-dependent and not guaranteed. |
| **Connection keeps dropping** | The client auto-reconnects with backoff. Check your internet and that your network/firewall allows WebSocket connections to `stream.aisstream.io`. |
| **Live map has no vessels but status is Connected** | Give it a few seconds; positions appear as they arrive. If still empty, the region may have no live traffic right now. |
| **You changed the key** | Update `.env` (local) or Secrets (cloud) and reload/reboot the app. |
| **Network blocks WebSockets** | Some corporate/campus networks block `wss://`. Try another network. |

The app never falls back to fake data while showing a LIVE status. If live data
is unavailable, switch explicitly to **Simulated**, **Historical AIS**, or
**Real-Time Simulation** mode from the sidebar.

---

## Section D — Understanding the dashboard modes

| Mode | What it is |
|---|---|
| **Simulated** | Synthetic AIS demonstration data (built-in 18-vessel fleet). |
| **Historical AIS** | A historical AIS dataset (Danish waters, 2025-02-27). |
| **Real-Time Simulation** | A *replay* of the historical dataset — **not** live. |
| **Live AIS** | **Actual live data** from AISStream.io for the selected region. |

**What a risk score means:** it combines geofencing, behavioural indicators,
an Isolation Forest anomaly score, a temporal sequence anomaly score, and AIS
reporting gaps into a 0–100 score (LOW / MEDIUM / HIGH). It flags **suspicious
movement behaviour worth reviewing** — it is **not** proof of illegal fishing.
Monitoring zones shown on the map are **demonstration zones**, not official
legal boundaries.
