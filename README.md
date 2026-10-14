<div align="center">

# 🛰️ Illegal Fishing Detection System (IFDS)

**AI-assisted maritime vessel-monitoring and suspicious-behaviour detection.**

Ingests vessel movement data → cleans & validates it → derives behavioural
indicators → runs two unsupervised anomaly detectors → computes a unified,
explainable risk score → raises prioritised alerts → persists results → and
presents it all in an interactive Streamlit dashboard.

![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.53-FF4B4B?logo=streamlit&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.8-F7931E?logo=scikitlearn&logoColor=white)
![Live AIS](https://img.shields.io/badge/Live%20AIS-AISStream.io-00B2A9)
![Tests](https://img.shields.io/badge/tests-105%20passing-brightgreen)

</div>

---

## ⚡ TL;DR — get it running in 60 seconds

```powershell
# 1. Clone and enter the project
git clone https://github.com/InfoNaveen/illegal-fishing-detection.git
cd illegal-fishing-detection

# 2. Create a virtual environment and install dependencies
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

# 3. Run the dashboard
python -m streamlit run app.py
```

Open **http://localhost:8501**. That's it — the **Simulated**, **Historical
AIS**, and **Real-Time Simulation** modes work immediately with **no API key**.

> This is a **Python / Streamlit** app. There is no `npm`, no Node, and no
> `npm run dev`. The only run command is `python -m streamlit run app.py`.

Want the genuinely live feed too? See **[🔑 API Key Setup](#-api-key-setup-live-ais)** below.

---

## 📑 Table of contents

1. [⚡ TL;DR](#-tldr--get-it-running-in-60-seconds)
2. [🔑 API Key Setup (Live AIS)](#-api-key-setup-live-ais)
3. [🚀 Full setup, step by step](#-full-setup-step-by-step)
4. [▶️ Running the app](#️-running-the-app)
5. [🧭 Honest scope](#-honest-scope)
6. [🎯 Problem statement](#-problem-statement)
7. [✨ Key features](#-key-features)
8. [🗺️ Data sources (four modes)](#️-data-sources-four-modes)
9. [🏗️ Architecture](#️-architecture)
10. [🔬 Pipeline internals](#-pipeline-internals)
11. [⚙️ Configuration](#️-configuration)
12. [🧪 Testing](#-testing)
13. [☁️ Deployment](#️-deployment)
14. [⚠️ Limitations](#️-limitations)
15. [🔭 Future improvements](#-future-improvements)
16. [📄 Disclaimer & license](#-disclaimer--license)

---

## 🔑 API Key Setup (Live AIS)

The **Live AIS** mode connects to the real **[AISStream.io](https://aisstream.io/)**
WebSocket feed. The other three modes need **no key** — only Live AIS does.

### Step 1 — get a free key

1. Go to **https://aisstream.io/** and create a free account.
2. Open your dashboard and create an **API key** (a ~40-character string).
3. Copy it. You'll paste it in Step 2 — **never into the code or Git**.

### Step 2 — configure the key

**You never edit source code.** Pick the option that matches where you run the app.

<details open>
<summary><b>Option A — Local machine (recommended)</b></summary>

```powershell
# From the project root, create your private .env from the template
Copy-Item .env.example .env
```

Open `.env` and set your key:

```ini
AISSTREAM_API_KEY=paste_your_real_40_character_key_here
```

`.env` is **git-ignored** — it is never committed or pushed. Done.

</details>

<details>
<summary><b>Option B — Streamlit Community Cloud</b></summary>

In your deployed app, open **Settings → Secrets** and add:

```toml
AISSTREAM_API_KEY = "paste_your_real_40_character_key_here"
```

Save. Streamlit restarts the app with the secret available.

</details>

### Step 3 — use it

Run the app, open the sidebar **Data Source** selector, choose **Live AIS**, and
pick a monitoring region.

**Key handling & precedence**

| Source | Checked | Notes |
|---|---|---|
| Environment variable / `.env` | **1st** | Local development |
| `st.secrets` (Streamlit Cloud) | **2nd** | Cloud deployment |

The placeholder (`your_aisstream_api_key_here`) counts as *not configured*. With
**no** key, Live AIS simply shows setup guidance and the other three modes keep
working. The key is **never** printed, logged, shown in the UI, written to the
database, or committed to Git.

> Full step-by-step local + cloud + troubleshooting walkthrough lives in
> **[`CLIENT_SETUP_GUIDE.md`](CLIENT_SETUP_GUIDE.md)**.

---

## 🚀 Full setup, step by step

### Prerequisites

- **Python 3.10 or later** — check with `python --version`
- **Git**

### Windows (PowerShell)

```powershell
git clone https://github.com/InfoNaveen/illegal-fishing-detection.git
cd illegal-fishing-detection
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### macOS / Linux (bash/zsh)

```bash
git clone https://github.com/InfoNaveen/illegal-fishing-detection.git
cd illegal-fishing-detection
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

> **PowerShell execution policy.** If `Activate.ps1` is blocked, run once:
> `Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned`.

---

## ▶️ Running the app

```powershell
python -m streamlit run app.py
```

- Dashboard: **http://localhost:8501**
- Health check: **http://localhost:8501/_stcore/health** → returns `ok`

On the **first run** the two models are trained and saved under `models/` (a few
seconds). Every later run **reuses** them, so startup is near-instant.

**Pick a data mode** from the sidebar **Data Source** selector:

| Mode | API key? | What it does |
|---|---|---|
| **Simulated** (default) | No | Built-in synthetic Bay of Bengal fleet |
| **Historical AIS** | No | Bundled Danish-waters AIS CSV |
| **Real-Time Simulation** | No | Deterministic replay of the historical data |
| **Live AIS** | **Yes** | Real AISStream.io feed (see API Key Setup) |

---

## 🧭 Honest scope

> This is a prototype / final-year-project **demonstrator**. It identifies
> **anomalous and suspicious movement behaviour** and produces risk
> *indicators* — it does **not** prove illegal fishing and makes **no**
> accuracy/precision/recall claims (there is no labelled illegal-fishing ground
> truth). The Live AIS mode uses a real third-party feed whose coverage is not
> guaranteed; the other three modes use historical/synthetic data. Monitoring
> zones shown on the map are **demonstration zones**, not official legal
> boundaries.

---

## 🎯 Problem statement

Illegal, unreported and unregulated (IUU) fishing is hard to monitor at scale.
Vessels broadcast AIS position reports, and suspicious patterns — loitering,
entering restricted areas, abnormal turning, going dark (AIS gaps), or moving
unlike the rest of the fleet — can indicate activity worth investigating. IFDS
demonstrates an end-to-end pipeline that surfaces these behavioural **risk
signals** in an explainable way, so an analyst can decide what to investigate.
It does not make enforcement decisions and does not label vessels as criminal.

---

## ✨ Key features

- Historical AIS CSV ingestion with flexible column normalisation
- Deterministic data cleaning + validation (no fabricated values)
- Source-aware, configurable geographic monitoring zones (ray-casting geofencing)
- Behavioural trajectory analysis (distance, speed stats, turning, loitering,
  stationarity, AIS gaps, duration)
- Isolation Forest point-anomaly detection with a persistent train-once model
- Temporal sequence anomaly detection (PCA sequence autoencoder, persistent)
- Unified, explainable risk engine combining five normalised signals
- Dynamic alerts with human-readable reasons
- Local SQLite persistence (history sink)
- Interactive dark maritime dashboard (map, metrics, inspector, history)
- Controlled **AIS replay simulation** mode with live metrics and dedup alerts
- **Live AIS** mode via the AISStream.io WebSocket API (real feed, India-focused
  configurable regions, secure environment/Secrets key handling, auto-reconnect)
- Four data modes: Simulated · Historical AIS · Real-Time Simulation · Live AIS

---

## 🗺️ Data sources (four modes)

IFDS runs in **four modes**, selected in the sidebar (default: **Simulated**).

### Simulated — Bay of Bengal demonstration

A built-in synthetic fleet of 18 vessels with hand-placed scenarios (e.g. V102,
V087, V215 inside restricted zones). Fully reproducible; used for demonstration
and regression testing. Zones here are **demonstration restricted zones**.

### Historical AIS — Danish waters

Ingests a local historical AIS CSV (`data/ifds_ais_sample.csv`) sourced from
Danish Maritime Authority open AIS data (2025-02-27): ~100,000 raw records
across ~2,107 vessels. This is **historical movement data**, not a live feed.
Monitoring zones here are **demonstration monitoring zones** placed over
observed dense-traffic regions — **not** authoritative legal restricted areas.

### Real-Time (Replay) Simulation

A controlled, deterministic **replay** of the cleaned historical AIS data,
emitted tick-by-tick with live metrics and alerts. **It is not a live AIS feed**
and does not receive vessels in real time.

### Live AIS (AISStream.io)

A genuinely live mode that connects to the **AISStream.io** WebSocket API,
subscribes to a configurable India-focused coastal region, receives real AIS
position reports, normalises and validates them, and feeds them through the
**same** detection pipeline (geofencing → behaviour → Isolation Forest →
temporal → unified risk → alerts). Requires an `AISSTREAM_API_KEY`
(see [🔑 API Key Setup](#-api-key-setup-live-ais) and `CLIENT_SETUP_GUIDE.md`).
Coverage depends on the external provider; "connected with no vessels" is a
legitimate state. Output is a suspicious-behaviour risk indicator, **not**
confirmed illegal fishing. With no key configured, Live AIS shows setup guidance
and the other three modes keep working.

**Monitoring regions** are selectable in the Live AIS view (Tamil Nadu /
Coromandel Coast — default, Bay of Bengal (India East), Andhra Pradesh, Kerala,
Karnataka, Gujarat, Maharashtra/Goa, plus the Danish historical region). These
are **demonstration monitoring regions**, not official maritime boundaries.

> Simulated, Historical AIS, and Real-Time Simulation modes do not use any live
> feed. Only **Live AIS** connects to an external provider, and only when an API
> key is configured.

---

## 🏗️ Architecture

```text
          Data source (Simulated | Historical AIS | Replay | Live AISStream)
                                   |
                           AIS ingestion (ais_loader)
                                   |
                     Data cleaning + validation (data_processing)
                                   |
              ┌────────────────────┴────────────────────┐
              |                                          |
       Geofencing (zones)                     Trajectory / behaviour
       (geofencing)                           analysis (behavior_analysis)
              |                                          |
              └────────────────────┬────────────────────┘
                                   |
                     Feature engineering (21 numeric features)
                                   |
              ┌────────────────────┴────────────────────┐
              |                                          |
     Isolation Forest                           Temporal sequence
     (anomaly_detector +                        autoencoder (temporal_model +
      model_manager, persistent)                temporal_model_manager, persistent)
              |                                          |
              └────────────────────┬────────────────────┘
                                   |
                     Unified risk engine (risk_engine)
                                   |
                        Alert generation (risk_engine /
                        alert_engine for replay)
                                   |
                        SQLite persistence (database)
                                   |
                        Streamlit dashboard (app.py)
```

---

## 🔬 Pipeline internals

### AIS ingestion (`ais_loader.py`)

Loads a local CSV and normalises common AIS column aliases to a fixed internal
schema. Recognised aliases:

| Internal field | Aliases | Required |
|---|---|---|
| `vessel_id` | MMSI, vessel_id, id, ship_id | Yes |
| `latitude`  | LAT, latitude, y | Yes |
| `longitude` | LON, long, lng, longitude, x | Yes |
| `speed`     | SOG, speed, speed_over_ground | No |
| `heading`   | COG, heading, course, course_over_ground | No |
| `timestamp` | BaseDateTime, timestamp, time, datetime | No |

Missing required columns raise a clear error; invalid coordinates are dropped
(never fabricated).

### Data cleaning (`data_processing.py`)

Deterministic cleaning between ingestion and the pipeline: day-first timestamp
parsing (`DD/MM/YYYY HH:MM:SS`), numeric coercion, coordinate validation,
negative-speed removal, heading normalisation (0 valid, 511 sentinel → NaN), and
**exact-duplicate removal** (`vessel_id, timestamp, latitude, longitude`). On
the bundled Danish sample, cleaning reduces 100,000 rows → ~53,597 by removing
exact-duplicate broadcasts while preserving every vessel. It never interpolates
or invents positions, and returns a deterministic data-quality report.

### Monitoring zones (`zones.py`)

Geofencing zones are configurable per data source; the ray-casting
point-in-polygon algorithm is shared and unchanged. Simulated → Bay of Bengal
demonstration restricted zones; Historical/Replay → Danish demonstration
monitoring zones. The Danish polygons are **demonstration monitoring zones
only**, not legally restricted waters.

### Behavioural analysis (`behavior_analysis.py`)

Derives explainable indicators from each vessel's ordered trajectory: Haversine
distance travelled; average/max speed and speed variability; mean/max heading
change and turning rate; stationary ratio; a loitering score (spatial
containment + stationarity); AIS reporting gaps (max/mean + significant-gap
count); trajectory duration; position count. A transparent, non-ML multi-label
classification (`NORMAL_TRANSIT`, `SLOW_MOVEMENT`, `LOITERING`, `HIGH_TURNING`,
`AIS_GAP`, `SPEED_ANOMALY`, `MIXED_SUSPICIOUS`, `INSUFFICIENT_DATA`) and a
plain-language summary accompany the numeric features. These are **suspicious
behaviour indicators**, not proof of illegal fishing.

### Isolation Forest (`anomaly_detector.py`, `model_manager.py`)

An **unsupervised** Isolation Forest (200 estimators, contamination 0.20,
`random_state=42`) scores each vessel's 21-feature row; `decision_function`
output is inverted and min-max normalised to a `[0, 1]` anomaly score. The model
follows a **train-once / load-reuse** lifecycle: it is trained on first run and
saved to `models/isolation_forest.joblib`; later runs reuse the saved artifact
after a compatibility check (feature count, names, order, model version). A
missing/corrupt/incompatible artifact triggers a safe retrain.

### Temporal model (`temporal_model.py`, `temporal_model_manager.py`)

An **unsupervised** temporal **sequence** anomaly detector implemented as a
scikit-learn **PCA sequence autoencoder** (no deep-learning dependency). It
builds fixed-length windows (`sequence_length=10`) from six per-step features
(speed, heading change, lat/lon change, step distance, time gap), compresses and
reconstructs them, and uses the reconstruction error (normalised to `[0, 1]`
against a stored training reference) as a `temporal_anomaly_score`. It evaluates
behaviour **across time windows** rather than independent points. Vessels with
too few observations return score 0 and status `insufficient_data`. The model is
persisted (`models/temporal_model.joblib`) with the same train-once / load-reuse
lifecycle and compatibility checks.

> This is a PCA-based linear sequence autoencoder, chosen for a reliable,
> dependency-light implementation. It is not an LSTM/GRU.

### Unified risk engine (`risk_engine.py`)

A single canonical scoring path combines five normalised `[0, 1]` components,
each counted **exactly once**, with documented weights:

| Component | Weight | Meaning |
|---|---|---|
| Geofence | 0.30 | inside a monitoring zone (1.0) / proximity fraction |
| Isolation Forest | 0.25 | point anomaly score |
| Temporal | 0.15 | sequence anomaly score (0 when unavailable) |
| Behaviour | 0.20 | max of loitering / speed / erratic / turning |
| AIS gap | 0.10 | single unified gap signal (no double-counting) |

`overall_score = 100 × Σ(weightᵢ × componentᵢ)`; **HIGH ≥ 55**, **MEDIUM ≥ 20**,
else **LOW**. Each vessel exposes the component values, a weighted factor
breakdown, an explainable `risk_factors` list, and a plain-language explanation.
Wording is deliberately phrased as *"suspicious fishing-related behaviour
indicators"* — never confirmed illegal fishing.

### Real-time replay simulation (`realtime_simulator.py`, `alert_engine.py`)

The replay mode orders the cleaned historical AIS stream by timestamp and emits
it in configurable **ticks**. Per-vessel current state and a capped recent-track
window are maintained; risk is computed only for currently active vessels each
tick. The `AlertEngine` raises alerts (HIGH_RISK, ZONE_ENTRY, AIS_GAP,
BEHAVIOUR_ANOMALY, TEMPORAL_ANOMALY, COMBINED) with a per-(vessel, type)
**cooldown** to prevent spam, and persists them to SQLite. Stepping is driven by
Streamlit session state with bounded `st.rerun()` — there is no runaway loop.
The UI states plainly that this is a replay of historical data and **not a live
AIS feed**.

### Live AIS client (`live_ais.py`)

Connects to the AISStream.io WebSocket API in a background thread (own asyncio
loop), subscribes to a configurable bounding box, normalises incoming position
reports, maintains bounded per-vessel in-memory state (stale vessels dropped),
and auto-reconnects on disconnect. The API key is resolved via environment →
Streamlit Secrets and is never exposed.

### SQLite persistence (`database.py`)

A local SQLite database (`data/ifds.db`) acts as a **history sink** — the
detection pipeline runs independently of it, and persistence failures surface a
warning without breaking the dashboard. Tables: `vessels` (UPSERT), `positions`
(deduplicated), `features` (JSON), `anomaly_scores`, `risk_scores`, `alerts`.
It is local application history, not cloud storage.

### Dashboard (`app.py`)

Sections: system overview header, data-source selector, fleet risk metrics,
interactive map (zones, trails, real historical track for a selected vessel),
alerts, vessel inspector (position, risk, behaviour analysis, Isolation Forest +
temporal scores, risk factors), ML model status, stored history, and a system
information panel. The Real-Time Simulation mode shows live metrics, a replay
map, and the latest deduplicated alerts.

---

## 📁 Project structure

```
illegal-fishing-detection/
│
├── app.py                      # Streamlit dashboard + pipeline orchestration
├── data_generator.py           # Simulated fleet, trajectories, zones accessor
├── ais_loader.py               # Historical AIS CSV ingestion + normalisation
├── data_processing.py          # Deterministic AIS cleaning + quality report
├── zones.py                    # Source-aware monitoring/zone configuration
├── geofencing.py               # Ray-casting point-in-polygon zone detection
├── feature_engineering.py      # 9 base + 12 behavioural = 21 numeric features
├── behavior_analysis.py        # Trajectory & behaviour metrics + classification
├── anomaly_detector.py         # Isolation Forest scoring (train-once/load)
├── model_manager.py            # Isolation Forest persistent lifecycle (joblib)
├── temporal_model.py           # PCA sequence autoencoder (temporal anomaly)
├── temporal_model_manager.py   # Temporal model persistent lifecycle (joblib)
├── risk_engine.py              # Unified risk scoring + alert generation
├── realtime_simulator.py       # AIS replay simulation (tick-driven)
├── live_ais.py                 # Live AISStream WebSocket client + vessel state
├── alert_engine.py             # Alert triggering with dedup/cooldown
├── map_builder.py              # Folium map construction
├── database.py                 # SQLite persistence (history sink)
│
├── test_ais_loader.py
├── test_data_processing.py
├── test_database.py
├── test_behavior_analysis.py
├── test_model_manager.py
├── test_temporal_model.py
├── test_temporal_model_manager.py
├── test_risk_engine.py
├── test_realtime_simulation.py
├── test_live_ais.py
│
├── data/
│   ├── ifds_ais_sample.csv     # Historical AIS sample — Danish waters (~9 MB, tracked)
│   ├── sample_ais.csv          # Tiny synthetic test fixture (not real AIS)
│   ├── ifds.db                 # SQLite runtime history (git-ignored)
│   └── README.md
├── models/                     # Trained model artifacts (git-ignored, runtime)
│   ├── isolation_forest.joblib
│   └── temporal_model.joblib
│
├── .env.example                # Template for the Live AIS API key
├── CLIENT_SETUP_GUIDE.md       # Full Live AIS setup + troubleshooting
├── requirements.txt
├── .gitignore
└── README.md
```

---

## ⚙️ Configuration

- **AIS CSV path:** `ais_loader.DEFAULT_AIS_CSV_PATH` (default
  `data/ifds_ais_sample.csv`).
- **Database path:** `database.DEFAULT_DB_PATH` (default `data/ifds.db`).
- **Model paths:** `models/isolation_forest.joblib`, `models/temporal_model.joblib`.
- **Risk weights / thresholds:** documented constants in `risk_engine.py`.
- **Temporal sequence length / hidden size:** `temporal_model.py`.
- **Replay ticks / alert cooldown:** `realtime_simulator.py`, `alert_engine.py`.
- **Live AIS region / refresh / stale timeout:** `.env` (see `.env.example`)
  and `LIVE_AIS_REGIONS` in `live_ais.py`.

---

## 🧪 Testing

```powershell
python -m pytest -q
```

**105 tests passing.** The suites cover ingestion, cleaning, persistence,
behaviour analysis, both model lifecycles (including train-once reuse proofs),
the unified risk engine (component isolation, no double-count, thresholds,
V102/V087/V215 regression), the replay simulation (ordering, state, alert
cooldown, DB persistence), and the Live AIS client (subscription message,
message normalisation, vessel-state management, key resolution).

---

## ☁️ Deployment

The app runs with `python -m streamlit run app.py`. The deployed app depends
only on the tracked ~9 MB historical sample — **not** on the original large raw
AIS CSV. The SQLite database and model artifacts are generated at runtime and
are git-ignored; they must not be committed. For Live AIS on Streamlit Community
Cloud, add `AISSTREAM_API_KEY` under **Settings → Secrets** (see
[🔑 API Key Setup](#-api-key-setup-live-ais)).

---

## ⚠️ Limitations

- Default fleet is **simulated**; Historical/Replay modes read a **local CSV
  only** — no live AIS feed except the dedicated **Live AIS** mode.
- The bundled sample is historical Danish AIS; monitoring zones are
  **demonstration** zones, not authoritative legal boundaries.
- No labelled illegal-fishing ground truth → **no accuracy/precision/recall/F1**
  is claimed. Both models are unsupervised.
- The temporal model is a PCA sequence autoencoder, not an LSTM/GRU.
- Risk weights and thresholds are prototype values, not domain-calibrated.
- Persistence is a local SQLite history sink, not an operational data store.
- Live AIS coverage depends entirely on the external AISStream.io provider.

---

## 🔭 Future improvements

- Broader authorised AIS data sources and historical back-fill
- Domain-validated zones and risk calibration
- Labelled evaluation data to enable supervised metrics
- Richer temporal modelling (e.g. recurrent networks) if resources allow
- Longer-horizon historical tracking and analyst workflow tooling

---

## 📄 Disclaimer & license

### Ethical / operational disclaimer

This system is a prototype and academic demonstrator. It identifies
**anomalous / suspicious maritime movement behaviour** using unsupervised
methods on historical, simulated, and (optionally) live third-party data. It
does **not** prove illegal fishing, is **not** an operational surveillance
system, uses **demonstration** monitoring zones, and must **not** be used for
operational maritime enforcement decisions. Any real-world use would require
authorised data sources, domain-expert calibration, and appropriate legal and
regulatory review.

### License

License information will be added separately.
