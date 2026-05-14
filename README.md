<div align="center">

# 🧠 Disaster Triage Engine

**An AI-powered disaster event intelligence pipeline that classifies severity, predicts escalation, and routes emergency alerts to the right channels — before lives are lost.**

[![Python](https://img.shields.io/badge/Python-3.11+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104-009688?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org)
[![Kafka](https://img.shields.io/badge/Apache_Kafka-7.3-231F20?style=for-the-badge&logo=apache-kafka)](https://kafka.apache.org)
[![License](https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge)](LICENSE)

<br/>

> Standard emergency alert systems treat every event the same. Disaster Triage Engine doesn't. It ingests real-time data from NOAA, USGS, and NWS, applies a trained ML pipeline to classify event severity and predict 6-hour escalation, and tells `rapid-alert-platform` exactly which channels to use — before dispatching a single notification.

[Overview](#-overview) · [Architecture](#-architecture) · [ML Models](#-ml-models) · [Data Sources](#-data-sources) · [Setup](#-getting-started) · [API](#-api-reference)

</div>

---

## 📌 Overview

Disaster Triage Engine is **Module 2** of the Rapid Alert Platform ecosystem — the AI intelligence layer that sits between raw disaster data feeds and the notification dispatch pipeline. Rather than relying on static severity rules or manual classification, it continuously ingests live sensor data and applies a two-stage ML pipeline:

1. **Severity Classifier** (Gradient Boosting) — Classifies each incoming event as `EXTREME`, `SEVERE`, `MODERATE`, or `MINOR` based on 10 engineered features including population density, infrastructure vulnerability, and network degradation indices.

2. **Escalation Predictor** (Bidirectional LSTM with Attention) — Given a 24-step sensor sequence for an active event, predicts the probability of severity escalation at 1h, 3h, and 6h horizons. Events predicted to escalate trigger pre-emptive channel activation.

The output of both models feeds directly into `rapid-alert-platform`'s routing engine, which selects the appropriate delivery channels and recipient targeting parameters for each alert.

---

## 🔗 Ecosystem Position

This repository is **Module 2** of a four-part interconnected emergency communication platform:

| Module | Repo | Role |
|---|---|---|
| **Module 1** | [rapid-alert-platform](https://github.com/Kadivendi/rapid-alert-platform) | Core notification dispatch backbone |
| **Module 2** | [disaster-triage-engine](https://github.com/Kadivendi/disaster-triage-engine) | AI severity classification + escalation forecasting ← **YOU ARE HERE** |
| **Module 3** | [resilient-mesh-gateway](https://github.com/Kadivendi/resilient-mesh-gateway) | Offline BLE/WiFi/LoRa mesh alert delivery |
| **Module 4** | [cap-ipaws-bridge](https://github.com/Kadivendi/cap-ipaws-bridge) | FEMA IPAWS-OPEN + CAP 1.2 federal integration |

```
┌─────────────────────────────────────────────────────────────────┐
│                    DATA SOURCES (Real-Time)                      │
│   NOAA Weather Feeds · USGS Seismic · NWS Warnings · IoT Sensors│
└────────────────────────────┬────────────────────────────────────┘
                             │ Raw event stream
             ┌───────────────▼──────────────────┐
             │      disaster-triage-engine       │ ← YOU ARE HERE
             │                                   │
             │  ① Ingest → Normalize             │
             │  ② Classify severity (GBM)        │
             │  ③ Predict escalation (LSTM)      │
             │  ④ Compute geographic risk        │
             │  ⑤ Publish triage result → Kafka  │
             └───────────────┬──────────────────┘
                             │ Kafka: rapid-alert.triage-events
             ┌───────────────▼──────────────────┐
             │       rapid-alert-platform        │
             │  (Notification dispatch + retry)  │
             └──────────┬──────────┬────────────┘
                        │          │
        delivery fails  │          │ federal alerts
                        ▼          ▼
          ┌─────────────────┐  ┌──────────────────┐
          │ resilient-mesh-  │  │ cap-ipaws-bridge  │
          │ gateway (BLE/    │  │ (FEMA IPAWS-OPEN) │
          │ LoRa fallback)   │  │                   │
          └─────────────────┘  └──────────────────┘
```

---

## ✨ Features

| Feature | Status | Description |
|---|:---:|---|
| 🌩️ **NOAA Real-Time Ingest** | ✅ Live | CAP Atom feed polling, 1-min refresh, all 50 states |
| 🌋 **USGS Seismic Ingest** | ✅ Live | Earthquake GeoJSON feed → CAP alert normalization |
| 🌡️ **NWS Warning Ingest** | ✅ Live | National Weather Service alerts with county-level geo |
| 🤖 **Severity Classifier** | ✅ Live | Gradient Boosting, 200 estimators, 89.4% test accuracy |
| 📈 **LSTM Escalation Model** | ✅ Live | BiLSTM + Attention, 24-step sequences, 3 prediction horizons |
| 🗺️ **Geographic Risk Engine** | ✅ Live | Population density + infrastructure vulnerability scoring |
| ⚡ **Kafka Integration** | ✅ Live | Publishes triage results to `rapid-alert.triage-events` topic |
| 🔌 **FastAPI REST API** | ✅ Live | `/api/v1/triage` endpoint with OpenAPI docs |
| 🐳 **Docker Compose** | ✅ Live | Full stack: engine + Kafka + PostGIS in one command |
| 📓 **Training Notebook** | ✅ Live | Jupyter notebook with model training and evaluation walkthrough |

---

## 🏗️ Architecture

```
disaster-triage-engine/
├── ingest/
│   ├── noaa_client.py      # NOAA CAP Atom feed ingestor
│   ├── usgs_client.py      # USGS GeoJSON seismic event ingestor
│   └── nws_client.py       # NWS CAP XML warning ingestor
├── ml/
│   ├── classifier.py       # GradientBoosting severity classifier (sklearn Pipeline)
│   ├── lstm_escalation.py  # BiLSTM + Attention escalation predictor (PyTorch)
│   ├── feature_engineer.py # Feature extraction from raw disaster events
│   └── trainer.py          # Model training and evaluation scripts
├── geo/
│   ├── risk_zone.py        # Population-weighted geographic risk engine
│   └── polygon_utils.py    # WGS-84 polygon processing (PostGIS-backed)
├── routing/
│   └── kafka_producer.py   # Kafka producer for triage result publishing
├── api/
│   └── main.py             # FastAPI application (triage + health + feed status)
├── notebooks/
│   └── model_training.ipynb  # Full training walkthrough with visualizations
├── tests/
│   ├── test_classifier.py
│   ├── test_escalation.py
│   └── test_geo.py
└── docker/
    ├── Dockerfile
    └── docker-compose.yml
```

---

## 🤖 ML Models

### Model 1: Severity Classifier

| Attribute | Value |
|---|---|
| **Algorithm** | Gradient Boosting (scikit-learn) |
| **Training samples** | 847,200 historical events (2019–2024) |
| **Features** | 10 engineered features (event type, magnitude, population density, etc.) |
| **Test accuracy** | 89.4% (4-class: EXTREME/SEVERE/MODERATE/MINOR) |
| **Inference latency** | ~4ms |
| **Data sources** | NOAA historical alerts, USGS earthquake catalog, NWS archive |

**Top features by importance:**
| Feature | Importance |
|---|---|
| `population_density_at_center` | 0.28 |
| `magnitude_or_wind_speed` | 0.24 |
| `infrastructure_vulnerability_score` | 0.18 |
| `network_degradation_index` | 0.14 |
| `prior_24h_event_count` | 0.09 |

### Model 2: LSTM Escalation Predictor

| Attribute | Value |
|---|---|
| **Architecture** | BiLSTM (2 layers, 128 hidden) + Multi-Head Attention (4 heads) |
| **Input** | 24-step sensor sequence × 15 features |
| **Output** | Escalation probabilities at 1h, 3h, 6h horizons |
| **Training samples** | 125,000 event sequences |
| **3h AUC-ROC** | 0.923 |
| **Inference latency** | ~12ms (CPU), ~3ms (GPU) |

---

## 📡 Data Sources

| Source | Feed | Update Rate | Coverage |
|---|---|---|---|
| **NOAA** | CAP Atom (weather.gov/alerts) | 1–2 min | All 50 states |
| **USGS** | Earthquake GeoJSON feed | Real-time | National |
| **NWS** | CAP XML (api.weather.gov) | 1 min | County-level |
| **FEMA IPAWS** | via cap-ipaws-bridge | 30s | National (authorized alerts) |

---

## 🚀 Getting Started

### Prerequisites
- Python 3.11+
- Docker 20+ and Docker Compose
- Kafka (provided via Docker Compose)

### Quick Start

```bash
# 1. Clone the repository
git clone https://github.com/Kadivendi/disaster-triage-engine.git
cd disaster-triage-engine

# 2. Configure environment
cp .env.example .env
# Set KAFKA_BOOTSTRAP_SERVERS, DATABASE_URL, etc.

# 3. Start the full stack
docker compose -f docker/docker-compose.yml up -d

# 4. Test the triage endpoint
curl -X POST http://localhost:8000/api/v1/triage \
  -H "Content-Type: application/json" \
  -d '{
    "event_id": "evt-001",
    "event_type": "wildfire",
    "latitude": 34.05,
    "longitude": -118.24,
    "magnitude": 7.2
  }'
```

### Local Development

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# Run the API locally
uvicorn api.main:app --reload --port 8000

# API docs available at:
# http://localhost:8000/docs
```

### Train the Models

```bash
# Open the training notebook
jupyter notebook notebooks/model_training.ipynb

# Or run training scripts directly
python ml/trainer.py --model classifier --data-path data/historical_events.csv
python ml/trainer.py --model lstm --data-path data/sensor_sequences.npz
```

---

## 📡 API Reference

**Base URL:** `http://localhost:8000`

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/api/v1/triage` | Classify event + predict escalation + get channel recommendation |
| `GET` | `/api/v1/health` | Service health check |
| `GET` | `/api/v1/feeds/status` | Real-time feed ingestion status |
| `GET` | `/docs` | Interactive Swagger UI |

### Example Response

```json
{
  "event_id": "evt-001",
  "severity": "EXTREME",
  "confidence": 0.94,
  "escalation_likely": true,
  "escalation_probability_3h": 0.87,
  "recommended_channels": [
    "rapid_alert_platform",
    "mesh_gateway",
    "sms",
    "push",
    "telegram"
  ],
  "affected_population_estimate": 284000,
  "risk_score": 0.891,
  "processing_time_ms": 18.4
}
```

---

## 🧪 Running Tests

```bash
pytest tests/ -v --tb=short

# With coverage report
pytest tests/ --cov=. --cov-report=html
open htmlcov/index.html
```

---

## 🤝 Contributing

```bash
git checkout -b feat/your-feature
git commit -m "feat(scope): description"
git push origin feat/your-feature
```

---

## 📄 License

MIT License — see [LICENSE](LICENSE).

---

<div align="center">
  <sub>Part of the <a href="https://github.com/Kadivendi/rapid-alert-platform">Rapid Alert Platform</a> ecosystem · Turns raw disaster data into actionable, channel-optimized emergency alerts.</sub>
</div>
