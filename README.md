# ZeroSight
### Behavioral Anomaly Detection for Zero-Day Protocol Exploits Using Network Flow Fingerprinting

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100%2B-green)](https://fastapi.tiangolo.com)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.1%2B-orange)](https://scikit-learn.org)
[![License](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)
[![Dataset](https://img.shields.io/badge/Dataset-CICIDS2017-red)](https://www.unb.ca/cic/datasets/ids-2017.html)

> **INFO-I520 — Security for Networked Systems** | Term Project  
> Zero-day network anomaly detection using Isolation Forest + Random Forest + SHAP-style XAI, trained on CICIDS2017, served via FastAPI with a real PCAP binary parser dashboard.

---

## 🔴 Live Demo

| Platform | Link |
|----------|------|
| 🌐 Web App (Render) | `https://zerosight.onrender.com` |
| 📹 Demo Video | [Google Drive / YouTube link] |
| 💻 GitHub | `https://github.com/yourusername/zerosight` |

---

## 📌 What Makes This Different from Signature-Based IDS

| Feature | Signature IDS (Snort/Suricata) | ZeroSight |
|---------|-------------------------------|-----------|
| Zero-day detection | ❌ No | ✅ Yes — behavior-based |
| Needs attack signatures | ✅ Yes | ❌ No |
| Explainability (XAI) | ❌ Black box | ✅ SHAP feature attribution |
| PCAP analysis | ❌ Partial | ✅ Full binary parser |
| Real-time API | ❌ No | ✅ FastAPI REST |
| Attack classification | ❌ No | ✅ 14 CICIDS2017 labels |

---

## 🏗️ Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                         INPUT LAYER                              │
│   .pcap / .pcapng Upload   ·   Manual Flow Entry   ·  REST API  │
└─────────────────────────────┬────────────────────────────────────┘
                              │
                              ▼
┌──────────────────────────────────────────────────────────────────┐
│                    FEATURE EXTRACTION                            │
│   Binary PCAP Parser → 5-tuple Flow Aggregation                 │
│   → 31 CICIDS2017 Feature Vector → StandardScaler               │
└─────────────────────────────┬────────────────────────────────────┘
                              │
              ┌───────────────┼───────────────┐
              ▼               ▼               ▼
    ┌──────────────┐  ┌──────────────┐  ┌──────────────┐
    │  ISOLATION   │  │   RANDOM     │  │     XAI      │
    │  FOREST      │  │   FOREST     │  │   ENGINE     │
    │  200 trees   │  │  150 trees   │  │  SHAP-style  │
    │  Unsupervised│  │  Supervised  │  │  Attribution │
    │  Score 0–100 │  │  14 classes  │  │  Per-feature │
    └──────┬───────┘  └──────┬───────┘  └──────┬───────┘
           │                 │                  │
           └─────────────────┴──────────────────┘
                              │
                              ▼
┌──────────────────────────────────────────────────────────────────┐
│                       THREAT SCORING                             │
│   Final Score (0–100) · Severity: CLEAN/LOW/MEDIUM/HIGH/CRITICAL│
└─────────────────────────────┬────────────────────────────────────┘
                              │
                              ▼
┌──────────────────────────────────────────────────────────────────┐
│                    FASTAPI BACKEND + DASHBOARD                   │
│   /api/predict · /api/batch · /api/status · WebUI @ /           │
└──────────────────────────────────────────────────────────────────┘
```

---

## 📁 Project Structure

```
zerosight/
├── data/
│   └── generate_cicids.py      # Synthetic CICIDS2017 dataset generator
├── models/                     # Saved model artifacts (after training)
│   ├── isolation_forest.pkl
│   ├── random_forest.pkl
│   ├── scaler.pkl
│   └── meta.json
├── frontend/
│   └── dashboard_full.html     # Full dashboard (PCAP + XAI + Results)
├── model.py                    # Training + inference engine (IF + RF + XAI)
├── server.py                   # FastAPI backend
├── test.py                     # 25-test suite (unittest)
├── requirements.txt
└── README.md
```

---

## ⚡ Quick Start

### Prerequisites
```bash
pip install -r requirements.txt
```

### Step 1 — Generate Dataset
```bash
python data/generate_cicids.py
# → data/cicids2017_synthetic.csv (9,991 rows, 14 attack types)
```
> **Use real CICIDS2017:** Download from [unb.ca/cic](https://www.unb.ca/cic/datasets/ids-2017.html) and use `--data path/to/real.csv`

### Step 2 — Train Models
```bash
python model.py --train
# Trains: Isolation Forest (200 trees) + Random Forest (150 trees)
# Saves: models/*.pkl + models/meta.json
```

### Step 3 — Start API Server
```bash
uvicorn server:app --reload --host 0.0.0.0 --port 8000
# Dashboard: http://localhost:8000
# API docs: http://localhost:8000/docs
```

### Step 4 — Run Tests
```bash
python test.py
# Runs 25 unit tests: model load, IF detection, XAI, batch, edge cases, metrics
```

---

## 🔌 API Reference

### `POST /api/predict` — Single Flow
```bash
curl -X POST http://localhost:8000/api/predict \
  -H "Content-Type: application/json" \
  -d '{
    "flow_duration": 5000,
    "flow_bytes_per_s": 3000000,
    "flow_packets_per_s": 15000,
    "syn_flag_count": 1,
    "ack_flag_count": 0
  }'
```

**Response:**
```json
{
  "threat_score": 87,
  "severity": "CRITICAL",
  "is_anomaly": true,
  "if_raw_score": 0.612341,
  "attack_label": "DoS Hulk",
  "rf_confidence": 94.2,
  "top_features": [
    { "feature": "Flow Packets/s", "contribution": 2.9674, "value": 15000 },
    { "feature": "Flow Bytes/s",   "contribution": 1.4821, "value": 3000000 }
  ],
  "latency_ms": 3.1
}
```

### `POST /api/batch` — Multiple Flows
```bash
curl -X POST http://localhost:8000/api/batch \
  -H "Content-Type: application/json" \
  -d '{"flows": [{...}, {...}]}'
```

### `GET /api/status` — Health Check
```json
{
  "status": "ok",
  "models_ready": true,
  "model_meta": { "if_accuracy": 0.7694, "rf_accuracy": 0.998 }
}
```

### `GET /api/sample?scenario=dos` — Load Sample
Scenarios: `benign`, `dos`, `portscan`, `bot`, `ddos`, `infiltration`

---

## 📊 Model Performance

| Model | Accuracy | Precision | Recall | F1 Score |
|-------|----------|-----------|--------|----------|
| Isolation Forest | 76.9% | 70.7% | 72.2% | 71.5% |
| Random Forest | **99.8%** | **99.9%** | **99.6%** | **99.75%** |

**Attack labels detected (CICIDS2017):**
DoS Hulk · DDoS · PortScan · Bot · DoS GoldenEye · FTP-Patator · SSH-Patator · DoS slowloris · DoS Slowhttptest · Web Attack XSS · Web Attack Brute Force · Infiltration · Heartbleed

---

## 🌐 Hosting (Free Platforms)

### Option 1 — Render (Recommended)
```bash
# 1. Push to GitHub
git init && git add . && git commit -m "ZeroSight v2"
git remote add origin https://github.com/yourusername/zerosight
git push -u origin main

# 2. Go to render.com → New Web Service → Connect GitHub repo
# 3. Settings:
#    Build Command:  pip install -r requirements.txt && python model.py --train
#    Start Command:  uvicorn server:app --host 0.0.0.0 --port $PORT
#    Plan: Free
```

### Option 2 — Railway
```bash
# railway.app → New Project → Deploy from GitHub
# Add start command: uvicorn server:app --host 0.0.0.0 --port $PORT
```

### Option 3 — Hugging Face Spaces (Gradio/FastAPI)
```bash
# huggingface.co/spaces → Create Space → FastAPI SDK
# Upload all files → add app.py pointing to server.py
```

### Dashboard Only (No Backend)
```bash
# GitHub Pages / Netlify Drop — just upload dashboard_full.html
# Works in offline demo mode (built-in JS fallback predictor)
```

---

## 🧠 How It Works

### Detection Pipeline
```
Flow Features → StandardScaler → Isolation Forest (anomaly score)
                               → Random Forest  (attack label)
                               → XAI Engine     (feature attribution)
                               → Threat Score [0–100]
```

### Isolation Forest (Zero-Day Capable)
- Unsupervised — no attack labels needed during detection
- 200 trees, each randomly partitions feature space
- Anomalous flows are isolated in fewer splits → lower path length → high score
- Raw score normalized to 0–100: `score = clip((raw - 0.25) / 0.45 * 100, 0, 100)`

### XAI Feature Attribution
```python
contribution[i] = |scaled_value[i]| × feature_importance[i]
# feature_importance from Random Forest (Gini-based)
# Returns top-10 features driving the anomaly score
```

### Severity Levels
| Score | Severity | Recommended Action |
|-------|----------|--------------------|
| ≥ 80 | CRITICAL | Immediate isolation |
| ≥ 60 | HIGH | Urgent investigation |
| ≥ 35 | MEDIUM | Monitor + correlate |
| ≥ 5  | LOW | Log and watch |
| < 5  | CLEAN | No action needed |

---

## 📖 References

1. Sharafaldin, I., Lashkari, A.H., Ghorbani, A.A. (2018). *Toward Generating a New Intrusion Detection Dataset and Intrusion Traffic Characterization*. ICISSP.
2. Liu, F.T., Ting, K.M., Zhou, Z.H. (2008). *Isolation Forest*. IEEE ICDM.
3. Lundberg, S.M., Lee, S.I. (2017). *A Unified Approach to Interpreting Model Predictions*. NeurIPS.
4. Breiman, L. (2001). *Random Forests*. Machine Learning, 45(1), 5–32.
5. MITRE ATT&CK Framework — Network-Based Techniques (T1046, T1071, T1498).

---

## 📄 License
MIT License — free to use for academic and research purposes.
