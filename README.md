# ZeroSight

### Behavioral Anomaly Detection for Zero-Day Protocol Exploits Using Network Flow Fingerprinting

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100%2B-green)](https://fastapi.tiangolo.com)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3%2B-orange)](https://scikit-learn.org)
[![License](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)
[![Dataset](https://img.shields.io/badge/Dataset-CICIDS2017-red)](https://www.unb.ca/cic/datasets/ids-2017.html)

> **INFO-I533 - Systems and Protocol Security and Information Assurance** | Term Project
> Zero-day network anomaly detection using Isolation Forest + Random Forest + SHAP-style XAI, trained on synthetic CICIDS2017, served via FastAPI with a real PCAP binary parser and interactive dashboard.

---

## Live Demo

| Platform | Link |
|---|---|
| Web App (Render) | `https://zerosight.onrender.com` |
| Demo Video | [Google Drive / YouTube link] |
| GitHub | `https://github.com/Sudeep72/ZeroSight` |

---

## What Makes This Different from Signature-Based IDS

| Feature | Signature IDS (Snort/Suricata) | ZeroSight |
|---|---|---|
| Zero-day detection | No | Yes - behavior-based |
| Needs attack signatures | Yes | No |
| Explainability (XAI) | Black box | SHAP-style feature attribution |
| PCAP analysis | Partial | Full binary parser - client-side, no upload |
| Real-time API | No | FastAPI REST |
| Attack classification | No | 17 CICIDS2017 attack classes |
| False positive explanation | None | Per-flow deviation table + XAI verdict |

---

## Model Performance

| Model | Accuracy | Precision | Recall | F1 Score |
|---|---|---|---|---|
| Isolation Forest | 89.6% | 89.4% | 90.3% | 89.9% |
| Random Forest (macro-18) | 90.8% | 89.4% | 94.4% | 89.5% |

- Training dataset: 19,950 flows - 9,750 BENIGN + 600 x 17 attack classes
- Features: 31 CICIDS2017 network flow features
- IF threshold: 0.50043 (90th percentile of BENIGN training scores)
- Score divisor: 0.17290 (attack p95 minus threshold)
- Validated on real Firefox browsing traffic: 500 packets, 12 bidirectional flows

> **Note on metrics:** Binary recall on synthetic CICIDS2017 = 100% (trivially separable). All metrics reported as macro-average across 18 classes, which is the operationally meaningful number.

---

## Architecture

```
+------------------------------------------------------------------+
|                         INPUT LAYER                              |
|   .pcap Upload (client-side)  -  Manual Flow Entry  -  REST API  |
+------------------------------+-----------------------------------+
                               |
                               v
+------------------------------------------------------------------+
|                    FEATURE EXTRACTION                            |
|   Binary PCAP Parser - Real timestamps (ts_sec x 10^6 + ts_usec) |
|   Canonical 5-tuple flow grouping (fwd/bwd tracking)             |
|   IAT from sorted real timestamps - Active/Idle split at 1s      |
|   31 CICIDS2017 Feature Vector - StandardScaler (fit on BENIGN)  |
+------------------------------+-----------------------------------+
                               |
              +----------------+-----------------+
              v                v                 v
    +--------------+  +---------------+  +---------------+
    |  ISOLATION   |  |    RANDOM     |  |     XAI       |
    |  FOREST      |  |    FOREST     |  |    ENGINE     |
    |  300 trees   |  |   100 trees   |  |  SHAP-style   |
    |  Unsupervised|  |   18 classes  |  |  Attribution  |
    |  BENIGN-only |  |  Supervised   |  |  Top-15 feats |
    |  Score 0-100 |  |  + FP guard   |  |  + Rate feats |
    +------+-------+  +------+--------+  +------+--------+
           |                 |                  |
           +-----------------+------------------+
                               |
                               v
+------------------------------------------------------------------+
|                       THREAT SCORING                             |
|   ThreatScore = clip( (IF_raw - theta) / delta x 100, 0, 100 )   |
|   Severity: CLEAN - LOW - MEDIUM - HIGH - CRITICAL               |
|   Cross-model consensus: IF + RF agreement escalates severity    |
|   FP suppression: RF BENIGN >= 85% confidence caps score at 25   |
+------------------------------+-----------------------------------+
                               |
                               v
+--------------------------------------------------------------------+
|                   FASTAPI BACKEND + DASHBOARD                      |
|   POST /predict  -  POST /batch  -  GET /model-info  -  GET /health|
|   Dashboard: Threat Score, SHAP Bar Chart, Deviation Table,        |
|   Batch PCAP Results, Alert Feed, Model Info                       |
+--------------------------------------------------------------------+
```

---

## Project Structure

```
ZeroSight/
- data/
  - generate_cicids.py       # Synthetic CICIDS2017 dataset generator
  - cicids2017_synthetic.csv # Generated training data (git-ignored)
- frontend/
  - dashboard_full.html      # Single-page dashboard: PCAP parser + XAI + results
- model.py                   # Training + inference: IF + RF + SHAP attribution
- server.py                  # FastAPI backend
- test.py                    # 25-test suite (unittest) - server must be running
- requirements.txt
- README.md
```

**Generated artifacts (not committed):**

```
IF.pkl         # Trained Isolation Forest
RF.pkl         # Trained Random Forest
scaler.pkl     # StandardScaler fit on BENIGN flows
meta.json      # Threshold, divisor, feature importances, metrics
```

---

## Quick Start

### Prerequisites

```bash
pip install -r requirements.txt
```

Or manually:

```bash
pip install fastapi uvicorn scikit-learn numpy pandas
```

### Step 1 - Generate Dataset

```bash
python data/generate_cicids.py
# Generates: data/cicids2017_synthetic.csv
# 19,950 rows - 9,750 BENIGN (4 subtypes) + 600 x 17 attack classes
```

> **Use real CICIDS2017:** Download from [unb.ca/cic](https://www.unb.ca/cic/datasets/ids-2017.html) and adapt `generate_cicids.py` to load it directly.

### Step 2 - Train Models

```bash
python model.py --train
# Trains: Isolation Forest (300 trees, BENIGN-only)
#       + Random Forest (100 trees, 18 classes, noise augmented)
# Saves: IF.pkl, RF.pkl, scaler.pkl, meta.json
# Prints: IF accuracy, RF macro metrics, threshold, divisor
```

### Step 3 - Start API Server

```bash
uvicorn server:app --reload --host 0.0.0.0 --port 8000
# Dashboard: http://localhost:8000
# API docs:  http://localhost:8000/docs
```

### Step 4 - Run Tests

```bash
# In a separate terminal (server must be running first)
python test.py
# Expected: 25/25 tests passing
```

---

## API Reference

### `POST /predict` - Single Flow

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "Flow Duration": 229883,
    "Flow Packets/s": 1513.8,
    "Flow Bytes/s": 1560655,
    "Total Fwd Packets": 30,
    "Total Backward Packets": 318,
    "SYN Flag Count": 2,
    "ACK Flag Count": 347
  }'
```

**Response:**

```json
{
  "score": 100,
  "severity": "CRITICAL",
  "is_anomaly": true,
  "if_raw": 0.68017,
  "if_threshold": 0.50043,
  "attack_label": "BENIGN",
  "rf_confidence": 0.61,
  "top_features": [
    { "feature": "Bwd Packets/s",      "contribution": 0.3312 },
    { "feature": "Total Bwd Packets",  "contribution": 0.2461 },
    { "feature": "SYN Flag Count",     "contribution": 0.1230 }
  ]
}
```

### `POST /batch` - Multiple Flows

```bash
curl -X POST http://localhost:8000/batch \
  -H "Content-Type: application/json" \
  -d '{ "flows": [{...}, {...}] }'
```

### `GET /model-info` - Metrics and Feature Importances

```json
{
  "if_accuracy": 0.896,
  "rf_accuracy": 0.908,
  "rf_macro_recall": 0.944,
  "if_threshold": 0.50043,
  "score_divisor": 0.17290,
  "feature_importances": { "Bwd Packets/s": 0.142, "Flow Packets/s": 0.118 }
}
```

### `GET /health` - Server Status

```json
{ "status": "ok", "models_loaded": true }
```

---

## How It Works

### Detection Pipeline

```
Flow Features
      |
      v
StandardScaler (z-score: xi = (xi - mean) / std)
      |
      +---> Isolation Forest: IF_raw = -score_samples(f_scaled)
      |         Decision: anomaly if IF_raw >= theta (0.50043)
      |
      +---> Random Forest: attack_label + rf_conf in [0, 1]
      |         FP guard: rf_conf >= 0.85 + BENIGN -> cap score at 25
      |         Protocol guard: UDP flows -> nullify TCP-only labels
      |
      +---> XAI: contribution_i = |scaled_value_i| x RF_importance_i
                 Top-15 by contribution + guaranteed rate features
```

### Isolation Forest - Zero-Day Capable

- Unsupervised: trained on BENIGN-only traffic, no attack labels needed
- 300 trees, each randomly partitions the 31-dimensional feature space
- Anomalous flows are isolated in fewer splits - lower path length - higher raw score
- Threshold at 90th percentile of BENIGN scores: theta = 0.50043
- Normalized threat score: `ThreatScore = clip( (IF_raw - theta) / delta x 100, 0, 100 )`

### Random Forest - 18-Class Supervised Classifier

- Trained on all 19,950 labeled flows with Gaussian noise augmentation (sigma = 0.30)
- 100 trees, max_depth = 8, class_weight = balanced
- Provides specific attack label + confidence for every flow
- Cross-model consensus: IF anomaly + RF confirms - severity escalated one level

### XAI Feature Attribution

```python
contribution_i = abs(scaled_value_i) * RF_importance_i
# scaled_value  = StandardScaler z-score (deviation from BENIGN mean)
# RF_importance = Gini impurity importance from trained Random Forest
# Returns top-15 features + guaranteed: Flow Pkt/s, Flow Bytes/s,
#                                        Fwd Pkt/s, Bwd Pkt/s
```

### PCAP Parser (Browser-Side)

The dashboard includes a full binary PCAP parser that runs entirely in the browser:

1. Parse global PCAP header: magic number, link type
2. Extract per-packet timestamps: `ts_sec * 10^6 + ts_usec` (microseconds)
3. Group by canonical 5-tuple: `min(src,dst):sport:dport:proto` for bidirectional flows
4. Track forward and backward packets separately
5. Compute IAT from sorted real timestamps
6. Split active/idle at 1-second gaps
7. Discard flows with fewer than 2 packets
8. Emit 31-feature CICIDS2017-compatible record per flow

No PCAP file is uploaded to any server. All parsing happens client-side.

### Severity Levels

| Score | Severity | Recommended Action |
|---|---|---|
| >= 80 | CRITICAL | Immediate investigation |
| >= 60 | HIGH | Urgent review |
| >= 35 | MEDIUM | Monitor and correlate |
| >= 5  | LOW | Log and watch |
| < 5   | CLEAN | No action needed |

---

## Dataset Details

ZeroSight trains on a 19,950-sample synthetic dataset generated by `data/generate_cicids.py`:

| Class | Samples | Notes |
|---|---|---|
| BENIGN - Web | 4,875 | 5-30M us duration, 500-1460 byte packets |
| BENIGN - Bulk | 1,219 | High-throughput, bwd pkt/s up to 2,000 |
| BENIGN - Interactive | 2,437 | Short 20-300 byte packets, low IAT |
| BENIGN - UDP/DNS | 1,219 | 1-6 packet flows, zero TCP flags |
| DoS Hulk | 600 | High-rate HTTP DoS |
| DoS GoldenEye | 600 | HTTP-layer DoS |
| DoS Slowhttptest | 600 | Slow HTTP DoS |
| DoS slowloris | 600 | Slow header DoS |
| DDoS | 600 | Distributed volumetric |
| PortScan | 600 | Sequential port probe |
| Bot | 600 | C2 callback traffic |
| Infiltration | 600 | Internal recon |
| Heartbleed | 600 | TLS memory leak |
| SSH-Patator | 600 | SSH brute force |
| FTP-Patator | 600 | FTP brute force |
| ICMP Flood | 600 | ICMP volumetric |
| DNS Tunneling | 600 | Data exfiltration over DNS |
| Web Attack - Brute Force | 600 | HTTP login brute force |
| Web Attack - XSS | 600 | Cross-site scripting probing |
| Web Attack - SQL Injection | 600 | SQL injection probing |
| MSSQL Bruteforce | 600 | Database brute force |
| **Total** | **19,950** | 31 CICIDS2017 features |

**Flag model:** Counts are proportional to packet count (ACK ~82%, PSH ~14%), not binary. SYN thresholds: > 3 = SUSPICIOUS, > 8 = ANOMALOUS.

---

## Live PCAP Validation

Tested on real Firefox browsing traffic captured on Kali Linux:

```bash
sudo tcpdump -i eth0 -c 500 -w test_capture.pcap
```

| Flow | Protocol | Rate | Score | Severity | RF Label |
|---|---|---|---|---|---|
| DNS queries (x6) | UDP/53 | 10-162 pkt/s | 0 | CLEAN | BENIGN |
| HTTPS short flows | TCP/443 | 28-202 pkt/s | 12-39 | LOW-MEDIUM | BENIGN |
| HTTPS mid-rate | TCP/443 | 135 pkt/s | 44 | HIGH | BENIGN |
| HTTPS burst (CDN) | TCP/443 | 1,514 pkt/s - 1.52 MB/s | 100 | CRITICAL | BENIGN |

**False positive note:** The CDN burst scores CRITICAL because 1,514 pkt/s in 0.23 s sits at the 99th percentile of the BENIGN training distribution. The XAI layer identifies Bwd Packets/s (0.3312) as the primary driver. RF correctly outputs BENIGN (61%). In production, CDN endpoints should be allowlisted.

---

## Hosting

### Render (Recommended)

```bash
# 1. Push to GitHub
git init && git add . && git commit -m "ZeroSight v2"
git remote add origin https://github.com/Sudeep72/ZeroSight
git push -u origin main

# 2. Go to render.com - New Web Service - Connect GitHub repo
# 3. Settings:
#    Build Command: pip install -r requirements.txt && python data/generate_cicids.py && python model.py --train
#    Start Command: uvicorn server:app --host 0.0.0.0 --port $PORT
#    Plan: Free
```

### Railway

```bash
# railway.app - New Project - Deploy from GitHub
# Add start command: uvicorn server:app --host 0.0.0.0 --port $PORT
```

### Hugging Face Spaces

```bash
# huggingface.co/spaces - Create Space - FastAPI SDK
# Upload all files - point app.py to server.py
```

### Dashboard Only (No Backend)

```bash
# GitHub Pages or Netlify Drop - upload dashboard_full.html
# Works in offline demo mode with built-in JS fallback predictor
```

---

## Known Limitations

- **Synthetic training data:** Generated to replicate CICIDS2017 statistics, not drawn from real production traffic. Real-world class variance may differ, especially for encrypted protocols.
- **SHAP approximation:** Uses `|scaled_value| x RF_importance`, not exact Shapley values. Feature interaction effects are not captured.
- **Burst download false positives:** High-bandwidth CDN downloads and browser connection multiplexing exceed BENIGN distribution bounds at the flow-feature level. Allowlisting recommended for known CDN ranges.
- **Static threshold:** The 90th-percentile threshold is calibrated at training time. Online adaptation to baseline drift is a planned future improvement.

---

## Future Work

- Train on real CICIDS2017 and UNSW-NB15 datasets
- Online threshold adaptation via operator feedback loop
- Streaming PCAP analysis with ring-buffer architecture for continuous monitoring
- Exact SHAP via TreeExplainer for full feature interaction attribution
- Per-application traffic baselining: separate IF model per traffic type

---

## References

1. Sharafaldin, I., Lashkari, A.H., Ghorbani, A.A. (2018). *Toward Generating a New Intrusion Detection Dataset and Intrusion Traffic Characterization*. ICISSP.
2. Liu, F.T., Ting, K.M., Zhou, Z.H. (2008). *Isolation Forest*. IEEE ICDM.
3. Lundberg, S.M., Lee, S.I. (2017). *A Unified Approach to Interpreting Model Predictions*. NeurIPS.
4. Breiman, L. (2001). *Random Forests*. Machine Learning, 45(1), 5-32.
5. Sommer, R., Paxson, V. (2010). *Outside the Closed World: On Using Machine Learning for Network Intrusion Detection*. IEEE S&P.
6. Buczak, A.L., Guven, E. (2016). *A Survey of Data Mining and Machine Learning Methods for Cyber Security Intrusion Detection*. IEEE Communications Surveys and Tutorials.
7. MITRE ATT&CK Framework - Network-Based Techniques (T1046, T1071, T1498).

---

## Course Context

Built for **INFO-I533: Systems and Protocol Security and Information Assurance** at Indiana University Bloomington. Differentiates from the prior **SecureZone** project (Python/Flask, rule-based 7-layer network security) through ML-based behavioral detection, PCAP-native flow analysis without server upload, and per-detection explainability.

---

## License

MIT License - free to use for academic and research purposes.

---

*ZeroSight v2 - INFO-I533 - Systems and Protocol Security and Information Assurance - IU Bloomington - Spring 2026*