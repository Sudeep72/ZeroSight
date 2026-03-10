"""
server.py  —  ZeroSight FastAPI Backend
========================================
Serves the anomaly detection API and the dashboard HTML.

Install:
    pip install fastapi uvicorn

Run:
    uvicorn server:app --reload --host 0.0.0.0 --port 8000

Endpoints:
    GET  /              → Dashboard HTML
    GET  /api/status    → Model metadata + health
    POST /api/predict   → Single flow prediction
    POST /api/batch     → Batch flow prediction
    GET  /api/sample    → Get sample flows for demo
"""

import json
import time
import random
from pathlib import Path
from typing import Optional

# ── FastAPI (install: pip install fastapi uvicorn) ────────────────────────────
try:
    from fastapi import FastAPI, HTTPException
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import HTMLResponse, JSONResponse
    from pydantic import BaseModel, Field
    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False
    print("[!] FastAPI not installed. Run:  pip install fastapi uvicorn")

import numpy as np
import pandas as pd

# ── Local imports ─────────────────────────────────────────────────────────────
import sys
sys.path.insert(0, str(Path(__file__).parent))
from model import get_detector, FEATURES, BENIGN, META_PATH, MODEL_DIR, IF_PATH, RF_PATH

# ══════════════════════════════════════════════════════════════════════════════
# Pydantic schemas
# ══════════════════════════════════════════════════════════════════════════════

class FlowRequest(BaseModel):
    """Single network flow for prediction. Defaults = BENIGN medians so
    unset fields don't look anomalous to Isolation Forest."""
    flow_duration:                  float = Field(344870.0,  description="Flow Duration (µs)")
    total_fwd_packets:              float = Field(18.0)
    total_backward_packets:         float = Field(14.0)
    total_length_fwd_packets:       float = Field(2181.0)
    total_length_bwd_packets:       float = Field(1808.0)
    fwd_packet_length_mean:         float = Field(300.0)
    bwd_packet_length_mean:         float = Field(280.0)
    flow_bytes_per_s:               float = Field(5625.0)
    flow_packets_per_s:             float = Field(40.0)
    flow_iat_mean:                  float = Field(20549.0)
    flow_iat_std:                   float = Field(17708.0)
    flow_iat_max:                   float = Field(56649.0)
    fwd_iat_mean:                   float = Field(13829.0)
    bwd_iat_mean:                   float = Field(14358.0)
    fwd_packets_per_s:              float = Field(21.0)
    bwd_packets_per_s:              float = Field(17.0)
    packet_length_mean:             float = Field(400.0)
    packet_length_std:              float = Field(199.0)
    packet_length_variance:         float = Field(39600.0)
    fin_flag_count:                 float = Field(0.0)
    syn_flag_count:                 float = Field(1.0)
    rst_flag_count:                 float = Field(0.0)
    psh_flag_count:                 float = Field(1.0)
    ack_flag_count:                 float = Field(1.0)
    average_packet_size:            float = Field(399.0)
    avg_fwd_segment_size:           float = Field(299.0)
    avg_bwd_segment_size:           float = Field(279.0)
    init_win_bytes_forward:         float = Field(16384.0)
    init_win_bytes_backward:        float = Field(32768.0)
    active_mean:                    float = Field(68977.0)
    idle_mean:                      float = Field(68529.0)

    def to_feature_dict(self) -> dict:
        mapping = {
            "Flow Duration":             self.flow_duration,
            "Total Fwd Packets":         self.total_fwd_packets,
            "Total Backward Packets":    self.total_backward_packets,
            "Total Length of Fwd Packets": self.total_length_fwd_packets,
            "Total Length of Bwd Packets": self.total_length_bwd_packets,
            "Fwd Packet Length Mean":    self.fwd_packet_length_mean,
            "Bwd Packet Length Mean":    self.bwd_packet_length_mean,
            "Flow Bytes/s":              self.flow_bytes_per_s,
            "Flow Packets/s":            self.flow_packets_per_s,
            "Flow IAT Mean":             self.flow_iat_mean,
            "Flow IAT Std":              self.flow_iat_std,
            "Flow IAT Max":              self.flow_iat_max,
            "Fwd IAT Mean":              self.fwd_iat_mean,
            "Bwd IAT Mean":              self.bwd_iat_mean,
            "Fwd Packets/s":             self.fwd_packets_per_s,
            "Bwd Packets/s":             self.bwd_packets_per_s,
            "Packet Length Mean":        self.packet_length_mean,
            "Packet Length Std":         self.packet_length_std,
            "Packet Length Variance":    self.packet_length_variance,
            "FIN Flag Count":            self.fin_flag_count,
            "SYN Flag Count":            self.syn_flag_count,
            "RST Flag Count":            self.rst_flag_count,
            "PSH Flag Count":            self.psh_flag_count,
            "ACK Flag Count":            self.ack_flag_count,
            "Average Packet Size":       self.average_packet_size,
            "Avg Fwd Segment Size":      self.avg_fwd_segment_size,
            "Avg Bwd Segment Size":      self.avg_bwd_segment_size,
            "Init_Win_bytes_forward":    self.init_win_bytes_forward,
            "Init_Win_bytes_backward":   self.init_win_bytes_backward,
            "Active Mean":               self.active_mean,
            "Idle Mean":                 self.idle_mean,
        }
        return mapping


class BatchRequest(BaseModel):
    flows: list[FlowRequest]


# ══════════════════════════════════════════════════════════════════════════════
# Sample flows for the demo UI
# ══════════════════════════════════════════════════════════════════════════════

SAMPLE_FLOWS = {
    "benign": {
        "label": "Normal HTTPS Traffic",
        "flow_duration": 500000, "total_fwd_packets": 18, "total_backward_packets": 14,
        "total_length_fwd_packets": 2200, "total_length_bwd_packets": 1800,
        "fwd_packet_length_mean": 300, "bwd_packet_length_mean": 280,
        "flow_bytes_per_s": 8000, "flow_packets_per_s": 60,
        "flow_iat_mean": 30000, "flow_iat_std": 25000, "flow_iat_max": 80000,
        "fwd_iat_mean": 20000, "bwd_iat_mean": 20000,
        "fwd_packets_per_s": 30, "bwd_packets_per_s": 25,
        "packet_length_mean": 400, "packet_length_std": 200, "packet_length_variance": 40000,
        "fin_flag_count": 1, "syn_flag_count": 1, "rst_flag_count": 0,
        "psh_flag_count": 1, "ack_flag_count": 1,
        "average_packet_size": 400, "avg_fwd_segment_size": 300, "avg_bwd_segment_size": 280,
        "init_win_bytes_forward": 65535, "init_win_bytes_backward": 65535,
        "active_mean": 100000, "idle_mean": 100000,
    },
    "dos": {
        "label": "DoS Hulk Attack",
        "flow_duration": 5000, "total_fwd_packets": 1500, "total_backward_packets": 200,
        "total_length_fwd_packets": 150000, "total_length_bwd_packets": 5000,
        "fwd_packet_length_mean": 100, "bwd_packet_length_mean": 25,
        "flow_bytes_per_s": 3000000, "flow_packets_per_s": 15000,
        "flow_iat_mean": 200, "flow_iat_std": 100, "flow_iat_max": 1000,
        "fwd_iat_mean": 150, "bwd_iat_mean": 5000,
        "fwd_packets_per_s": 12000, "bwd_packets_per_s": 2000,
        "packet_length_mean": 105, "packet_length_std": 20, "packet_length_variance": 400,
        "fin_flag_count": 0, "syn_flag_count": 1, "rst_flag_count": 0,
        "psh_flag_count": 0, "ack_flag_count": 0,
        "average_packet_size": 105, "avg_fwd_segment_size": 100, "avg_bwd_segment_size": 25,
        "init_win_bytes_forward": 8192, "init_win_bytes_backward": 0,
        "active_mean": 1000, "idle_mean": 500,
    },
    "portscan": {
        "label": "Port Scan",
        "flow_duration": 100, "total_fwd_packets": 1, "total_backward_packets": 0,
        "total_length_fwd_packets": 40, "total_length_bwd_packets": 0,
        "fwd_packet_length_mean": 40, "bwd_packet_length_mean": 0,
        "flow_bytes_per_s": 400, "flow_packets_per_s": 10000,
        "flow_iat_mean": 100, "flow_iat_std": 10, "flow_iat_max": 200,
        "fwd_iat_mean": 100, "bwd_iat_mean": 0,
        "fwd_packets_per_s": 10000, "bwd_packets_per_s": 0,
        "packet_length_mean": 40, "packet_length_std": 5, "packet_length_variance": 25,
        "fin_flag_count": 0, "syn_flag_count": 1, "rst_flag_count": 1,
        "psh_flag_count": 0, "ack_flag_count": 0,
        "average_packet_size": 40, "avg_fwd_segment_size": 40, "avg_bwd_segment_size": 0,
        "init_win_bytes_forward": 1024, "init_win_bytes_backward": 0,
        "active_mean": 0, "idle_mean": 0,
    },
    "bot": {
        "label": "Bot / C2 Beaconing",
        "flow_duration": 5000000, "total_fwd_packets": 5, "total_backward_packets": 5,
        "total_length_fwd_packets": 400, "total_length_bwd_packets": 400,
        "fwd_packet_length_mean": 80, "bwd_packet_length_mean": 80,
        "flow_bytes_per_s": 160, "flow_packets_per_s": 2,
        "flow_iat_mean": 1000000, "flow_iat_std": 5000, "flow_iat_max": 1100000,
        "fwd_iat_mean": 1000000, "bwd_iat_mean": 1000000,
        "fwd_packets_per_s": 1, "bwd_packets_per_s": 1,
        "packet_length_mean": 80, "packet_length_std": 5, "packet_length_variance": 25,
        "fin_flag_count": 0, "syn_flag_count": 0, "rst_flag_count": 0,
        "psh_flag_count": 1, "ack_flag_count": 1,
        "average_packet_size": 80, "avg_fwd_segment_size": 80, "avg_bwd_segment_size": 80,
        "init_win_bytes_forward": 65535, "init_win_bytes_backward": 65535,
        "active_mean": 500000, "idle_mean": 4500000,
    },
    "ddos": {
        "label": "DDoS Attack",
        "flow_duration": 2000, "total_fwd_packets": 800, "total_backward_packets": 100,
        "total_length_fwd_packets": 48000, "total_length_bwd_packets": 2000,
        "fwd_packet_length_mean": 60, "bwd_packet_length_mean": 20,
        "flow_bytes_per_s": 25000000, "flow_packets_per_s": 500000,
        "flow_iat_mean": 10, "flow_iat_std": 5, "flow_iat_max": 50,
        "fwd_iat_mean": 10, "bwd_iat_mean": 1000,
        "fwd_packets_per_s": 400000, "bwd_packets_per_s": 50000,
        "packet_length_mean": 60, "packet_length_std": 10, "packet_length_variance": 100,
        "fin_flag_count": 0, "syn_flag_count": 1, "rst_flag_count": 0,
        "psh_flag_count": 0, "ack_flag_count": 0,
        "average_packet_size": 60, "avg_fwd_segment_size": 60, "avg_bwd_segment_size": 20,
        "init_win_bytes_forward": 512, "init_win_bytes_backward": 0,
        "active_mean": 500, "idle_mean": 100,
    },
    "infiltration": {
        "label": "Infiltration / Exfiltration",
        "flow_duration": 800000, "total_fwd_packets": 250, "total_backward_packets": 80,
        "total_length_fwd_packets": 250000, "total_length_bwd_packets": 8000,
        "fwd_packet_length_mean": 1000, "bwd_packet_length_mean": 100,
        "flow_bytes_per_s": 322500, "flow_packets_per_s": 412,
        "flow_iat_mean": 2500, "flow_iat_std": 1000, "flow_iat_max": 10000,
        "fwd_iat_mean": 3000, "bwd_iat_mean": 10000,
        "fwd_packets_per_s": 312, "bwd_packets_per_s": 100,
        "packet_length_mean": 960, "packet_length_std": 300, "packet_length_variance": 90000,
        "fin_flag_count": 1, "syn_flag_count": 1, "rst_flag_count": 0,
        "psh_flag_count": 1, "ack_flag_count": 1,
        "average_packet_size": 960, "avg_fwd_segment_size": 1000, "avg_bwd_segment_size": 100,
        "init_win_bytes_forward": 65535, "init_win_bytes_backward": 65535,
        "active_mean": 200000, "idle_mean": 50000,
    },
}


# ══════════════════════════════════════════════════════════════════════════════
# FastAPI app
# ══════════════════════════════════════════════════════════════════════════════

if FASTAPI_AVAILABLE:
    app = FastAPI(
        title="ZeroSight API",
        description="Zero-Day Protocol Anomaly Detection — CICIDS2017 + Isolation Forest + XAI",
        version="2.0.0",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ── Health & status ────────────────────────────────────────────────────
    @app.get("/api/status")
    def status():
        models_ready = IF_PATH.exists() and RF_PATH.exists()
        meta = {}
        if META_PATH.exists():
            with open(META_PATH) as f:
                meta = json.load(f)
        return {
            "status"       : "ok" if models_ready else "models_not_trained",
            "models_ready" : models_ready,
            "model_meta"   : meta,
            "timestamp"    : time.strftime("%Y-%m-%d %H:%M:%S"),
        }

    # ── Single prediction ──────────────────────────────────────────────────
    @app.post("/api/predict")
    def predict(req: FlowRequest):
        try:
            detector = get_detector()
        except FileNotFoundError as e:
            raise HTTPException(status_code=503, detail=str(e))
        result = detector.predict_one(req.to_feature_dict())
        return result

    # ── Batch prediction ───────────────────────────────────────────────────
    @app.post("/api/batch")
    def batch_predict(req: BatchRequest):
        try:
            detector = get_detector()
        except FileNotFoundError as e:
            raise HTTPException(status_code=503, detail=str(e))
        results = [detector.predict_one(f.to_feature_dict()) for f in req.flows]
        n_anomalies = sum(1 for r in results if r["is_anomaly"])
        return {
            "n_flows"    : len(results),
            "n_anomalies": n_anomalies,
            "results"    : results,
        }

    # ── Sample flows ───────────────────────────────────────────────────────
    @app.get("/api/sample")
    def sample(scenario: str = "benign"):
        if scenario not in SAMPLE_FLOWS:
            raise HTTPException(400, f"Unknown scenario. Choose: {list(SAMPLE_FLOWS.keys())}")
        return SAMPLE_FLOWS[scenario]

    @app.get("/api/scenarios")
    def scenarios():
        return {k: v["label"] for k, v in SAMPLE_FLOWS.items()}

    # ── Dashboard HTML (served at /) ───────────────────────────────────────
    @app.get("/", response_class=HTMLResponse)
    def dashboard():
        html_path = Path(__file__).parent / "frontend" / "dashboard_full.html"
        if html_path.exists():
            return HTMLResponse(content=html_path.read_text(encoding="utf-8"))
        return HTMLResponse(content="<h1>Dashboard not found. Build frontend first.</h1>")

    # ── Run directly ───────────────────────────────────────────────────────
    if __name__ == "__main__":
        import uvicorn
        uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True)

else:
    print("FastAPI not available. Install with:  pip install fastapi uvicorn")