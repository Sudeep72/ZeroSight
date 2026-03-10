"""
model.py  —  ZeroSight Anomaly Detection Engine
================================================
Trains an Isolation Forest (+ optional Random Forest classifier) on CICIDS2017
features and exposes a clean predict() API consumed by both the FastAPI server
and the test suite.

Usage (standalone):
    python model.py --train                        # train & save model
    python model.py --train --data data/cicids2017_synthetic.csv
"""

import argparse
import json
import os
import pickle
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.metrics import (classification_report, confusion_matrix,
                             f1_score, precision_score, recall_score,
                             roc_auc_score)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR   = Path(__file__).parent
MODEL_DIR  = BASE_DIR / "models"
MODEL_DIR.mkdir(exist_ok=True)

IF_PATH      = MODEL_DIR / "isolation_forest.pkl"
RF_PATH      = MODEL_DIR / "random_forest.pkl"
SCALER_PATH  = MODEL_DIR / "scaler.pkl"
META_PATH    = MODEL_DIR / "meta.json"

# ── Feature set (subset of CICIDS2017 — most discriminative) ──────────────────
FEATURES = [
    "Flow Duration", "Total Fwd Packets", "Total Backward Packets",
    "Total Length of Fwd Packets", "Total Length of Bwd Packets",
    "Fwd Packet Length Mean", "Bwd Packet Length Mean",
    "Flow Bytes/s", "Flow Packets/s",
    "Flow IAT Mean", "Flow IAT Std", "Flow IAT Max",
    "Fwd IAT Mean", "Bwd IAT Mean",
    "Fwd Packets/s", "Bwd Packets/s",
    "Packet Length Mean", "Packet Length Std", "Packet Length Variance",
    "FIN Flag Count", "SYN Flag Count", "RST Flag Count",
    "PSH Flag Count", "ACK Flag Count",
    "Average Packet Size", "Avg Fwd Segment Size", "Avg Bwd Segment Size",
    "Init_Win_bytes_forward", "Init_Win_bytes_backward",
    "Active Mean", "Idle Mean",
]

LABEL_COL = "Label"
BENIGN    = "BENIGN"

# Median feature values for BENIGN traffic (from synthetic CICIDS2017 training data).
# Used to fill missing/unset fields so partial inputs don't score as extreme anomalies.
BENIGN_DEFAULTS = {
    "Flow Duration": 344870.85, "Total Fwd Packets": 18.0,
    "Total Backward Packets": 14.0, "Total Length of Fwd Packets": 2181.54,
    "Total Length of Bwd Packets": 1808.42, "Fwd Packet Length Mean": 300.19,
    "Bwd Packet Length Mean": 280.3, "Flow Bytes/s": 5625.12,
    "Flow Packets/s": 39.95, "Flow IAT Mean": 20548.95,
    "Flow IAT Std": 17707.99, "Flow IAT Max": 56649.5,
    "Fwd IAT Mean": 13829.44, "Bwd IAT Mean": 14358.06,
    "Fwd Packets/s": 21.21, "Bwd Packets/s": 17.08,
    "Packet Length Mean": 400.62, "Packet Length Std": 199.1,
    "Packet Length Variance": 39638.91, "FIN Flag Count": 0.0,
    "SYN Flag Count": 1.0, "RST Flag Count": 0.0, "PSH Flag Count": 1.0,
    "ACK Flag Count": 1.0, "Average Packet Size": 399.16,
    "Avg Fwd Segment Size": 298.82, "Avg Bwd Segment Size": 278.59,
    "Init_Win_bytes_forward": 16384.0, "Init_Win_bytes_backward": 32768.0,
    "Active Mean": 68977.36, "Idle Mean": 68529.49,
}


# ══════════════════════════════════════════════════════════════════════════════
# Data loading & preprocessing
# ══════════════════════════════════════════════════════════════════════════════

def load_data(csv_path: str) -> pd.DataFrame:
    """Load CICIDS2017 CSV (real or synthetic). Normalise column names."""
    df = pd.read_csv(csv_path)
    # strip whitespace from column names (CICIDS has leading spaces sometimes)
    df.columns = df.columns.str.strip()
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.fillna(0, inplace=True)

    # keep only columns we need
    available = [f for f in FEATURES if f in df.columns]
    missing   = [f for f in FEATURES if f not in df.columns]
    if missing:
        print(f"[!] {len(missing)} features not in CSV, filling with 0: {missing[:5]}…")
        for m in missing:
            df[m] = 0.0

    if LABEL_COL not in df.columns:
        raise ValueError(f"CSV must have a '{LABEL_COL}' column.")

    print(f"[✓] Loaded {len(df):,} rows from {csv_path}")
    print(f"    Label distribution:\n{df[LABEL_COL].value_counts().to_string()}")
    return df


def preprocess(df: pd.DataFrame, scaler: StandardScaler = None, fit: bool = True):
    """Extract features, scale, return X, y, scaler."""
    X = df[FEATURES].values.astype(np.float64)
    y = (df[LABEL_COL] != BENIGN).astype(int).values   # 0=benign 1=attack
    y_labels = df[LABEL_COL].values

    if fit:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
    else:
        X_scaled = scaler.transform(X)

    return X_scaled, y, y_labels, scaler


# ══════════════════════════════════════════════════════════════════════════════
# Training
# ══════════════════════════════════════════════════════════════════════════════

def train(csv_path: str):
    """Train Isolation Forest + Random Forest. Save models to disk."""
    print("\n" + "═"*60)
    print("  ZeroSight — Model Training")
    print("═"*60)

    df = load_data(csv_path)
    X, y, y_labels, scaler = preprocess(df, fit=True)
    X_train, X_test, y_train, y_test, yl_train, yl_test = train_test_split(
        X, y, y_labels, test_size=0.25, random_state=42, stratify=y)

    # ── 1. Isolation Forest (unsupervised) ───────────────────────────────────
    print("\n[1/3] Training Isolation Forest …")

    # Train IF on BENIGN-only — learns the normal traffic distribution.
    # contamination = actual attack ratio in dataset, capped at 0.45.
    benign_mask_train = (yl_train == BENIGN)
    X_train_benign    = X_train[benign_mask_train]
    n_benign          = len(X_train_benign)
    print(f"    IF training on {n_benign:,} BENIGN-only samples")

    contamination = float(np.mean(y_train))
    contamination = round(max(0.05, min(0.45, contamination)), 4)
    score_threshold = 0.0   # placeholder; overridden by percentile below
    print(f"    Contamination: {contamination}")

    iforest = IsolationForest(
        n_estimators=300,
        max_samples=min(4096, n_benign),
        contamination=contamination,
        random_state=42,
        n_jobs=-1,
    )
    t0 = time.time()
    iforest.fit(X_train_benign)
    print(f"    Done in {time.time()-t0:.2f}s")

    # Calibrate threshold + divisor from actual score distributions
    benign_val_scores  = -iforest.score_samples(X_train_benign)
    attack_mask_test   = (y_test == 1)
    attack_scores_test = -iforest.score_samples(X_test[attack_mask_test])

    # threshold = 90th pct of BENIGN (10% FP rate — better for real traffic)
    score_threshold = float(np.percentile(benign_val_scores, 90))  # 90th pct = fewer false positives
    # divisor = span from threshold to 95th pct of ATTACK scores → maps to 0-100
    attack_p95      = float(np.percentile(attack_scores_test, 95))
    score_divisor   = max(attack_p95 - score_threshold, 0.01)
    print(f"    Score threshold (90th pct BENIGN): {score_threshold:.5f}")
    print(f"    Score divisor   (attack p95-thr):  {score_divisor:.5f}")

    # Evaluate on full test set
    if_scores_test = -iforest.score_samples(X_test)
    if_pred_test   = (if_scores_test >= score_threshold).astype(int)

    print("\n  Isolation Forest Test Metrics:")
    print(f"    Accuracy  : {np.mean(if_pred_test == y_test)*100:.2f}%")
    print(f"    Precision : {precision_score(y_test, if_pred_test, zero_division=0)*100:.2f}%")
    print(f"    Recall    : {recall_score(y_test, if_pred_test, zero_division=0)*100:.2f}%")
    print(f"    F1 Score  : {f1_score(y_test, if_pred_test, zero_division=0)*100:.2f}%")

    # ── 2. Random Forest (supervised, for label classification) ──────────────
    print("\n[2/3] Training Random Forest Classifier …")
    rf = RandomForestClassifier(
        n_estimators=100,
        class_weight='balanced',
        max_depth=8,           # enough depth to capture attack patterns
        min_samples_leaf=30,   # prevents overfitting to tiny synthetic clusters
        max_features="sqrt",
        min_samples_split=60,
        random_state=42,
        n_jobs=-1,
    )
    rng = np.random.RandomState(42)
    # Noise std=0.30 adds generalisation without destroying class boundaries
    X_train_noisy = X_train + rng.normal(0, 0.30, X_train.shape)

    t0 = time.time()
    rf.fit(X_train_noisy, yl_train)
    print(f"    Done in {time.time()-t0:.2f}s")

    rf_pred_test = rf.predict(X_test)
    rf_binary    = (rf_pred_test != BENIGN).astype(int)

    # Multi-class macro-average — gives a realistic picture unlike binary which hits 100% recall
    from sklearn.metrics import classification_report as cr_fn
    cr = cr_fn(yl_test, rf_pred_test, output_dict=True, zero_division=0)
    rf_macro_prec = cr['macro avg']['precision']
    rf_macro_rec  = cr['macro avg']['recall']
    rf_macro_f1   = cr['macro avg']['f1-score']
    rf_acc        = cr['accuracy']

    # Binary recall is always ~100% on synthetic data (attacks perfectly separated).
    # Use macro-avg for honest reporting in the dashboard.
    print("\n  Random Forest Test Metrics (macro-average over all 18 classes):")
    print(f"    Accuracy  : {rf_acc*100:.2f}%")
    print(f"    Precision : {rf_macro_prec*100:.2f}%")
    print(f"    Recall    : {rf_macro_rec*100:.2f}%")
    print(f"    F1 Score  : {rf_macro_f1*100:.2f}%")
    print(f"  [Note: Binary recall=100% is expected on synthetic data — macro-avg is more informative]")

    # ── 3. Save ───────────────────────────────────────────────────────────────
    print("\n[3/3] Saving models …")
    with open(IF_PATH, "wb") as f:     pickle.dump(iforest, f)
    with open(RF_PATH, "wb") as f:     pickle.dump(rf, f)
    with open(SCALER_PATH, "wb") as f: pickle.dump(scaler, f)

    # ── Save metadata for dashboard ───────────────────────────────────────────
    attack_ratio = float(np.mean(y))
    meta = {
        "features"        : FEATURES,
        "n_features"      : len(FEATURES),
        "n_train_samples" : int(len(X_train)),
        "n_test_samples"  : int(len(X_test)),
        "attack_ratio"    : round(attack_ratio, 4),
        "contamination"   : round(getattr(iforest, "contamination", 0.10), 4),
        "if_score_threshold": round(score_threshold, 6),
        "if_score_divisor"  : round(score_divisor, 6),
        "if_accuracy"     : round(float(np.mean(if_pred_test == y_test)), 4),
        "if_precision"    : round(float(precision_score(y_test, if_pred_test, zero_division=0)), 4),
        "if_recall"       : round(float(recall_score(y_test, if_pred_test, zero_division=0)), 4),
        "if_f1"           : round(float(f1_score(y_test, if_pred_test, zero_division=0)), 4),
        "rf_accuracy"     : round(rf_acc, 4),
        "rf_precision"    : round(rf_macro_prec, 4),
        "rf_recall"       : round(rf_macro_rec, 4),
        "rf_f1"           : round(rf_macro_f1, 4),
        "attack_labels"   : sorted(df[LABEL_COL].unique().tolist()),
        "feature_importances": {
            FEATURES[i]: round(float(v), 6)
            for i, v in enumerate(rf.feature_importances_)
        },
        "trained_at"      : time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    with open(META_PATH, "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\n[✓] Models saved to {MODEL_DIR}/")
    print(f"    isolation_forest.pkl | random_forest.pkl | scaler.pkl | meta.json")
    return iforest, rf, scaler, meta


# ══════════════════════════════════════════════════════════════════════════════
# Inference
# ══════════════════════════════════════════════════════════════════════════════

class ZeroSightDetector:
    """
    Thin wrapper around saved models.
    Load once, call predict() repeatedly.
    """

    def __init__(self):
        self.iforest = None
        self.rf      = None
        self.scaler  = None
        self.meta    = {}
        self._loaded = False

    def load(self):
        if self._loaded:
            return self
        if not IF_PATH.exists():
            raise FileNotFoundError(
                "Models not found. Run:  python model.py --train  first.")
        with open(IF_PATH, "rb") as f:     self.iforest = pickle.load(f)
        with open(RF_PATH, "rb") as f:     self.rf      = pickle.load(f)
        with open(SCALER_PATH, "rb") as f: self.scaler  = pickle.load(f)
        with open(META_PATH)  as f:        self.meta    = json.load(f)
        self._loaded = True
        return self

    # ── Single-flow prediction ─────────────────────────────────────────────
    def predict_one(self, flow: dict) -> dict:
        """
        flow: dict with keys matching FEATURES (missing keys default to 0).
        Returns a rich result dict.
        """
        t0 = time.time()
        # Use BENIGN median defaults for missing keys — prevents zero-fill anomalies
        vec = np.array([[float(flow.get(f, BENIGN_DEFAULTS.get(f, 0))) for f in FEATURES]])
        vec_scaled = self.scaler.transform(vec)

        # Isolation Forest — use calibrated score threshold from training
        if_raw  = float(-self.iforest.score_samples(vec_scaled)[0])
        thresh  = float(self.meta.get("if_score_threshold", 0.45))
        if_pred = int(if_raw >= thresh)

        # Normalize: thresh=0, thresh+0.50=100 (wider range = normal flows stay low)
        div   = float(self.meta.get("if_score_divisor", 0.10))
        threat_score = int(np.clip((if_raw - thresh) / div * 100, 0, 100))

        # Random Forest label
        rf_label = str(self.rf.predict(vec_scaled)[0])
        rf_proba = self.rf.predict_proba(vec_scaled)[0]
        rf_classes = self.rf.classes_.tolist()
        top_idx  = int(np.argmax(rf_proba))
        rf_confidence = round(float(rf_proba[top_idx]) * 100, 1)

        # RF cross-check: if IF flags anomaly but RF says BENIGN with
        # >=70% confidence, it is almost certainly a false positive.
        # Cap score at 34 (MEDIUM max) and clear the anomaly flag.
        # RF cross-check: borderline IF + high-confidence BENIGN RF = cap score
        if if_pred and rf_label == BENIGN and rf_confidence >= 85.0:
            threat_score = min(threat_score, 25)  # cap but keep if_pred

        # SHAP-style: feature contribution = |scaled_value| * feature_importance
        importances = np.array([self.meta["feature_importances"].get(f, 0) for f in FEATURES])
        contributions = np.abs(vec_scaled[0]) * importances
        all_features = [{"feature": FEATURES[i], "contribution": round(float(contributions[i]), 4),
                          "value": round(float(vec[0][i]), 4)}
                         for i in range(len(FEATURES))]
        top_features = sorted(all_features, key=lambda x: x["contribution"], reverse=True)[:15]

        # Always ensure key rate features appear (even if outside top-15)
        # so the deviation table can always show their contribution
        must_show = {'Flow Packets/s', 'Flow Bytes/s', 'Bwd Packets/s', 'Fwd Packets/s'}
        top_names = {t['feature'] for t in top_features}
        for feat_entry in all_features:
            if feat_entry['feature'] in must_show and feat_entry['feature'] not in top_names:
                top_features.append(feat_entry)

        severity = _severity(threat_score, rf_label)

        return {
            "threat_score"    : threat_score,
            "severity"        : severity,
            "is_anomaly"      : bool(if_pred),
            "if_raw_score"    : round(if_raw, 6),
            "if_threshold"    : round(thresh, 6),   # expose threshold for deviation table
            # Always show RF label — use it regardless of IF decision
            # so users see what RF thinks even on borderline flows
            "attack_label"    : rf_label,
            "rf_confidence"   : rf_confidence,
            "top_features"    : top_features,
            "latency_ms"      : round((time.time() - t0) * 1000, 2),
        }

    # ── Batch prediction ───────────────────────────────────────────────────
    def predict_batch(self, flows: list[dict]) -> list[dict]:
        """Predict a list of flows. Returns list of result dicts."""
        return [self.predict_one(f) for f in flows]

    # ── Predict from DataFrame ─────────────────────────────────────────────
    def predict_df(self, df: pd.DataFrame) -> pd.DataFrame:
        """Predict a DataFrame (with CICIDS columns). Returns df + result cols."""
        available = [f for f in FEATURES if f in df.columns]
        missing   = [f for f in FEATURES if f not in df.columns]
        X = df.reindex(columns=FEATURES, fill_value=0).values.astype(np.float64)
        X_scaled = self.scaler.transform(X)

        if_raw   = -self.iforest.score_samples(X_scaled)
        if_pred  = (self.iforest.predict(X_scaled) == -1).astype(int)
        thresh_b = float(self.meta.get("if_score_threshold", 0.45))
        div_b    = float(self.meta.get("if_score_divisor", 0.10))
        ts       = np.clip((if_raw - thresh_b) / div_b * 100, 0, 100).astype(int)
        rf_labels = self.rf.predict(X_scaled)

        out = df.copy()
        out["threat_score"] = ts
        out["is_anomaly"]   = if_pred
        out["if_raw"]       = if_raw.round(6)
        out["attack_label"] = np.where(if_pred, rf_labels, BENIGN)
        out["severity"]     = [_severity(int(s), str(rl)) for s, rl in zip(ts, rf_labels)]
        return out


# ══════════════════════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════════════════════

def _severity(score: int, rf_label: str = None) -> str:
    """Severity from IF score. RF attack label escalates one level."""
    if score >= 80: base = "CRITICAL"
    elif score >= 60: base = "HIGH"
    elif score >= 35: base = "MEDIUM"
    elif score >  5:  base = "LOW"
    else: base = "CLEAN"
    # RF confirms a specific attack — escalate severity one level
    if rf_label and rf_label != "BENIGN":
        escalate = {"CLEAN":"LOW","LOW":"MEDIUM","MEDIUM":"HIGH",
                    "HIGH":"CRITICAL","CRITICAL":"CRITICAL"}
        return escalate.get(base, base)
    return base



# ── Singleton ──────────────────────────────────────────────────────────────
_detector = ZeroSightDetector()

def get_detector() -> ZeroSightDetector:
    """Return loaded singleton detector (lazy-load on first call)."""
    return _detector.load()


# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ZeroSight model training CLI")
    parser.add_argument("--train", action="store_true", help="Train and save models")
    parser.add_argument("--data", default="data/cicids2017_synthetic.csv",
                        help="Path to CICIDS2017 CSV")
    parser.add_argument("--predict", type=str, default=None,
                        help="Predict a CSV file and print results")
    args = parser.parse_args()

    if args.train:
        if not Path(args.data).exists():
            print(f"[!] Data not found at {args.data}")
            print("    Generating synthetic CICIDS2017 dataset first …")
            import sys
            sys.path.insert(0, str(BASE_DIR / "data"))
            from generate_cicids import generate
            Path("data").mkdir(exist_ok=True)
            generate(out_path=args.data)
        train(args.data)

    elif args.predict:
        detector = get_detector()
        df = pd.read_csv(args.predict)
        df.columns = df.columns.str.strip()
        result_df = detector.predict_df(df)
        print(result_df[["threat_score","severity","is_anomaly","attack_label"]].to_string())

    else:
        parser.print_help()