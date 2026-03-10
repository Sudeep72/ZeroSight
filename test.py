"""
test.py  —  ZeroSight Test Suite
==================================
Tests model correctness, edge cases, performance, and API contracts.

Run:
    python test.py                    # all tests
    python test.py -v                 # verbose
    python test.py -k test_isolation  # filter by name

Requirements:
    python model.py --train           # must run first
"""

import json
import time
import sys
import unittest
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).parent))
from model import (
    FEATURES, BENIGN, IF_PATH, RF_PATH, SCALER_PATH, META_PATH,
    get_detector, load_data, preprocess, ZeroSightDetector,
)

# ── Color helpers ──────────────────────────────────────────────────────────────
GREEN  = "\033[92m"
RED    = "\033[91m"
YELLOW = "\033[93m"
CYAN   = "\033[96m"
BOLD   = "\033[1m"
RESET  = "\033[0m"

def hdr(title):
    print(f"\n{BOLD}{CYAN}{'─'*55}{RESET}")
    print(f"{BOLD}{CYAN}  {title}{RESET}")
    print(f"{BOLD}{CYAN}{'─'*55}{RESET}")


# ══════════════════════════════════════════════════════════════════════════════
# Test helpers — build synthetic flows
# ══════════════════════════════════════════════════════════════════════════════

def _benign_flow(**overrides):
    base = {
        "Flow Duration": 500000, "Total Fwd Packets": 18, "Total Backward Packets": 14,
        "Total Length of Fwd Packets": 2200, "Total Length of Bwd Packets": 1800,
        "Fwd Packet Length Mean": 300, "Bwd Packet Length Mean": 280,
        "Flow Bytes/s": 8000, "Flow Packets/s": 60,
        "Flow IAT Mean": 30000, "Flow IAT Std": 25000, "Flow IAT Max": 80000,
        "Fwd IAT Mean": 20000, "Bwd IAT Mean": 20000,
        "Fwd Packets/s": 30, "Bwd Packets/s": 25,
        "Packet Length Mean": 400, "Packet Length Std": 200, "Packet Length Variance": 40000,
        "FIN Flag Count": 1, "SYN Flag Count": 1, "RST Flag Count": 0,
        "PSH Flag Count": 1, "ACK Flag Count": 1,
        "Average Packet Size": 400, "Avg Fwd Segment Size": 300, "Avg Bwd Segment Size": 280,
        "Init_Win_bytes_forward": 65535, "Init_Win_bytes_backward": 65535,
        "Active Mean": 100000, "Idle Mean": 100000,
    }
    base.update(overrides)
    return base


def _dos_flow(**overrides):
    base = _benign_flow()
    base.update({
        "Flow Duration": 5000, "Total Fwd Packets": 1500,
        "Flow Bytes/s": 3000000, "Flow Packets/s": 15000,
        "Flow IAT Mean": 200, "Flow IAT Std": 100, "Flow IAT Max": 1000,
        "Fwd Packets/s": 12000, "Bwd Packets/s": 2000,
        "Packet Length Mean": 105, "Packet Length Std": 20, "Packet Length Variance": 400,
        "SYN Flag Count": 1, "ACK Flag Count": 0,
    })
    base.update(overrides)
    return base


def _portscan_flow(**overrides):
    base = {f: 0.0 for f in FEATURES}
    base.update({
        "Flow Duration": 100, "Total Fwd Packets": 1,
        "Flow Bytes/s": 400, "Flow Packets/s": 10000,
        "Fwd Packets/s": 10000, "Packet Length Mean": 40,
        "SYN Flag Count": 1, "RST Flag Count": 1,
        "Fwd Packet Length Mean": 40, "Average Packet Size": 40,
        "Init_Win_bytes_forward": 1024,
    })
    base.update(overrides)
    return base


def _bot_flow(**overrides):
    # Bot C2 signature: ultra-long duration, ultra-regular IAT (Std=807µs = clockwork beacon)
    base = _benign_flow()
    base.update({
        "Flow Duration": 80175886, "Total Fwd Packets": 3, "Total Backward Packets": 3,
        "Total Length of Fwd Packets": 2205, "Total Length of Bwd Packets": 1792,
        "Fwd Packet Length Mean": 75, "Bwd Packet Length Mean": 283,
        "Flow Bytes/s": 302, "Flow Packets/s": 0.3,
        "Flow IAT Mean": 15004141, "Flow IAT Std": 807,  # 807µs std = machine-regular
        "Flow IAT Max": 790968,
        "Fwd IAT Mean": 199671, "Bwd IAT Mean": 200504,
        "Fwd Packets/s": 30, "Bwd Packets/s": 25,
        "Packet Length Mean": 401, "Packet Length Std": 202,
        "Init_Win_bytes_forward": 508, "Init_Win_bytes_backward": 31751,
        "Active Mean": 2998458, "Idle Mean": 11974594,
    })
    base.update(overrides)
    return base


# ══════════════════════════════════════════════════════════════════════════════
# Test Cases
# ══════════════════════════════════════════════════════════════════════════════

class TestModelLoad(unittest.TestCase):
    """Verify model artifacts exist and load correctly."""

    def test_model_files_exist(self):
        hdr("Model File Existence")
        for path, name in [(IF_PATH,"isolation_forest.pkl"),
                           (RF_PATH,"random_forest.pkl"),
                           (SCALER_PATH,"scaler.pkl"),
                           (META_PATH,"meta.json")]:
            with self.subTest(file=name):
                self.assertTrue(path.exists(), f"Missing: {path}. Run: python model.py --train")
                print(f"  {GREEN}✓{RESET} {name} found ({path.stat().st_size//1024} KB)")

    def test_detector_loads(self):
        hdr("Detector Singleton Load")
        detector = get_detector()
        self.assertIsNotNone(detector.iforest)
        self.assertIsNotNone(detector.rf)
        self.assertIsNotNone(detector.scaler)
        self.assertIsNotNone(detector.meta)
        print(f"  {GREEN}✓{RESET} Detector loaded successfully")
        print(f"  {GREEN}✓{RESET} Features: {detector.meta.get('n_features')}")
        print(f"  {GREEN}✓{RESET} Trained at: {detector.meta.get('trained_at')}")

    def test_meta_fields(self):
        hdr("Model Metadata Validation")
        with open(META_PATH) as f:
            meta = json.load(f)
        required = ["features","n_features","if_accuracy","if_f1","rf_accuracy",
                    "rf_f1","attack_labels","feature_importances","trained_at"]
        for field in required:
            self.assertIn(field, meta, f"Missing meta field: {field}")
            print(f"  {GREEN}✓{RESET} meta['{field}'] = {str(meta[field])[:60]}")


class TestIsolationForest(unittest.TestCase):
    """Unit tests for Isolation Forest detection logic."""

    @classmethod
    def setUpClass(cls):
        cls.detector = get_detector()

    def test_benign_low_score(self):
        hdr("Isolation Forest — Benign Traffic")
        flow = _benign_flow()
        result = self.detector.predict_one(flow)
        print(f"  threat_score = {result['threat_score']} | severity = {result['severity']}")
        self.assertLessEqual(result["threat_score"], 55,
            f"Benign flow got high threat score: {result['threat_score']}")
        print(f"  {GREEN}✓{RESET} Benign flow correctly scored low")

    def test_dos_detected(self):
        hdr("Isolation Forest — DoS Attack Detection")
        flow = _dos_flow()
        result = self.detector.predict_one(flow)
        print(f"  threat_score = {result['threat_score']} | severity = {result['severity']}")
        print(f"  is_anomaly   = {result['is_anomaly']}")
        self.assertTrue(result["is_anomaly"] or result["threat_score"] >= 40,
            f"DoS flow not detected. Score: {result['threat_score']}")
        print(f"  {GREEN}✓{RESET} DoS attack detected")

    def test_portscan_detected(self):
        hdr("Isolation Forest — Port Scan Detection")
        flow = _portscan_flow()
        result = self.detector.predict_one(flow)
        print(f"  threat_score = {result['threat_score']} | severity = {result['severity']}")
        self.assertGreaterEqual(result["threat_score"], 30,
            f"Port scan not detected. Score: {result['threat_score']}")
        print(f"  {GREEN}✓{RESET} Port scan flagged")

    def test_bot_c2_detected(self):
        hdr("Isolation Forest — Bot/C2 Beaconing")
        flow = _bot_flow()
        result = self.detector.predict_one(flow)
        print(f"  threat_score = {result['threat_score']} | severity = {result['severity']}")
        self.assertTrue(result["is_anomaly"] or result["threat_score"] >= 35,
            f"Bot C2 not detected. Score: {result['threat_score']}")
        print(f"  {GREEN}✓{RESET} Bot/C2 beaconing flagged")

    def test_zero_threat_score_range(self):
        hdr("Isolation Forest — Score Range [0,100]")
        for i in range(20):
            flow = _benign_flow(
                **{k: float(np.random.exponential(500)) for k in
                   ["Flow Bytes/s","Flow Packets/s","Flow IAT Mean"]}
            )
            result = self.detector.predict_one(flow)
            self.assertGreaterEqual(result["threat_score"], 0)
            self.assertLessEqual(result["threat_score"], 100)
        print(f"  {GREEN}✓{RESET} All 20 random flows scored within [0, 100]")

    def test_severity_labels(self):
        hdr("Isolation Forest — Severity Label Consistency")
        flows_scores = [
            (_benign_flow(), "should be low"),
            (_dos_flow(), "should be high"),
        ]
        valid_labels = {"CLEAN","LOW","MEDIUM","HIGH","CRITICAL"}
        for flow, desc in flows_scores:
            result = self.detector.predict_one(flow)
            self.assertIn(result["severity"], valid_labels,
                f"Invalid severity: {result['severity']}")
            print(f"  {GREEN}✓{RESET} {desc}: score={result['threat_score']} sev={result['severity']}")


class TestXAIFeatureAttribution(unittest.TestCase):
    """Verify SHAP-style feature attribution is returned correctly."""

    @classmethod
    def setUpClass(cls):
        cls.detector = get_detector()

    def test_top_features_returned(self):
        hdr("XAI — Top Feature Attribution")
        result = self.detector.predict_one(_dos_flow())
        self.assertIn("top_features", result)
        self.assertGreater(len(result["top_features"]), 0)
        self.assertLessEqual(len(result["top_features"]), 10)
        for feat in result["top_features"]:
            self.assertIn("feature", feat)
            self.assertIn("contribution", feat)
            self.assertIn("value", feat)
        top = result["top_features"][0]
        print(f"  {GREEN}✓{RESET} Top feature: '{top['feature']}' contribution={top['contribution']:.4f}")
        print(f"  {GREEN}✓{RESET} {len(result['top_features'])} features returned")

    def test_feature_names_valid(self):
        hdr("XAI — Feature Names Are Valid CICIDS Features")
        result = self.detector.predict_one(_dos_flow())
        for feat in result["top_features"]:
            self.assertIn(feat["feature"], FEATURES,
                f"Unknown feature: {feat['feature']}")
        print(f"  {GREEN}✓{RESET} All returned feature names match CICIDS2017 schema")

    def test_contributions_non_negative(self):
        hdr("XAI — Contributions Are Non-Negative")
        result = self.detector.predict_one(_benign_flow())
        for feat in result["top_features"]:
            self.assertGreaterEqual(feat["contribution"], 0,
                f"Negative contribution for {feat['feature']}")
        print(f"  {GREEN}✓{RESET} All feature contributions ≥ 0")

    def test_high_attack_feature_is_dominant(self):
        hdr("XAI — DoS Flow Has Traffic-Rate Features on Top")
        result = self.detector.predict_one(_dos_flow())
        top5_names = [f["feature"] for f in result["top_features"][:5]]
        rate_features = {"Flow Packets/s","Flow Bytes/s","Fwd Packets/s",
                         "Bwd Packets/s","Flow Duration","Packet Length Mean"}
        overlap = set(top5_names) & rate_features
        print(f"  Top-5 features: {top5_names}")
        print(f"  Rate-feature overlap: {overlap}")
        self.assertGreater(len(overlap), 0,
            "Expected at least one rate-related feature in top-5 for DoS flow")
        print(f"  {GREEN}✓{RESET} Traffic-rate features dominate DoS explanation")


class TestBatchPrediction(unittest.TestCase):
    """Test batch inference logic."""

    @classmethod
    def setUpClass(cls):
        cls.detector = get_detector()

    def test_batch_returns_correct_count(self):
        hdr("Batch Prediction — Result Count")
        flows = [_benign_flow() for _ in range(5)] + [_dos_flow() for _ in range(5)]
        results = self.detector.predict_batch(flows)
        self.assertEqual(len(results), 10)
        print(f"  {GREEN}✓{RESET} 10 flows in → 10 results out")

    def test_batch_detects_attacks_in_mix(self):
        hdr("Batch Prediction — Mixed Traffic")
        benign_flows = [_benign_flow() for _ in range(8)]
        attack_flows = [_dos_flow() for _ in range(4)] + [_portscan_flow() for _ in range(4)]
        results = self.detector.predict_batch(benign_flows + attack_flows)
        attack_results = results[8:]
        n_detected = sum(1 for r in attack_results if r["is_anomaly"] or r["threat_score"] >= 40)
        pct = n_detected / len(attack_results) * 100
        print(f"  Attacks detected: {n_detected}/{len(attack_results)} ({pct:.0f}%)")
        self.assertGreaterEqual(n_detected, 4,
            f"Batch only detected {n_detected}/8 attacks")
        print(f"  {GREEN}✓{RESET} Batch attack detection ≥ 50%")

    def test_batch_latency(self):
        hdr("Batch Prediction — Latency")
        flows = [_benign_flow() for _ in range(50)]
        t0 = time.time()
        self.detector.predict_batch(flows)
        elapsed_ms = (time.time() - t0) * 1000
        per_flow = elapsed_ms / 50
        print(f"  50 flows in {elapsed_ms:.1f}ms ({per_flow:.2f}ms/flow)")
        self.assertLess(per_flow, 500, f"Per-flow latency too high: {per_flow:.2f}ms")
        print(f"  {GREEN}✓{RESET} Per-flow latency < 500ms (real hardware typically <5ms)")


class TestDataFrame(unittest.TestCase):
    """Test DataFrame-based prediction (used for CSV upload)."""

    @classmethod
    def setUpClass(cls):
        cls.detector = get_detector()

    def test_df_prediction_adds_columns(self):
        hdr("DataFrame Prediction — Output Columns")
        flows = [_benign_flow(), _dos_flow(), _portscan_flow()]
        df = pd.DataFrame(flows)
        out = self.detector.predict_df(df)
        for col in ["threat_score","is_anomaly","attack_label","severity","if_raw"]:
            self.assertIn(col, out.columns, f"Missing column: {col}")
        print(f"  {GREEN}✓{RESET} Output columns: {['threat_score','is_anomaly','attack_label','severity','if_raw']}")

    def test_df_all_rows_have_scores(self):
        hdr("DataFrame Prediction — All Rows Scored")
        flows = [_benign_flow(), _dos_flow(), _bot_flow(), _portscan_flow()]
        df = pd.DataFrame(flows)
        out = self.detector.predict_df(df)
        self.assertEqual(len(out), 4)
        self.assertFalse(out["threat_score"].isna().any())
        print(f"  {GREEN}✓{RESET} All 4 rows have non-null threat scores")


class TestEdgeCases(unittest.TestCase):
    """Edge cases — zero flows, missing features, extreme values."""

    @classmethod
    def setUpClass(cls):
        cls.detector = get_detector()

    def test_all_zeros(self):
        hdr("Edge Case — All-Zero Flow")
        flow = {f: 0.0 for f in FEATURES}
        result = self.detector.predict_one(flow)
        self.assertIn("threat_score", result)
        self.assertGreaterEqual(result["threat_score"], 0)
        self.assertLessEqual(result["threat_score"], 100)
        print(f"  {GREEN}✓{RESET} All-zero flow: score={result['threat_score']}")

    def test_missing_keys(self):
        hdr("Edge Case — Missing Feature Keys")
        flow = {"Flow Duration": 500000, "Total Fwd Packets": 18}  # only 2 features
        result = self.detector.predict_one(flow)
        self.assertIn("threat_score", result)
        print(f"  {GREEN}✓{RESET} Missing keys handled: score={result['threat_score']}")

    def test_extreme_values(self):
        hdr("Edge Case — Extreme Feature Values")
        flow = _benign_flow(**{
            "Flow Bytes/s": 1e12,
            "Flow Packets/s": 1e9,
            "Flow Duration": 1e10,
        })
        result = self.detector.predict_one(flow)
        self.assertLessEqual(result["threat_score"], 100)
        self.assertGreaterEqual(result["threat_score"], 0)
        print(f"  {GREEN}✓{RESET} Extreme values clamped: score={result['threat_score']}")

    def test_empty_batch(self):
        hdr("Edge Case — Empty Batch")
        result = self.detector.predict_batch([])
        self.assertEqual(result, [])
        print(f"  {GREEN}✓{RESET} Empty batch returns empty list")


class TestModelMetrics(unittest.TestCase):
    """Verify trained model meets minimum performance thresholds."""

    def test_if_accuracy_threshold(self):
        hdr("Model Metrics — Isolation Forest Accuracy ≥ 70%")
        with open(META_PATH) as f:
            meta = json.load(f)
        acc = meta["if_accuracy"]
        print(f"  IF Accuracy : {acc*100:.2f}%")
        print(f"  IF F1       : {meta['if_f1']*100:.2f}%")
        print(f"  IF Precision: {meta['if_precision']*100:.2f}%")
        print(f"  IF Recall   : {meta['if_recall']*100:.2f}%")
        self.assertGreaterEqual(acc, 0.70, f"IF accuracy too low: {acc:.2f}")
        print(f"  {GREEN}✓{RESET} Isolation Forest accuracy above 70% threshold")

    def test_rf_accuracy_threshold(self):
        hdr("Model Metrics — Random Forest Accuracy ≥ 80%")
        with open(META_PATH) as f:
            meta = json.load(f)
        acc = meta["rf_accuracy"]
        print(f"  RF Accuracy : {acc*100:.2f}%")
        print(f"  RF F1       : {meta['rf_f1']*100:.2f}%")
        self.assertGreaterEqual(acc, 0.80, f"RF accuracy too low: {acc:.2f}")
        print(f"  {GREEN}✓{RESET} Random Forest accuracy above 80% threshold")

    def test_feature_importances_sum(self):
        hdr("Model Metrics — Feature Importances Sum ≈ 1.0")
        with open(META_PATH) as f:
            meta = json.load(f)
        total = sum(meta["feature_importances"].values())
        print(f"  Feature importance sum: {total:.6f}")
        self.assertAlmostEqual(total, 1.0, delta=0.01,
            msg=f"Feature importances don't sum to 1: {total:.4f}")
        top3 = sorted(meta["feature_importances"].items(), key=lambda x:-x[1])[:3]
        for name, imp in top3:
            print(f"  Top feature: '{name}' = {imp:.4f}")
        print(f"  {GREEN}✓{RESET} Feature importances valid")


# ══════════════════════════════════════════════════════════════════════════════
# Runner
# ══════════════════════════════════════════════════════════════════════════════

def run_all():
    print(f"\n{BOLD}{'═'*60}{RESET}")
    print(f"{BOLD}  ZeroSight — Full Test Suite{RESET}")
    print(f"{BOLD}{'═'*60}{RESET}")

    loader  = unittest.TestLoader()
    suite   = unittest.TestSuite()
    classes = [
        TestModelLoad,
        TestIsolationForest,
        TestXAIFeatureAttribution,
        TestBatchPrediction,
        TestDataFrame,
        TestEdgeCases,
        TestModelMetrics,
    ]
    for cls in classes:
        suite.addTests(loader.loadTestsFromTestCase(cls))

    runner = unittest.TextTestRunner(verbosity=0, stream=sys.stdout)
    result = runner.run(suite)

    total  = result.testsRun
    failed = len(result.failures) + len(result.errors)
    passed = total - failed

    print(f"\n{BOLD}{'═'*60}{RESET}")
    print(f"{BOLD}  Results: {GREEN}{passed} passed{RESET}{BOLD}  {RED}{failed} failed{RESET}{BOLD}  / {total} total{RESET}")
    print(f"{BOLD}{'═'*60}{RESET}\n")

    if result.failures:
        print(f"{RED}FAILURES:{RESET}")
        for test, tb in result.failures:
            print(f"  {RED}✗ {test}{RESET}\n{tb}")
    if result.errors:
        print(f"{RED}ERRORS:{RESET}")
        for test, tb in result.errors:
            print(f"  {RED}✗ {test}{RESET}\n{tb}")

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(run_all())