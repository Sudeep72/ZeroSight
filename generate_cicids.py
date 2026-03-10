"""ZeroSight — Synthetic CICIDS2017 Generator v4
Each attack type has a FULLY distinct 31-feature signature.
No 'inherit BENIGN then override' — every feature is set independently.
"""
import os, numpy as np, pandas as pd
SEED = 42; np.random.seed(SEED)

FEATURES = [
    "Flow Duration","Total Fwd Packets","Total Backward Packets",
    "Total Length of Fwd Packets","Total Length of Bwd Packets",
    "Fwd Packet Length Mean","Bwd Packet Length Mean",
    "Flow Bytes/s","Flow Packets/s",
    "Flow IAT Mean","Flow IAT Std","Flow IAT Max",
    "Fwd IAT Mean","Bwd IAT Mean","Fwd Packets/s","Bwd Packets/s",
    "Packet Length Mean","Packet Length Std","Packet Length Variance",
    "FIN Flag Count","SYN Flag Count","RST Flag Count",
    "PSH Flag Count","ACK Flag Count",
    "Average Packet Size","Avg Fwd Segment Size","Avg Bwd Segment Size",
    "Init_Win_bytes_forward","Init_Win_bytes_backward",
    "Active Mean","Idle Mean",
]
N = len(FEATURES)

def R(n): return np.random.RandomState(n)

def sample(rng, mu, sigma, n, lo=None, hi=None):
    v = rng.normal(mu, sigma, n)
    if lo is not None: v = np.clip(v, lo, None)
    if hi is not None: v = np.clip(v, None, hi)
    return v

def make(n, specs):
    """specs: dict feature→(mu, sigma[, lo[, hi]])"""
    rng = np.random.RandomState(SEED + hash(str(specs)) % 10000)
    out = {}
    for f, args in specs.items():
        mu, sigma = args[0], args[1]
        lo = args[2] if len(args)>2 else None
        hi = args[3] if len(args)>3 else None
        out[f] = sample(rng, mu, sigma, n, lo, hi)
    return out

# ── BENIGN ──────────────────────────────────────────────────────────
# Wide realistic distributions covering:
# - Short bursts (HTTP/DNS) to long sessions (SSH/streaming) 
# - Small (ACK=40B) to large (bulk=1460B) packets
# - Low-rate to medium-rate flows
def BENIGN(n):
    rng_b = np.random.RandomState(SEED+1)
    # Mix 3 traffic types: web browsing, bulk transfer, interactive
    n_web  = n // 2        # short flows, medium packets
    n_bulk = n // 4        # long flows, large packets
    n_int  = n - n_web - n_bulk  # interactive: small packets, variable rate

    def web(m):
        return make(m, {
            "Flow Duration":          (5_000_000, 3_000_000, 100_000, 30_000_000),
            "Total Fwd Packets":      (15, 8, 2, 80),
            "Total Backward Packets": (12, 6, 1, 60),
            "Total Length of Fwd Packets": (8_000, 4_000, 100, 60_000),
            "Total Length of Bwd Packets": (15_000, 8_000, 100, 80_000),
            "Fwd Packet Length Mean": (500, 200, 40, 1_460),
            "Bwd Packet Length Mean": (800, 250, 40, 1_460),
            "Flow Bytes/s":           (18_000, 10_000, 200, 100_000),
            "Flow Packets/s":         (20, 12, 1, 100),
            "Flow IAT Mean":          (300_000, 150_000, 10_000, 2_000_000),
            "Flow IAT Std":           (200_000, 100_000, 5_000, 1_500_000),
            "Flow IAT Max":           (1_500_000, 800_000, 50_000, 8_000_000),
            "Fwd IAT Mean":           (400_000, 200_000, 5_000, 3_000_000),
            "Bwd IAT Mean":           (400_000, 200_000, 5_000, 3_000_000),
            "Fwd Packets/s":          (10, 6, 0.5, 60),
            "Bwd Packets/s":          (10, 6, 0.5, 60),
            "Packet Length Mean":     (650, 250, 40, 1_400),
            "Packet Length Std":      (250, 100, 5, 600),
            "Packet Length Variance": (80_000, 50_000, 25, 360_000),
            "Average Packet Size":    (650, 250, 40, 1_400),
            "Avg Fwd Segment Size":   (500, 200, 40, 1_460),
            "Avg Bwd Segment Size":   (800, 250, 40, 1_460),
            "Init_Win_bytes_forward":  (56_000, 10_000, 8_000, 65_535),
            "Init_Win_bytes_backward": (60_000, 8_000, 8_000, 65_535),
            "Active Mean":            (500_000, 400_000, 10_000, 3_000_000),
            "Idle Mean":              (2_000_000, 1_500_000, 100_000, 10_000_000),
        })

    def bulk(m):
        return make(m, {
            "Flow Duration":          (30_000_000, 15_000_000, 5_000_000, 120_000_000),
            "Total Fwd Packets":      (200, 100, 20, 800),
            "Total Backward Packets": (180, 90, 15, 700),
            "Total Length of Fwd Packets": (280_000, 100_000, 10_000, 1_000_000),
            "Total Length of Bwd Packets": (250_000, 100_000, 10_000, 1_000_000),
            "Fwd Packet Length Mean": (1_200, 150, 400, 1_460),
            "Bwd Packet Length Mean": (1_100, 150, 400, 1_460),
            "Flow Bytes/s":           (2_000_000, 1_500_000, 2_000, 15_000_000),  # streaming/bulk
            "Flow Packets/s":         (500, 400, 2, 2500),  # burst download
            "Flow IAT Mean":          (600_000, 300_000, 20_000, 3_000_000),
            "Flow IAT Std":           (400_000, 200_000, 10_000, 2_000_000),
            "Flow IAT Max":           (3_000_000, 1_500_000, 200_000, 15_000_000),
            "Fwd IAT Mean":           (600_000, 300_000, 10_000, 3_000_000),
            "Bwd IAT Mean":           (600_000, 300_000, 10_000, 3_000_000),
            "Fwd Packets/s":          (30, 20, 1, 200),  # client→server
            "Bwd Packets/s":          (400, 300, 5, 2000),  # server→client download bursts
            "Packet Length Mean":     (1_150, 150, 400, 1_400),
            "Packet Length Std":      (400, 150, 10, 650),  # wide spread ACK+data
            "Packet Length Variance": (150_000, 100_000, 100, 450_000),  # mixed ACK+data
            "Average Packet Size":    (1_150, 150, 400, 1_400),
            "Avg Fwd Segment Size":   (1_200, 150, 400, 1_460),
            "Avg Bwd Segment Size":   (1_100, 150, 400, 1_460),
            "Init_Win_bytes_forward":  (60_000, 8_000, 16_000, 65_535),
            "Init_Win_bytes_backward": (60_000, 8_000, 16_000, 65_535),
            "Active Mean":            (2_000_000, 1_000_000, 100_000, 8_000_000),
            "Idle Mean":              (5_000_000, 2_000_000, 200_000, 20_000_000),
        })

    def interactive(m):
        return make(m, {
            "Flow Duration":          (2_000_000, 1_000_000, 100_000, 10_000_000),
            "Total Fwd Packets":      (8, 4, 2, 30),
            "Total Backward Packets": (6, 3, 1, 25),
            "Total Length of Fwd Packets": (500, 300, 40, 3_000),
            "Total Length of Bwd Packets": (400, 250, 40, 2_500),
            "Fwd Packet Length Mean": (80, 40, 20, 300),
            "Bwd Packet Length Mean": (70, 35, 20, 250),
            "Flow Bytes/s":           (1_500, 1_000, 50, 8_000),
            "Flow Packets/s":         (8, 4, 0.5, 30),
            "Flow IAT Mean":          (200_000, 100_000, 5_000, 1_000_000),
            "Flow IAT Std":           (150_000, 80_000, 2_000, 800_000),
            "Flow IAT Max":           (800_000, 400_000, 30_000, 4_000_000),
            "Fwd IAT Mean":           (250_000, 120_000, 5_000, 1_500_000),
            "Bwd IAT Mean":           (250_000, 120_000, 5_000, 1_500_000),
            "Fwd Packets/s":          (4, 2, 0.3, 20),
            "Bwd Packets/s":          (4, 2, 0.3, 20),
            "Packet Length Mean":     (75, 40, 20, 280),
            "Packet Length Std":      (30, 15, 2, 120),
            "Packet Length Variance": (1_000, 800, 4, 14_400),
            "Average Packet Size":    (75, 40, 20, 280),
            "Avg Fwd Segment Size":   (80, 40, 20, 300),
            "Avg Bwd Segment Size":   (70, 35, 20, 250),
            "Init_Win_bytes_forward":  (32_000, 16_000, 1_024, 65_535),
            "Init_Win_bytes_backward": (32_000, 16_000, 1_024, 65_535),
            "Active Mean":            (150_000, 100_000, 5_000, 1_000_000),
            "Idle Mean":              (800_000, 600_000, 50_000, 5_000_000),
        })

    def udp_dns(m):
        """Short UDP flows: DNS queries, NTP — no ACK/SYN/WIN flags, small fast bursts"""
        return make(m, {
            "Flow Duration":          (50_000, 80_000, 1_000, 500_000),
            "Total Fwd Packets":      (2, 1, 1, 6),
            "Total Backward Packets": (2, 1, 0, 6),
            "Total Length of Fwd Packets": (200, 150, 40, 1_200),
            "Total Length of Bwd Packets": (400, 300, 0, 2_400),
            "Fwd Packet Length Mean": (100, 80, 40, 400),
            "Bwd Packet Length Mean": (200, 150, 0, 600),
            "Flow Bytes/s":           (15_000, 20_000, 500, 120_000),
            "Flow Packets/s":         (80, 60, 5, 400),
            "Flow IAT Mean":          (15_000, 20_000, 500, 150_000),
            "Flow IAT Std":           (10_000, 15_000, 100, 100_000),
            "Flow IAT Max":           (40_000, 60_000, 1_000, 400_000),
            "Fwd IAT Mean":           (30_000, 40_000, 1_000, 300_000),
            "Bwd IAT Mean":           (30_000, 40_000, 1_000, 300_000),
            "Fwd Packets/s":          (40, 30, 2, 200),
            "Bwd Packets/s":          (40, 30, 0, 200),
            "Packet Length Mean":     (150, 100, 30, 500),
            "Packet Length Std":      (80, 60, 5, 300),
            "Packet Length Variance": (6_400, 8_000, 25, 90_000),
            "Average Packet Size":    (150, 100, 30, 500),
            "Avg Fwd Segment Size":   (100, 80, 40, 400),
            "Avg Bwd Segment Size":   (200, 150, 0, 600),
            "Init_Win_bytes_forward":  (0, 0, 0, 0),   # UDP: no TCP window
            "Init_Win_bytes_backward": (0, 0, 0, 0),
            "Active Mean":            (15_000, 20_000, 500, 150_000),
            "Idle Mean":              (35_000, 60_000, 500, 350_000),
        })

    # merge four traffic types: web, bulk, interactive, UDP/DNS
    n_udp  = max(n // 8, 1)
    n_bulk2 = max(n_bulk - n_udp, 1)
    w = web(n_web); b2 = bulk(n_bulk2); i = interactive(n_int); u = udp_dns(n_udp)
    s = {}
    # lengths may be slightly off due to integer division — concat and trim to exactly n
    for f in w.keys():
        arr = np.concatenate([w[f], b2[f], i[f], u[f]])
        s[f] = arr[:n] if len(arr) >= n else np.concatenate([arr, arr[:n-len(arr)]])
    n_actual = len(s[list(s.keys())[0]])
    # shuffle so types are mixed
    idx = np.random.RandomState(SEED+99).permutation(n_actual)
    s = {f: v[idx] for f, v in s.items()}
    # Flag counts = realistic per-flow packet counts, not binary
    total_pkts = s["Total Fwd Packets"] + s["Total Backward Packets"]
    nn = len(total_pkts)
    s["ACK Flag Count"] = np.clip(rng_b.normal(total_pkts * 0.82, total_pkts * 0.08 + 0.1, nn), 0, total_pkts).astype(float)
    s["PSH Flag Count"] = np.clip(rng_b.normal(total_pkts * 0.14, total_pkts * 0.05 + 0.1, nn), 0, total_pkts * 0.4).astype(float)
    s["SYN Flag Count"] = rng_b.binomial(1, 0.85, nn).astype(float)
    s["FIN Flag Count"] = rng_b.binomial(1, 0.75, nn).astype(float)
    s["RST Flag Count"] = rng_b.binomial(1, 0.04, nn).astype(float)
    # UDP flows have no TCP flags or window sizes — zero them out
    # (they end up at shuffled positions; zero by Init_Win=0 marker)
    udp_mask = (s["Init_Win_bytes_forward"] == 0) & (s["Init_Win_bytes_backward"] == 0)
    for flag in ["ACK Flag Count","PSH Flag Count","SYN Flag Count","FIN Flag Count","RST Flag Count"]:
        s[flag][udp_mask] = 0
    return s

# ── DoS Hulk ─────────────────────────────────────────────────────────
# Ultra-high rate flood: short flows, massive pkt/s, tiny IAT, large fwd pkts
def DOS_HULK(n):
    s = make(n, {
        "Flow Duration":          (3_000, 500, 500, 8_000),
        "Total Fwd Packets":      (2_500, 300, 1_000),
        "Total Backward Packets": (15, 5, 2, 40),
        "Total Length of Fwd Packets": (1_500_000, 200_000, 400_000),
        "Total Length of Bwd Packets": (1_800, 300, 100),
        "Fwd Packet Length Mean": (600, 40, 200, 1_400),
        "Bwd Packet Length Mean": (280, 50, 50, 800),
        "Flow Bytes/s":           (15_000_000, 2_000_000, 3_000_000),
        "Flow Packets/s":         (800_000, 80_000, 200_000),
        "Flow IAT Mean":          (200, 50, 20, 800),
        "Flow IAT Std":           (100, 20, 5, 400),
        "Flow IAT Max":           (1_000, 200, 100, 5_000),
        "Fwd IAT Mean":           (200_000, 35_000, 5_000),
        "Bwd IAT Mean":           (200_000, 35_000, 5_000),
        "Fwd Packets/s":          (600_000, 60_000, 150_000),
        "Bwd Packets/s":          (25, 5, 2, 80),
        "Packet Length Mean":     (400, 60, 50, 900),
        "Packet Length Std":      (200, 30, 10),
        "Packet Length Variance": (40_000, 8_000, 100),
        "Average Packet Size":    (400, 60, 50, 900),
        "Avg Fwd Segment Size":   (600, 40, 200),
        "Avg Bwd Segment Size":   (280, 50, 50),
        "Init_Win_bytes_forward":  (100, 50, 0, 300),
        "Init_Win_bytes_backward": (16_000, 8_000, 0),
        "Active Mean":            (1_000, 200, 100),
        "Idle Mean":              (500, 100, 50),
    })
    rng = np.random.RandomState(SEED+2)
    s["FIN Flag Count"] = np.zeros(n)
    s["SYN Flag Count"] = np.ones(n)
    s["RST Flag Count"] = np.zeros(n)
    s["PSH Flag Count"] = rng.binomial(1, 0.3, n).astype(float)
    s["ACK Flag Count"] = np.zeros(n)
    return s

# ── PortScan ──────────────────────────────────────────────────────────
# 1 fwd packet, 0 backward, tiny size, SYN+RST, ultra-short duration
def PORTSCAN(n):
    s = make(n, {
        "Flow Duration":          (150, 30, 20, 500),
        "Total Fwd Packets":      (1, 0.1, 1, 1),
        "Total Backward Packets": (0, 0, 0, 0),
        "Total Length of Fwd Packets": (40, 4, 20, 60),
        "Total Length of Bwd Packets": (0, 0, 0, 0),
        "Fwd Packet Length Mean": (40, 4, 20, 60),
        "Bwd Packet Length Mean": (0, 0, 0, 0),
        "Flow Bytes/s":           (250_000, 30_000, 80_000),
        "Flow Packets/s":         (6_500, 500, 2_000),
        "Flow IAT Mean":          (200, 40, 20, 800),
        "Flow IAT Std":           (50, 10, 1, 200),
        "Flow IAT Max":           (200, 40, 20, 800),
        "Fwd IAT Mean":           (200_000, 35_000, 5_000),
        "Bwd IAT Mean":           (200_000, 35_000, 5_000),
        "Fwd Packets/s":          (6_000, 500, 2_000),
        "Bwd Packets/s":          (0, 0, 0, 0),
        "Packet Length Mean":     (40, 4, 20, 60),
        "Packet Length Std":      (5, 1, 0, 20),
        "Packet Length Variance": (25, 5, 0, 100),
        "Average Packet Size":    (400, 60, 50),
        "Avg Fwd Segment Size":   (300, 50, 50),
        "Avg Bwd Segment Size":   (280, 50, 50),
        "Init_Win_bytes_forward":  (100, 50, 0, 300),
        "Init_Win_bytes_backward": (32_000, 16_000, 0),
        "Active Mean":            (500, 100, 50),
        "Idle Mean":              (1_000, 200, 100),
    })
    s["FIN Flag Count"] = np.zeros(n)
    s["SYN Flag Count"] = np.ones(n)
    s["RST Flag Count"] = np.random.binomial(1, 0.85, n).astype(float)
    s["PSH Flag Count"] = np.zeros(n)
    s["ACK Flag Count"] = np.zeros(n)
    return s

# ── DDoS ──────────────────────────────────────────────────────────────
# Many tiny packets, massive rate, very short IAT, 0 backward bytes
def DDOS(n):
    s = make(n, {
        "Flow Duration":          (5_000, 800, 800, 15_000),
        "Total Fwd Packets":      (3_000, 400, 1_000),
        "Total Backward Packets": (14, 3, 2, 30),
        "Total Length of Fwd Packets": (2_200, 400, 100),
        "Total Length of Bwd Packets": (1_800, 300, 100),
        "Fwd Packet Length Mean": (25, 3, 18, 40),
        "Bwd Packet Length Mean": (0, 0, 0, 0),
        "Flow Bytes/s":           (20_000_000, 2_000_000, 5_000_000),
        "Flow Packets/s":         (600_000, 60_000, 150_000),
        "Flow IAT Mean":          (150, 30, 10, 600),
        "Flow IAT Std":           (50, 10, 1, 200),
        "Flow IAT Max":           (800_000, 100_000, 50_000),
        "Fwd IAT Mean":           (200_000, 35_000, 5_000),
        "Bwd IAT Mean":           (200_000, 35_000, 5_000),
        "Fwd Packets/s":          (30, 5, 2),
        "Bwd Packets/s":          (25, 5, 2),
        "Packet Length Mean":     (25, 3, 18, 40),
        "Packet Length Std":      (3, 1, 0, 10),
        "Packet Length Variance": (9, 2, 0, 50),
        "Average Packet Size":    (400, 60, 50),
        "Avg Fwd Segment Size":   (300, 50, 50),
        "Avg Bwd Segment Size":   (280, 50, 50),
        "Init_Win_bytes_forward":  (100, 50, 0, 300),
        "Init_Win_bytes_backward": (16_000, 8_000, 0),
        "Active Mean":            (800, 150, 80),
        "Idle Mean":              (300, 60, 30),
    })
    s["FIN Flag Count"] = np.zeros(n)
    s["SYN Flag Count"] = np.ones(n)
    s["RST Flag Count"] = np.zeros(n)
    s["PSH Flag Count"] = np.ones(n)
    s["ACK Flag Count"] = np.zeros(n)
    return s

# ── Bot / C2 ──────────────────────────────────────────────────────────
# Very long flows, tiny packet rate, ultra-regular IAT (low Std), small window
def BOT(n):
    s = make(n, {
        "Flow Duration":          (80_000_000, 5_000_000, 20_000_000),
        "Total Fwd Packets":      (4, 0.5, 2, 8),
        "Total Backward Packets": (4, 0.5, 2, 8),
        "Total Length of Fwd Packets": (2_200, 400, 100),
        "Total Length of Bwd Packets": (1_800, 300, 100),
        "Fwd Packet Length Mean": (75, 8, 30, 130),
        "Bwd Packet Length Mean": (280, 50, 50, 800),
        "Flow Bytes/s":           (300, 40, 30, 1_000),
        "Flow Packets/s":         (0.3, 0.05, 0.05, 1),
        "Flow IAT Mean":          (15_000_000, 500_000, 3_000_000),
        "Flow IAT Std":           (800, 100, 50, 3_000),        # KEY: ultra-regular
        "Flow IAT Max":           (800_000, 100_000, 50_000),
        "Fwd IAT Mean":           (200_000, 35_000, 5_000),
        "Bwd IAT Mean":           (200_000, 35_000, 5_000),
        "Fwd Packets/s":          (30, 5, 2),
        "Bwd Packets/s":          (25, 5, 2),
        "Packet Length Mean":     (400, 60, 50),
        "Packet Length Std":      (200, 30, 10),
        "Packet Length Variance": (40_000, 8_000, 100),
        "Average Packet Size":    (400, 60, 50),
        "Avg Fwd Segment Size":   (300, 50, 50),
        "Avg Bwd Segment Size":   (280, 50, 50),
        "Init_Win_bytes_forward":  (512, 200, 128, 1_024),      # KEY: tiny window
        "Init_Win_bytes_backward": (32_000, 16_000, 0),
        "Active Mean":            (3_000_000, 300_000, 500_000),
        "Idle Mean":              (12_000_000, 1_000_000, 2_000_000),
    })
    s["FIN Flag Count"] = np.zeros(n)
    s["SYN Flag Count"] = np.zeros(n)
    s["RST Flag Count"] = np.zeros(n)
    s["PSH Flag Count"] = np.ones(n)
    s["ACK Flag Count"] = np.ones(n)
    return s

# ── DoS Slowloris ──────────────────────────────────────────────────────
# Very long flows, ultra-slow packet rate, tiny bytes/s, large IAT
def DOS_SLOWLORIS(n):
    s = make(n, {
        "Flow Duration":          (50_000_000, 5_000_000, 10_000_000),
        "Total Fwd Packets":      (15, 3, 4, 40),
        "Total Backward Packets": (14, 3, 2, 30),
        "Total Length of Fwd Packets": (2_200, 400, 100),
        "Total Length of Bwd Packets": (1_800, 300, 100),
        "Fwd Packet Length Mean": (180, 20, 50, 400),
        "Bwd Packet Length Mean": (280, 50, 50, 800),
        "Flow Bytes/s":           (80, 15, 5, 300),             # KEY: near-zero
        "Flow Packets/s":         (0.5, 0.1, 0.05, 2),         # KEY: near-zero
        "Flow IAT Mean":          (5_000_000, 500_000, 500_000),# KEY: huge
        "Flow IAT Std":           (200_000, 30_000, 10_000),
        "Flow IAT Max":           (800_000, 100_000, 50_000),
        "Fwd IAT Mean":           (200_000, 35_000, 5_000),
        "Bwd IAT Mean":           (200_000, 35_000, 5_000),
        "Fwd Packets/s":          (30, 5, 2),
        "Bwd Packets/s":          (25, 5, 2),
        "Packet Length Mean":     (400, 60, 50),
        "Packet Length Std":      (200, 30, 10),
        "Packet Length Variance": (40_000, 8_000, 100),
        "Average Packet Size":    (400, 60, 50),
        "Avg Fwd Segment Size":   (300, 50, 50),
        "Avg Bwd Segment Size":   (280, 50, 50),
        "Init_Win_bytes_forward":  (1_024, 512, 128, 2_048),
        "Init_Win_bytes_backward": (24_000, 8_000, 0),
        "Active Mean":            (2_000_000, 300_000, 200_000),
        "Idle Mean":              (8_000_000, 800_000, 1_000_000),
    })
    s["FIN Flag Count"] = np.zeros(n)
    s["SYN Flag Count"] = np.zeros(n)
    s["RST Flag Count"] = np.zeros(n)
    s["PSH Flag Count"] = np.ones(n)
    s["ACK Flag Count"] = np.ones(n)
    return s

# ── DoS Slowhttptest ──────────────────────────────────────────────────
def DOS_SLOWHTTP(n):
    s = DOS_SLOWLORIS(n)
    rng = np.random.RandomState(SEED+10)
    s["Flow Packets/s"]  = sample(rng, 1.2, 0.2, n, 0.2, 4)
    s["Flow Bytes/s"]    = sample(rng, 200, 30, n, 20, 800)
    s["Flow Duration"]   = sample(rng, 35_000_000, 4_000_000, n, 8_000_000)
    return s

# ── DoS GoldenEye ──────────────────────────────────────────────────────
# Medium rate, PSH+ACK HTTP keep-alive, very tight IAT
def DOS_GOLDENEYE(n):
    s = make(n, {
        "Flow Duration":          (40_000, 8_000, 5_000, 150_000),
        "Total Fwd Packets":      (600, 80, 100),
        "Total Backward Packets": (14, 3, 2, 30),
        "Total Length of Fwd Packets": (2_200, 400, 100),
        "Total Length of Bwd Packets": (1_800, 300, 100),
        "Fwd Packet Length Mean": (600, 50, 200, 1_400),
        "Bwd Packet Length Mean": (280, 50, 50, 800),
        "Flow Bytes/s":           (1_200_000, 150_000, 200_000),
        "Flow Packets/s":         (15_000, 2_000, 3_000),
        "Flow IAT Mean":          (2_500, 500, 200, 10_000),   # KEY: very tight
        "Flow IAT Std":           (1_200, 200, 50, 4_000),
        "Flow IAT Max":           (800_000, 100_000, 50_000),
        "Fwd IAT Mean":           (200_000, 35_000, 5_000),
        "Bwd IAT Mean":           (200_000, 35_000, 5_000),
        "Fwd Packets/s":          (30, 5, 2),
        "Bwd Packets/s":          (25, 5, 2),
        "Packet Length Mean":     (400, 60, 50),
        "Packet Length Std":      (200, 30, 10),
        "Packet Length Variance": (40_000, 8_000, 100),
        "Average Packet Size":    (400, 60, 50),
        "Avg Fwd Segment Size":   (300, 50, 50),
        "Avg Bwd Segment Size":   (280, 50, 50),
        "Init_Win_bytes_forward":  (32_000, 16_000, 0),
        "Init_Win_bytes_backward": (16_000, 8_000, 0),
        "Active Mean":            (100_000, 20_000, 10_000),
        "Idle Mean":              (500_000, 80_000, 50_000),
    })
    s["FIN Flag Count"] = np.zeros(n)
    s["SYN Flag Count"] = np.zeros(n)
    s["RST Flag Count"] = np.zeros(n)
    s["PSH Flag Count"] = np.ones(n)
    s["ACK Flag Count"] = np.ones(n)
    return s

# ── FTP-Patator ────────────────────────────────────────────────────────
# Slow brute-force: small fwd pkts, large IAT between attempts
def FTP_PATATOR(n):
    s = make(n, {
        "Flow Duration":          (4_000_000, 400_000, 800_000),
        "Total Fwd Packets":      (10, 2, 3, 25),
        "Total Backward Packets": (14, 3, 2, 30),
        "Total Length of Fwd Packets": (2_200, 400, 100),
        "Total Length of Bwd Packets": (1_800, 300, 100),
        "Fwd Packet Length Mean": (55, 6, 20, 100),             # KEY: tiny auth pkts
        "Bwd Packet Length Mean": (85, 8, 30, 150),
        "Flow Bytes/s":           (1_500, 200, 100, 5_000),
        "Flow Packets/s":         (3, 0.5, 0.5, 8),
        "Flow IAT Mean":          (220_000, 40_000, 10_000),
        "Flow IAT Std":           (180_000, 30_000, 5_000),
        "Flow IAT Max":           (800_000, 100_000, 50_000),
        "Fwd IAT Mean":           (3_000_000, 300_000, 300_000),# KEY: large fwd gap
        "Bwd IAT Mean":           (3_000_000, 300_000, 300_000),
        "Fwd Packets/s":          (30, 5, 2),
        "Bwd Packets/s":          (25, 5, 2),
        "Packet Length Mean":     (400, 60, 50),
        "Packet Length Std":      (200, 30, 10),
        "Packet Length Variance": (40_000, 8_000, 100),
        "Average Packet Size":    (400, 60, 50),
        "Avg Fwd Segment Size":   (300, 50, 50),
        "Avg Bwd Segment Size":   (280, 50, 50),
        "Init_Win_bytes_forward":  (32_000, 16_000, 0),
        "Init_Win_bytes_backward": (32_000, 16_000, 0),
        "Active Mean":            (100_000, 20_000, 10_000),
        "Idle Mean":              (500_000, 80_000, 50_000),
    })
    s["FIN Flag Count"] = np.zeros(n)
    s["SYN Flag Count"] = np.zeros(n)
    s["RST Flag Count"] = np.zeros(n)
    s["PSH Flag Count"] = np.ones(n)
    s["ACK Flag Count"] = np.ones(n)
    return s

# ── SSH-Patator ────────────────────────────────────────────────────────
def SSH_PATATOR(n):
    s = FTP_PATATOR(n)
    rng = np.random.RandomState(SEED+11)
    s["Fwd Packet Length Mean"]  = sample(rng, 75, 8, n, 30, 130)
    s["Avg Fwd Segment Size"]    = s["Fwd Packet Length Mean"].copy()
    s["Flow Duration"]           = sample(rng, 3_000_000, 300_000, n, 600_000)
    s["Flow Packets/s"]          = sample(rng, 4, 0.8, n, 1, 12)
    s["Flow Bytes/s"]            = sample(rng, 1_800, 250, n, 200, 6_000)
    s["Init_Win_bytes_forward"]  = sample(rng, 16_000, 8_000, n, 0)
    s["Init_Win_bytes_backward"] = sample(rng, 16_000, 8_000, n, 0)
    return s

# ── Heartbleed ────────────────────────────────────────────────────────
# Tiny packets (18 bytes), very short flow, near-zero pkt length std
def HEARTBLEED(n):
    s = make(n, {
        "Flow Duration":          (8_000, 1_000, 1_000, 20_000),
        "Total Fwd Packets":      (3, 0.4, 2, 5),
        "Total Backward Packets": (3, 0.4, 2, 5),
        "Total Length of Fwd Packets": (55, 6, 25, 100),
        "Total Length of Bwd Packets": (1_800, 300, 100),
        "Fwd Packet Length Mean": (18, 1.5, 14, 24),           # KEY: TLS heartbeat
        "Bwd Packet Length Mean": (18, 1.5, 14, 24),
        "Flow Bytes/s":           (4_000, 500, 500, 12_000),
        "Flow Packets/s":         (60, 8, 10, 120),
        "Flow IAT Mean":          (2_000, 300, 200, 8_000),
        "Flow IAT Std":           (180_000, 30_000, 5_000),
        "Flow IAT Max":           (800_000, 100_000, 50_000),
        "Fwd IAT Mean":           (200_000, 35_000, 5_000),
        "Bwd IAT Mean":           (200_000, 35_000, 5_000),
        "Fwd Packets/s":          (30, 5, 2),
        "Bwd Packets/s":          (25, 5, 2),
        "Packet Length Mean":     (18, 1.5, 14, 24),           # KEY: tiny uniform
        "Packet Length Std":      (2, 0.3, 0, 5),
        "Packet Length Variance": (4, 1, 0, 15),
        "Average Packet Size":    (400, 60, 50),
        "Avg Fwd Segment Size":   (300, 50, 50),
        "Avg Bwd Segment Size":   (280, 50, 50),
        "Init_Win_bytes_forward":  (0, 0, 0, 0),               # KEY: no window
        "Init_Win_bytes_backward": (0, 0, 0, 0),
        "Active Mean":            (100_000, 20_000, 10_000),
        "Idle Mean":              (500_000, 80_000, 50_000),
    })
    rng = np.random.RandomState(SEED+5)
    s["FIN Flag Count"] = np.zeros(n)
    s["SYN Flag Count"] = rng.binomial(1, 0.5, n).astype(float)
    s["RST Flag Count"] = np.zeros(n)
    s["PSH Flag Count"] = np.ones(n)
    s["ACK Flag Count"] = np.ones(n)
    return s

# ── Infiltration ──────────────────────────────────────────────────────
# Maxed-out packet sizes (exfil), asymmetric fwd>>bwd, medium rate
def INFILTRATION(n):
    s = make(n, {
        "Flow Duration":          (6_000_000, 600_000, 1_000_000),
        "Total Fwd Packets":      (500, 50, 100),
        "Total Backward Packets": (14, 3, 2, 30),
        "Total Length of Fwd Packets": (600_000, 60_000, 100_000),# KEY: huge
        "Total Length of Bwd Packets": (1_800, 300, 100),
        "Fwd Packet Length Mean": (1_200, 60, 600, 1_460),    # KEY: max-size pkts
        "Bwd Packet Length Mean": (280, 50, 50, 800),
        "Flow Bytes/s":           (500_000, 50_000, 80_000),
        "Flow Packets/s":         (60, 10, 5, 200),
        "Flow IAT Mean":          (220_000, 40_000, 10_000),
        "Flow IAT Std":           (180_000, 30_000, 5_000),
        "Flow IAT Max":           (800_000, 100_000, 50_000),
        "Fwd IAT Mean":           (200_000, 35_000, 5_000),
        "Bwd IAT Mean":           (200_000, 35_000, 5_000),
        "Fwd Packets/s":          (30, 5, 2),
        "Bwd Packets/s":          (25, 5, 2),
        "Packet Length Mean":     (1_100, 60, 500, 1_460),    # KEY: large
        "Packet Length Std":      (200, 30, 10),
        "Packet Length Variance": (40_000, 8_000, 100),
        "Average Packet Size":    (400, 60, 50),
        "Avg Fwd Segment Size":   (300, 50, 50),
        "Avg Bwd Segment Size":   (280, 50, 50),
        "Init_Win_bytes_forward":  (32_000, 16_000, 0),
        "Init_Win_bytes_backward": (32_000, 16_000, 0),
        "Active Mean":            (100_000, 20_000, 10_000),
        "Idle Mean":              (500_000, 80_000, 50_000),
    })
    s["FIN Flag Count"] = np.ones(n)
    s["SYN Flag Count"] = np.ones(n)
    s["RST Flag Count"] = np.zeros(n)
    s["PSH Flag Count"] = np.ones(n)
    s["ACK Flag Count"] = np.ones(n)
    return s

# ── Web Attack – Brute Force ──────────────────────────────────────────
# Medium pkt size, HTTP-like, many fwd packets, PSH+ACK
def WEB_BF(n):
    s = make(n, {
        "Flow Duration":          (300_000, 60_000, 50_000),
        "Total Fwd Packets":      (60, 8, 20, 150),
        "Total Backward Packets": (50, 8, 10, 120),
        "Total Length of Fwd Packets": (28_000, 5_000, 5_000),
        "Total Length of Bwd Packets": (22_000, 4_000, 4_000),
        "Fwd Packet Length Mean": (480, 40, 150, 1_200),
        "Bwd Packet Length Mean": (440, 40, 150, 1_200),
        "Flow Bytes/s":           (60_000, 8_000, 8_000),
        "Flow Packets/s":         (100, 12, 20, 300),
        "Flow IAT Mean":          (15_000, 2_000, 2_000, 50_000),
        "Flow IAT Std":           (12_000, 2_000, 2_000),
        "Flow IAT Max":           (800_000, 100_000, 50_000),
        "Fwd IAT Mean":           (200_000, 35_000, 5_000),
        "Bwd IAT Mean":           (200_000, 35_000, 5_000),
        "Fwd Packets/s":          (55, 8, 10),
        "Bwd Packets/s":          (45, 7, 8),
        "Packet Length Mean":     (460, 40, 150),
        "Packet Length Std":      (180, 25, 10),
        "Packet Length Variance": (32_400, 7_000, 100),
        "Average Packet Size":    (460, 40, 150),
        "Avg Fwd Segment Size":   (480, 40, 150),
        "Avg Bwd Segment Size":   (440, 40, 150),
        "Init_Win_bytes_forward":  (32_000, 16_000, 0),
        "Init_Win_bytes_backward": (32_000, 16_000, 0),
        "Active Mean":            (100_000, 20_000, 10_000),
        "Idle Mean":              (500_000, 80_000, 50_000),
    })
    s["FIN Flag Count"] = np.ones(n)
    s["SYN Flag Count"] = np.ones(n)
    s["RST Flag Count"] = np.zeros(n)
    s["PSH Flag Count"] = np.ones(n)
    s["ACK Flag Count"] = np.ones(n)
    return s

# ── Web Attack – XSS ──────────────────────────────────────────────────
def WEB_XSS(n):
    s = WEB_BF(n)
    rng = np.random.RandomState(SEED+12)
    s["Fwd Packet Length Mean"] = sample(rng, 620, 50, n, 200, 1_400)
    s["Avg Fwd Segment Size"]   = s["Fwd Packet Length Mean"].copy()
    s["Flow Packets/s"]         = sample(rng, 80, 10, n, 15, 250)
    return s

# ── NEW: Web Attack – SQL Injection ──────────────────────────────────
def WEB_SQLI(n):
    s = WEB_BF(n)
    rng = np.random.RandomState(SEED+13)
    s["Fwd Packet Length Mean"] = sample(rng, 750, 60, n, 300, 1_400)
    s["Avg Fwd Segment Size"]   = s["Fwd Packet Length Mean"].copy()
    s["Total Fwd Packets"]      = sample(rng, 40, 6, n, 10, 100)
    s["Flow Duration"]          = sample(rng, 200_000, 40_000, n, 30_000)
    return s

# ── NEW: MSSQL Bruteforce ──────────────────────────────────────────────
# Very short flows, SYN-heavy, tiny packets
def MSSQL_BF(n):
    s = make(n, {
        "Flow Duration":          (2_000, 400, 200, 8_000),
        "Total Fwd Packets":      (3, 0.5, 1, 6),
        "Total Backward Packets": (2, 0.4, 0, 5),
        "Total Length of Fwd Packets": (120, 20, 40, 300),
        "Total Length of Bwd Packets": (80, 15, 20, 200),
        "Fwd Packet Length Mean": (40, 5, 20, 70),
        "Bwd Packet Length Mean": (40, 5, 20, 70),
        "Flow Bytes/s":           (80_000, 10_000, 10_000),
        "Flow Packets/s":         (2_000, 300, 400),
        "Flow IAT Mean":          (600, 100, 50, 2_000),
        "Flow IAT Std":           (300, 60, 10, 1_000),
        "Flow IAT Max":           (800_000, 100_000, 50_000),
        "Fwd IAT Mean":           (200_000, 35_000, 5_000),
        "Bwd IAT Mean":           (200_000, 35_000, 5_000),
        "Fwd Packets/s":          (1_500, 200, 200),
        "Bwd Packets/s":          (800, 100, 100),
        "Packet Length Mean":     (40, 5, 20, 70),
        "Packet Length Std":      (5, 1, 0, 20),
        "Packet Length Variance": (25, 5, 0, 100),
        "Average Packet Size":    (400, 60, 50),
        "Avg Fwd Segment Size":   (300, 50, 50),
        "Avg Bwd Segment Size":   (280, 50, 50),
        "Init_Win_bytes_forward":  (0, 0, 0, 0),
        "Init_Win_bytes_backward": (0, 0, 0, 0),
        "Active Mean":            (500, 100, 50),
        "Idle Mean":              (1_000, 200, 100),
    })
    s["FIN Flag Count"] = np.zeros(n)
    s["SYN Flag Count"] = np.ones(n)
    s["RST Flag Count"] = np.ones(n)
    s["PSH Flag Count"] = np.zeros(n)
    s["ACK Flag Count"] = np.zeros(n)
    return s

# ── NEW: DNS Tunneling ────────────────────────────────────────────────
# Tiny packets with unusually high fwd counts, low bytes/s
def DNS_TUNNEL(n):
    s = make(n, {
        "Flow Duration":          (10_000_000, 1_000_000, 2_000_000),
        "Total Fwd Packets":      (200, 30, 50),
        "Total Backward Packets": (180, 25, 40),
        "Total Length of Fwd Packets": (12_000, 2_000, 2_000),
        "Total Length of Bwd Packets": (22_000, 3_000, 3_000),
        "Fwd Packet Length Mean": (60, 8, 20, 100),
        "Bwd Packet Length Mean": (120, 15, 40, 200),
        "Flow Bytes/s":           (3_000, 400, 400),
        "Flow Packets/s":         (35, 5, 5, 100),
        "Flow IAT Mean":          (50_000, 8_000, 5_000),
        "Flow IAT Std":           (25_000, 4_000, 2_000),
        "Flow IAT Max":           (800_000, 100_000, 50_000),
        "Fwd IAT Mean":           (200_000, 35_000, 5_000),
        "Bwd IAT Mean":           (200_000, 35_000, 5_000),
        "Fwd Packets/s":          (18, 3, 2),
        "Bwd Packets/s":          (17, 3, 2),
        "Packet Length Mean":     (90, 12, 30, 200),
        "Packet Length Std":      (40, 6, 5, 100),
        "Packet Length Variance": (1_600, 400, 25),
        "Average Packet Size":    (90, 12, 30),
        "Avg Fwd Segment Size":   (60, 8, 20),
        "Avg Bwd Segment Size":   (120, 15, 40),
        "Init_Win_bytes_forward":  (32_000, 16_000, 0),
        "Init_Win_bytes_backward": (32_000, 16_000, 0),
        "Active Mean":            (100_000, 20_000, 10_000),
        "Idle Mean":              (500_000, 80_000, 50_000),
    })
    s["FIN Flag Count"] = np.zeros(n)
    s["SYN Flag Count"] = np.zeros(n)
    s["RST Flag Count"] = np.zeros(n)
    s["PSH Flag Count"] = np.ones(n)
    s["ACK Flag Count"] = np.ones(n)
    return s

# ── NEW: ICMP Flood ───────────────────────────────────────────────────
# Tiny fixed-size 64-byte packets, no TCP flags, massive rate
def ICMP_FLOOD(n):
    s = make(n, {
        "Flow Duration":          (8_000, 1_000, 1_000),
        "Total Fwd Packets":      (5_000, 600, 1_000),
        "Total Backward Packets": (0, 0, 0, 0),
        "Total Length of Fwd Packets": (320_000, 40_000, 60_000),
        "Total Length of Bwd Packets": (0, 0, 0, 0),
        "Fwd Packet Length Mean": (64, 2, 56, 72),
        "Bwd Packet Length Mean": (0, 0, 0, 0),
        "Flow Bytes/s":           (40_000_000, 4_000_000, 8_000_000),
        "Flow Packets/s":         (625_000, 60_000, 100_000),
        "Flow IAT Mean":          (160, 30, 10, 500),
        "Flow IAT Std":           (80, 15, 5, 250),
        "Flow IAT Max":           (800_000, 100_000, 50_000),
        "Fwd IAT Mean":           (200_000, 35_000, 5_000),
        "Bwd IAT Mean":           (200_000, 35_000, 5_000),
        "Fwd Packets/s":          (625_000, 60_000, 100_000),
        "Bwd Packets/s":          (0, 0, 0, 0),
        "Packet Length Mean":     (64, 2, 56, 72),
        "Packet Length Std":      (2, 0.3, 0, 5),
        "Packet Length Variance": (4, 1, 0, 15),
        "Average Packet Size":    (400, 60, 50),
        "Avg Fwd Segment Size":   (300, 50, 50),
        "Avg Bwd Segment Size":   (280, 50, 50),
        "Init_Win_bytes_forward":  (0, 0, 0, 0),
        "Init_Win_bytes_backward": (0, 0, 0, 0),
        "Active Mean":            (1_000, 200, 100),
        "Idle Mean":              (500, 100, 50),
    })
    s["FIN Flag Count"] = np.zeros(n)
    s["SYN Flag Count"] = np.zeros(n)
    s["RST Flag Count"] = np.zeros(n)
    s["PSH Flag Count"] = np.zeros(n)
    s["ACK Flag Count"] = np.zeros(n)
    return s

GENERATORS = {
    "DoS Hulk":               DOS_HULK,
    "PortScan":               PORTSCAN,
    "DDoS":                   DDOS,
    "DoS GoldenEye":          DOS_GOLDENEYE,
    "DoS slowloris":          DOS_SLOWLORIS,
    "DoS Slowhttptest":       DOS_SLOWHTTP,
    "FTP-Patator":            FTP_PATATOR,
    "SSH-Patator":            SSH_PATATOR,
    "Bot":                    BOT,
    "Web Attack – Brute Force": WEB_BF,
    "Web Attack – XSS":       WEB_XSS,
    "Web Attack – Sql Injection": WEB_SQLI,
    "Infiltration":           INFILTRATION,
    "Heartbleed":             HEARTBLEED,
    "MSSQL Bruteforce":       MSSQL_BF,
    "DNS Tunneling":          DNS_TUNNEL,
    "ICMP Flood":             ICMP_FLOOD,
}

def generate(n_benign=9750, n_per_attack=600, out="data/cicids2017_synthetic.csv"):
    print(f"Generating {n_benign:,} BENIGN + {n_per_attack}×{len(GENERATORS)} attacks …")
    rows = []
    bd = BENIGN(n_benign)
    for i in range(n_benign):
        rows.append({f: float(bd[f][i]) for f in FEATURES} | {"Label":"BENIGN"})
    for label, fn in GENERATORS.items():
        ad = fn(n_per_attack)
        for i in range(n_per_attack):
            rows.append({f: float(ad[f][i]) for f in FEATURES} | {"Label": label})
    df = pd.DataFrame(rows).sample(frac=1,random_state=SEED).reset_index(drop=True)
    df.replace([float('inf'),float('-inf')],np.nan,inplace=True); df.fillna(0,inplace=True)
    os.makedirs(os.path.dirname(out) if os.path.dirname(out) else ".", exist_ok=True)
    df.to_csv(out, index=False)
    na = (df["Label"]!="BENIGN").sum()
    print(f"[✓] {len(df):,} rows → {out}  (BENIGN:{n_benign:,} | Attacks:{na:,} | {na/len(df)*100:.1f}%)")

if __name__ == "__main__":
    generate()