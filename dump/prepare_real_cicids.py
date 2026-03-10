"""
prepare_real_cicids.py — Real CICIDS2017 Dataset Preparation
=============================================================
Combines the 5 real CICIDS2017 CSV files into a single clean CSV
that model.py can train on directly.

Steps:
  1. Download CSVs from https://www.unb.ca/cic/datasets/ids-2017.html
     (request the MachineLearningCSV folder)
  2. Put all 5 CSV files in the same folder as this script (data/)
  3. Run:  python data/prepare_real_cicids.py
  4. Then: python model.py --train --data data/cicids2017_real.csv

Output: data/cicids2017_real.csv
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd

# ── Config ────────────────────────────────────────────────────────────────────
DATA_DIR   = Path(__file__).parent
OUTPUT     = DATA_DIR / "cicids2017_real.csv"

# Real CICIDS2017 CSV filenames (exact names from the download)
CSV_FILES = [
    "Monday-WorkingHours.pcap_ISCX.csv",
    "Tuesday-WorkingHours.pcap_ISCX.csv",
    "Wednesday-workingHours.pcap_ISCX.csv",        # note: lowercase 'w' in working
    "Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv",
    "Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv",
    "Friday-WorkingHours-Morning.pcap_ISCX.csv",
    "Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv",
    "Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv",
]

# The real dataset uses " Label" (with leading space) — we strip it
LABEL_COL = "Label"

# CICIDS2017 column name fixes: real CSV names → our internal names
# The real CSVs have slightly different capitalization/spacing on some columns
COL_MAP = {
    # Real CSV name                          : Internal name
    "Flow Bytes/s"                           : "Flow Bytes/s",
    "Flow Packets/s"                         : "Flow Packets/s",
    "Fwd Packets/s"                          : "Fwd Packets/s",
    "Bwd Packets/s"                          : "Bwd Packets/s",
    "Total Length of Fwd Packets"            : "Total Length of Fwd Packets",
    "Total Length of Bwd Packets"            : "Total Length of Bwd Packets",
    "Fwd Packet Length Mean"                 : "Fwd Packet Length Mean",
    "Bwd Packet Length Mean"                 : "Bwd Packet Length Mean",
    "Flow IAT Mean"                          : "Flow IAT Mean",
    "Flow IAT Std"                           : "Flow IAT Std",
    "Flow IAT Max"                           : "Flow IAT Max",
    "Fwd IAT Mean"                           : "Fwd IAT Mean",
    "Bwd IAT Mean"                           : "Bwd IAT Mean",
    "Packet Length Mean"                     : "Packet Length Mean",
    "Packet Length Std"                      : "Packet Length Std",
    "Packet Length Variance"                 : "Packet Length Variance",
    "Average Packet Size"                    : "Average Packet Size",
    "Avg Fwd Segment Size"                   : "Avg Fwd Segment Size",
    "Avg Bwd Segment Size"                   : "Avg Bwd Segment Size",
    "Init_Win_bytes_forward"                 : "Init_Win_bytes_forward",
    "Init_Win_bytes_backward"                : "Init_Win_bytes_backward",
    "act_data_pkt_fwd"                       : "act_data_pkt_fwd",
    # Some versions use these alternate names:
    "Init Win bytes forward"                 : "Init_Win_bytes_forward",
    "Init Win bytes backward"                : "Init_Win_bytes_backward",
}

# Label normalization — real CICIDS has some inconsistent label strings
# Normalise every known label variant including garbled UTF-8 encodings
def _fix_label(raw: str) -> str:
    s = raw.strip()
    # normalise all "Web Attack" variants — handle garbled chars robustly
    if "Web Attack" in s or "Web attack" in s:
        sl = s.lower()
        if "brute" in sl:   return "Web Attack – Brute Force"
        if "xss"   in sl:   return "Web Attack – XSS"
        if "sql"   in sl:   return "Web Attack – SQL Injection"
        return "Web Attack – Other"
    fixes = {
        "BENIGN": "BENIGN", "Bot": "Bot", "DDoS": "DDoS",
        "DoS GoldenEye": "DoS GoldenEye", "DoS Hulk": "DoS Hulk",
        "DoS Slowhttptest": "DoS Slowhttptest", "DoS slowloris": "DoS slowloris",
        "FTP-Patator": "FTP-Patator", "Heartbleed": "Heartbleed",
        "Infiltration": "Infiltration", "PortScan": "PortScan",
        "SSH-Patator": "SSH-Patator",
    }
    return fixes.get(s, s)

LABEL_FIXES = {}  # kept for compatibility — actual fixing done by _fix_label()

# Max rows to sample per class (keeps dataset manageable — ~500k total)
# Set to None to use ALL rows (warning: full dataset is ~2.8M rows, slow to train)
MAX_BENIGN  = 200_000   # real dataset has ~2M BENIGN rows — sample it down
MAX_ATTACK  = None      # keep all attack rows (they're naturally rarer)


def load_and_clean(path: Path) -> pd.DataFrame:
    print(f"  Loading {path.name} …", end=" ", flush=True)
    try:
        df = pd.read_csv(path, encoding="utf-8", low_memory=False)
    except UnicodeDecodeError:
        df = pd.read_csv(path, encoding="latin-1", low_memory=False)

    # Strip whitespace from column names (CICIDS has " Label" with space)
    df.columns = df.columns.str.strip()

    # Fix label column
    if LABEL_COL not in df.columns:
        # try with space
        spacecols = [c for c in df.columns if c.lower().strip() == "label"]
        if spacecols:
            df.rename(columns={spacecols[0]: LABEL_COL}, inplace=True)
        else:
            print(f"\n    [!] No Label column found in {path.name} — skipping")
            return pd.DataFrame()

    # Normalize label strings
    df[LABEL_COL] = df[LABEL_COL].astype(str).str.strip()
    df[LABEL_COL] = df[LABEL_COL].map(_fix_label)

    # Drop rows with inf/nan in numeric columns
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(subset=[LABEL_COL], inplace=True)

    print(f"{len(df):,} rows  |  labels: {df[LABEL_COL].nunique()}")
    return df


def main():
    print("=" * 60)
    print("  CICIDS2017 Real Data Preparation")
    print("=" * 60)

    # Check files exist
    found, missing = [], []
    for fname in CSV_FILES:
        p = DATA_DIR / fname
        if p.exists():
            found.append(p)
        else:
            missing.append(fname)

    if missing:
        print(f"\n[!] Missing files in {DATA_DIR}:")
        for m in missing:
            print(f"    ✗  {m}")
        print("\nDownload from: https://www.unb.ca/cic/datasets/ids-2017.html")
        print("Put all 5 CSV files in the data/ folder then rerun.\n")
        if not found:
            sys.exit(1)
        print(f"[~] Proceeding with {len(found)} available files...\n")

    # Load all found CSVs
    frames = []
    for p in found:
        df = load_and_clean(p)
        if not df.empty:
            frames.append(df)

    if not frames:
        print("[✗] No data loaded. Check your CSV files.")
        sys.exit(1)

    combined = pd.concat(frames, ignore_index=True)
    print(f"\n[✓] Combined: {len(combined):,} total rows")
    print(f"\nFull label distribution:")
    print(combined[LABEL_COL].value_counts().to_string())

    # ── Sample BENIGN down to avoid massive class imbalance ───────────────────
    benign_df = combined[combined[LABEL_COL] == "BENIGN"]
    attack_df = combined[combined[LABEL_COL] != "BENIGN"]

    if MAX_BENIGN and len(benign_df) > MAX_BENIGN:
        benign_df = benign_df.sample(MAX_BENIGN, random_state=42)
        print(f"\n[~] Sampled BENIGN down to {MAX_BENIGN:,} rows")

    if MAX_ATTACK and len(attack_df) > MAX_ATTACK:
        attack_df = attack_df.sample(MAX_ATTACK, random_state=42)

    final = pd.concat([benign_df, attack_df], ignore_index=True)
    final = final.sample(frac=1, random_state=42).reset_index(drop=True)  # shuffle

    # Fill remaining NaN with 0
    final.replace([np.inf, -np.inf], np.nan, inplace=True)
    final.fillna(0, inplace=True)

    # ── Final stats ───────────────────────────────────────────────────────────
    print(f"\n[✓] Final dataset: {len(final):,} rows")
    print(f"\nFinal label distribution:")
    print(final[LABEL_COL].value_counts().to_string())
    attack_ratio = (final[LABEL_COL] != "BENIGN").mean()
    print(f"\nAttack ratio: {attack_ratio*100:.1f}%")

    # Save
    final.to_csv(OUTPUT, index=False)
    print(f"\n[✓] Saved to {OUTPUT}")
    print("\nNext step:")
    print("  python model.py --train --data data/cicids2017_real.csv")
    print("=" * 60)


if __name__ == "__main__":
    main()