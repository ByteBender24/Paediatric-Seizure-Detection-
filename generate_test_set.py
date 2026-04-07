"""
generate_test_set.py
────────────────────
One-time script. Reads your training CSV, applies the exact same pipeline
used in Paediatric_V5 (SMOTE, train_test_split with stratify & random_state=42),
and saves the held-out test split as:

    test_set/X_test.npy   — shape (N, 23, 256)
    test_set/y_test.npy   — shape (N,)

Run once:
    python generate_test_set.py

Then the app uses test_set/ for all metric evaluation and GNN topology.
"""

import os, gc
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE

# ─── CONFIG — edit these to match your setup ─────────────────────────────────
CSV_PATH   = './data/training_data.csv'   # ← path to your labelled training CSV
TARGET_COL = 'target'                     # ← column name for labels (0=normal, 1=seizure)
OUT_DIR    = './test_set'

# Must match Paediatric_V5 exactly
NUM_CHANNELS   = 23
TIME_STEPS     = 256
FEATS_PER_SAMPLE = NUM_CHANNELS * TIME_STEPS   # 5888
RANDOM_STATE   = 42
# ─────────────────────────────────────────────────────────────────────────────

os.makedirs(OUT_DIR, exist_ok=True)

print("=" * 60)
print("  generate_test_set.py  —  Paediatric V5 pipeline")
print("=" * 60)

# 1. Load CSV
print(f"\n[1/5] Loading CSV from: {CSV_PATH}")
df = pd.read_csv(CSV_PATH)
print(f"      Shape: {df.shape}")

y_raw = df[TARGET_COL].values.astype(int)
X_raw = df.drop(columns=[TARGET_COL]).values.astype(np.float32)
print(f"      Classes: {dict(zip(*np.unique(y_raw, return_counts=True)))}")

# 2. Trim to FEATS_PER_SAMPLE (5888) — exactly as V5
print(f"\n[2/5] Feature trimming: {X_raw.shape[1]} → {FEATS_PER_SAMPLE}")
if X_raw.shape[1] > FEATS_PER_SAMPLE:
    X_raw = X_raw[:, :FEATS_PER_SAMPLE]
    print(f"      Trimmed to {X_raw.shape}")
else:
    print(f"      No trimming needed.")

# 3. SMOTE — same random_state as V5
print(f"\n[3/5] SMOTE balancing (random_state={RANDOM_STATE})...")
smote = SMOTE(random_state=RANDOM_STATE)
X_bal, y_bal = smote.fit_resample(X_raw, y_raw)
print(f"      Balanced shape: {X_bal.shape}")
print(f"      Classes after SMOTE: {dict(zip(*np.unique(y_bal, return_counts=True)))}")

# 4. Reshape → (N, 23, 256)
X_bal = X_bal.reshape(-1, NUM_CHANNELS, TIME_STEPS)
print(f"\n[4/5] Reshape → {X_bal.shape}")

# 5. 3-way split — exactly V5: 70/15/15, stratify, random_state=42
print(f"\n[5/5] Splitting: 70% train / 15% val / 15% test (stratified)...")
X_train, X_temp, y_train, y_temp = train_test_split(
    X_bal, y_bal, test_size=0.30, stratify=y_bal, random_state=RANDOM_STATE
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.50, stratify=y_temp, random_state=RANDOM_STATE
)
print(f"      Train: {X_train.shape}  Val: {X_val.shape}  Test: {X_test.shape}")

# 6. Save test set
X_test_path = os.path.join(OUT_DIR, 'X_test.npy')
y_test_path = os.path.join(OUT_DIR, 'y_test.npy')
np.save(X_test_path, X_test)
np.save(y_test_path, y_test)

print(f"\n{'='*60}")
print(f"  ✓ Saved X_test → {X_test_path}  {X_test.shape}")
print(f"  ✓ Saved y_test → {y_test_path}  {y_test.shape}")
print(f"  Classes in test: {dict(zip(*np.unique(y_test, return_counts=True)))}")
print(f"{'='*60}")
print("\nDone! The app will auto-load these on next startup.\n")
