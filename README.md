# 🧠 NeuroScan — Paediatric EEG Seizure Analyzer

Single-page Flask web app for seizure detection and brain spatial mapping.

## ── QUICK START (Local Machine) ──────────────────────────────────────────────

### Prerequisites
- Python 3.9 or newer
- pip

### Steps

```bash
# 1. Extract the zip
unzip seizure_app.zip
cd seizure_app

# 2. One-command setup + launch
./run_local.sh

# 3. Browser opens at → http://localhost:5000
```

### First time takes ~2 minutes to install packages.
### After that:  `./run_local.sh`  is instant.

---

## ── MODEL FILES ──────────────────────────────────────────────────────────────

### Option A — Auto-load (recommended)
Copy your .pth files before running:
```
models/gru_model.pth   ←  best_cnn_gru_stable_model.pth
models/gnn_model.pth   ←  final_model.pth  (from Paediatric_V5/)
```

### Option B — Upload via UI
Use the upload buttons in the sidebar after launching.

---

## ── WHAT HAPPENS ─────────────────────────────────────────────────────────────

```
EDF Upload
  ├─ MNE: bandpass 0.5–40Hz, resample 256Hz, pick 23 channels
  ├─ Z-score normalize
  ├─ Segment: 5-second windows (256×5=1280 samples)
  ├─ Flatten: 23×1280 = 29440 → pad to 36864 columns  (CHB-MIT spec)
  ├─ Second z-score, reshape → (N, 23, time_steps)
  │
  ├─ [EDA]        Signal overview · PSD · Amplitude heatmap · Channel RMS
  ├─ [CNN-GRU]    Seizure probability per window → timeline plot
  ├─ [Snapshot]   23-channel EEG waveform of first detected seizure window
  ├─ [GNN]        Adjacency matrix → functional connectivity graph
  ├─ [Topomap]    MNE standard 10-20 brain heatmap (seizure focus)
  └─ [Electrodes] Top 5 seizure-focus electrodes by degree centrality
```

---

## ── FILE STRUCTURE ───────────────────────────────────────────────────────────

```
seizure_app/
├── app.py              ← Flask backend + all ML logic
├── templates/
│   └── index.html      ← Single-page frontend (no frameworks)
├── models/             ← Place your .pth files here
├── uploads/            ← Temp EDF storage (auto-deleted after analysis)
├── requirements.txt
├── run_local.sh        ← One-click runner
└── README.md
```

---

## ── THRESHOLD ────────────────────────────────────────────────────────────────

Default: **80%**. Edit `SEIZURE_THRESHOLD = 0.80` in `app.py` to change.
