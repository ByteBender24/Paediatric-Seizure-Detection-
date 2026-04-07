#!/bin/bash
# ─── NeuroScan Local Runner ───────────────────────────────────────────────────
set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

echo ""
echo "  🧠  NeuroScan — Paediatric Seizure Analyzer"
echo "  ─────────────────────────────────────────────"

# Create venv if not exists
if [ ! -d "venv" ]; then
  echo "  [1/3] Creating virtual environment..."
  python3 -m venv venv
fi

echo "  [2/3] Activating venv & installing deps..."
source venv/bin/activate
pip install -q --upgrade pip
pip install -q -r requirements.txt

mkdir -p models uploads

echo "  [3/3] Starting server..."
echo ""
echo "  ─────────────────────────────────────────────"
echo "  📌  Open in browser:  http://localhost:5000"
echo "  ─────────────────────────────────────────────"
echo ""
python app.py
