#!/bin/bash
# NeuroScan EEG Seizure Analyzer — Setup Script
echo "============================================"
echo "  NeuroScan — Paediatric Seizure Analyzer"
echo "============================================"
echo ""

# Create virtual environment
echo "[1/4] Creating virtual environment..."
python3 -m venv venv
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip --quiet

# Install dependencies
echo "[2/4] Installing dependencies..."
pip install -r requirements.txt

echo "[3/4] Creating directories..."
mkdir -p models uploads

echo "[4/4] Setup complete!"
echo ""
echo "============================================"
echo "  USAGE:"
echo ""
echo "  1. Copy your .pth model files to ./models/"
echo "     - CNN-GRU model → ./models/gru_model.pth"
echo "     - GNN model     → ./models/gnn_model.pth"
echo ""
echo "  2. Run the app:"
echo "     source venv/bin/activate"
echo "     python app.py"
echo ""
echo "  3. Open: http://localhost:5000"
echo "============================================"
