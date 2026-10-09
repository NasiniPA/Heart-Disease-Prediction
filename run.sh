#!/usr/bin/env bash
echo "======================================================================"
echo "         CardioPredict ML - Heart Disease Prediction Launcher"
echo "======================================================================"

if ! command -v python3 &> /dev/null; then
    echo "[ERROR] python3 could not be found. Please install Python 3.9+."
    exit 1
fi

echo "[1/3] Installing dependencies..."
python3 -m pip install -r requirements.txt

echo "[2/3] Checking trained model..."
if [ ! -f "model.pkl" ]; then
    echo "Training model..."
    python3 train_model.py
fi

echo "[3/3] Launching Streamlit..."
python3 -m streamlit run app.py
