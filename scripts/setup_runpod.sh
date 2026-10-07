#!/bin/bash
# One-click environment setup script for RunPod GPU instances
# Usage: bash scripts/setup_runpod.sh

set -e

echo "=== [JerboaLM] Setting up RunPod Environment ==="

# 1. Verify working directory
WORKSPACE_DIR="/workspace"
if [ -d "$WORKSPACE_DIR" ]; then
    echo "[1/4] Persistent workspace detected at $WORKSPACE_DIR."
    VENV_PATH="$WORKSPACE_DIR/venv"
else
    echo "[1/4] Running in local/custom container without /workspace, using local venv."
    VENV_PATH="./venv"
fi

# 2. Setup Python Virtual Environment in persistent storage
if [ ! -d "$VENV_PATH" ]; then
    echo "[2/4] Creating virtual environment at $VENV_PATH..."
    python3 -m venv "$VENV_PATH"
else
    echo "[2/4] Existing virtual environment found at $VENV_PATH."
fi

source "$VENV_PATH/bin/activate"

# 3. Upgrade pip and install core dependencies
echo "[3/4] Installing core requirements from requirements.txt..."
pip install --upgrade pip
pip install -r requirements.txt

# 4. Optional: FlashAttention-2 installation for Ampere/Ada/Blackwell
echo "[4/4] Checking GPU for FlashAttention-2..."
if python3 -c "import torch; assert torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8" 2>/dev/null; then
    echo "Ampere+ GPU detected (SM >= 8.0). Installing flash-attn for 2x speedup..."
    pip install flash-attn --no-build-isolation || echo "flash-attn install skipped (fallback to PyTorch SDPA is active)."
else
    echo "Non-CUDA or SM < 8.0 environment. PyTorch SDPA will be used natively."
fi

echo ""
echo "=== [JerboaLM] Environment Setup Complete! ==="
echo "To activate environment: source $VENV_PATH/bin/activate"
echo "To start pre-training:   python pipeline/pretrain.py --recipe recipes/pretrain/phase1_base.yaml"
