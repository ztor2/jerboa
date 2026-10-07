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
RECREATE_VENV=false
if [ ! -d "$VENV_PATH" ]; then
    RECREATE_VENV=true
else
    # Check if existing venv has CUDA working
    if ! "$VENV_PATH/bin/python3" -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
        echo "[!] Notice: Existing venv at $VENV_PATH lacks CUDA support. Recreating..."
        rm -rf "$VENV_PATH"
        RECREATE_VENV=true
    else
        echo "[2/4] Valid CUDA-enabled virtual environment found at $VENV_PATH."
    fi
fi

if [ "$RECREATE_VENV" = true ]; then
    echo "[2/4] Creating virtual environment at $VENV_PATH (inheriting system PyTorch/CUDA)..."
    python3 -m venv --system-site-packages "$VENV_PATH"
fi

source "$VENV_PATH/bin/activate"

# Verify and enforce CUDA inside venv
if ! python3 -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
    echo "[!] PyTorch does not see CUDA. Installing PyTorch with CUDA 12.4 support..."
    pip install --upgrade pip
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
fi
echo "[✓] CUDA Hardware: $(python3 -c "import torch; print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU (No CUDA)')")"


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
