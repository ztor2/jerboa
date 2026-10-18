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

# Verify CUDA availability and actual kernel execution (e.g. Blackwell sm_120 on RTX 5090 requires cu128)
if ! python3 -c "
import torch
assert torch.cuda.is_available(), 'CUDA not available'
x = torch.ones((4, 4), device='cuda', dtype=torch.bfloat16) @ torch.ones((4, 4), device='cuda', dtype=torch.bfloat16)
torch.cuda.synchronize()
" 2>/dev/null; then
    echo "[!] PyTorch CUDA kernel test failed (e.g. Blackwell sm_120 / RTX 5090 requires CUDA 12.8). Installing PyTorch cu128..."
    pip install --upgrade pip
    pip install --upgrade torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
fi
echo "[✓] CUDA Hardware: $(python3 -c "import torch; print(f'{torch.cuda.get_device_name(0)} (sm_{torch.cuda.get_device_capability(0)[0]}{torch.cuda.get_device_capability(0)[1]}, {torch.cuda.get_device_properties(0).total_memory // (1024**3)}GB)' if torch.cuda.is_available() else 'CPU (No CUDA)')")"


# 3. Upgrade pip and install core dependencies
echo "[3/4] Installing core requirements from requirements.txt..."
pip install --upgrade pip
pip install -r requirements.txt

# 4. PyTorch Native SDPA & FlashAttention Verification
echo "[4/4] Verifying PyTorch Native SDPA & FlashAttention..."
python3 -c "import torch; print(f'[✓] PyTorch SDPA FlashAttention Backend: {torch.backends.cuda.flash_sdp_enabled()}')" 2>/dev/null || true

echo ""
echo "=== [JerboaLM] Environment Setup Complete! ==="
echo "To activate environment: source $VENV_PATH/bin/activate"
echo "To start pre-training:   bash scripts/start_runpod.sh"
