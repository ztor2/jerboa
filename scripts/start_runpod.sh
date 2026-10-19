#!/bin/bash
# ==============================================================================
# JerboaLM All-In-One Automated RunPod Launch Script
# Usage:
#   bash scripts/start_runpod.sh             # Auto-detects 1x or 2x GPU and starts
#   bash scripts/start_runpod.sh --no_wandb  # Forwards CLI flags directly
# ==============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

echo "======================================================================"
echo "          JerboaLM Pre-training Automated Launcher"
echo "======================================================================"

# 1. Environment & Path Setup
if [ -f "$PROJECT_ROOT/.env" ]; then
    set -a
    source "$PROJECT_ROOT/.env"
    set +a
fi
export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export PIP_BREAK_SYSTEM_PACKAGES=1

WORKSPACE_DIR="/workspace"
if [ -d "$WORKSPACE_DIR" ]; then
    VENV_PATH="$WORKSPACE_DIR/venv"
else
    VENV_PATH="$PROJECT_ROOT/venv"
fi

# 2. Virtual Environment Verification (with Ubuntu 24.04 fallback)
if [ ! -d "$VENV_PATH" ]; then
    echo "[1/4] Creating persistent venv with system CUDA packages..."
    if ! python3 -m venv --system-site-packages "$VENV_PATH" 2>/dev/null; then
        echo "[!] Notice: python3-venv not pre-installed on this image; using system Python environment directly."
        VENV_PATH=""
    fi
fi
if [ -n "$VENV_PATH" ] && [ -f "$VENV_PATH/bin/activate" ]; then
    source "$VENV_PATH/bin/activate"
fi

# 3. Fast Dependency Sync
echo "[2/4] Ensuring core packages are installed..."
python3 -m pip install -r requirements.txt --quiet

# 4. GPU Hardware & Kernel Compatibility Detection
echo "[3/4] Detecting GPU Hardware & Topology..."
NUM_GPUS=$(python3 -c "import torch; print(torch.cuda.device_count() if torch.cuda.is_available() else 0)")

if [ "$NUM_GPUS" -eq 0 ]; then
    echo "[!] ERROR: No CUDA GPUs detected. Cannot start GPU pre-training."
    exit 1
fi

# Ensure Blackwell (sm_120, RTX 5090) has compatible CUDA 12.8 kernels installed
if ! python3 -c "
import torch
x = torch.ones((4, 4), device='cuda', dtype=torch.bfloat16) @ torch.ones((4, 4), device='cuda', dtype=torch.bfloat16)
torch.cuda.synchronize()
" 2>/dev/null; then
    echo "[!] Notice: Current PyTorch build lacks kernels for this GPU architecture (e.g. Blackwell sm_120)."
    echo "[!] Upgrading PyTorch to CUDA 12.8 (cu128)..."
    python3 -m pip install --upgrade torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
fi

GPU_NAME=$(python3 -c "import torch; print(torch.cuda.get_device_name(0))")
GPU_VRAM_GB=$(python3 -c "import torch; print(torch.cuda.get_device_properties(0).total_memory // (1024**3))")
echo "[✓] GPU Detected: $NUM_GPUS x $GPU_NAME (${GPU_VRAM_GB}GB VRAM per GPU)"

# Start background Hugging Face Hub backup daemon if authenticated (.env HF_TOKEN or hf auth login)
BACKUP_PID=""
HF_USER=$(python3 -c "
import os
from huggingface_hub import whoami
try:
    print(whoami(token=os.getenv('HF_TOKEN'))['name'])
except Exception:
    print('')
" 2>/dev/null)

if [ -n "$HF_USER" ]; then
    echo "[✓] Hugging Face Authenticated as: $HF_USER"
    if ! pgrep -f "hub_backup_daemon.py" >/dev/null 2>&1; then
        echo "[✓] Starting background HF Hub backup daemon (ztor2/jerboa-base, every 50 chunks)..."
        nohup python3 scripts/tools/hub_backup_daemon.py --repo_id ztor2/jerboa-base --interval_chunks 50 --public > hub_backup.log 2>&1 &
        BACKUP_PID=$!
    else
        echo "[✓] Background HF Hub backup daemon is already running."
    fi
else
    echo "[!] WARNING: Hugging Face token not detected (.env HF_TOKEN or 'hf auth login'). Automatic Hub backup is disabled!"
fi

# 5. Launch Training Pipeline (Auto-select 5090 vs 4090 recipe)
echo "[4/4] Launching Training Pipeline..."
if [[ "$GPU_NAME" == *"5090"* ]] || [ "$GPU_VRAM_GB" -ge 30 ]; then
    if [ "$NUM_GPUS" -ge 2 ]; then
        RECIPE="recipes/pretrain/runpod_5090_ddp.yaml"
        TARGET_SPEED="~140,000 - 170,000 tok/s"
    else
        RECIPE="recipes/pretrain/runpod_5090.yaml"
        TARGET_SPEED="~75,000 - 95,000 tok/s"
    fi
else
    if [ "$NUM_GPUS" -ge 2 ]; then
        RECIPE="recipes/pretrain/runpod_4090_ddp.yaml"
        TARGET_SPEED="~110,000 - 120,000 tok/s"
    else
        RECIPE="recipes/pretrain/runpod_4090.yaml"
        TARGET_SPEED="~55,000 - 65,000 tok/s"
    fi
fi

if [ "$NUM_GPUS" -ge 2 ]; then
    echo "======================================================================"
    echo " Mode:     Multi-GPU DDP ($NUM_GPUS x $GPU_NAME)"
    echo " Recipe:   $RECIPE"
    echo " Target:   $TARGET_SPEED"
    echo "======================================================================"
    python3 -m torch.distributed.run --nproc_per_node="$NUM_GPUS" pipeline/pretrain.py --recipe "$RECIPE" "$@"
else
    echo "======================================================================"
    echo " Mode:     Single GPU ($GPU_NAME)"
    echo " Recipe:   $RECIPE"
    echo " Target:   $TARGET_SPEED"
    echo "======================================================================"
    python3 pipeline/pretrain.py --recipe "$RECIPE" "$@"
fi

# Final checkpoint upload sync
if [ -n "$HF_USER" ]; then
    echo "[✓] Running final checkpoint backup to Hugging Face Hub (ztor2/jerboa-base)..."
    python3 scripts/tools/hub_backup_daemon.py --repo_id ztor2/jerboa-base --interval_chunks 1 --once --public || true
fi
if [ -n "$BACKUP_PID" ]; then
    kill "$BACKUP_PID" 2>/dev/null || true
fi

