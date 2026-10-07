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
export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false

WORKSPACE_DIR="/workspace"
if [ -d "$WORKSPACE_DIR" ]; then
    VENV_PATH="$WORKSPACE_DIR/venv"
else
    VENV_PATH="$PROJECT_ROOT/venv"
fi

# 2. Virtual Environment Verification
if [ ! -d "$VENV_PATH" ]; then
    echo "[1/4] Creating persistent venv with system CUDA packages..."
    python3 -m venv --system-site-packages "$VENV_PATH"
fi
source "$VENV_PATH/bin/activate"

# 3. Fast Dependency Sync
echo "[2/4] Ensuring core packages are installed across all interpreters..."
python3 -m pip install -r requirements.txt --quiet
if [ -f "/usr/local/bin/pip" ]; then
    /usr/local/bin/pip install -r requirements.txt --quiet 2>/dev/null || true
fi

# 4. GPU Hardware Detection
echo "[3/4] Detecting GPU Hardware & Topology..."
NUM_GPUS=$(python3 -c "import torch; print(torch.cuda.device_count() if torch.cuda.is_available() else 0)")

if [ "$NUM_GPUS" -eq 0 ]; then
    echo "[!] ERROR: No CUDA GPUs detected. Cannot start GPU pre-training."
    exit 1
fi

GPU_NAME=$(python3 -c "import torch; print(torch.cuda.get_device_name(0))")
echo "[✓] GPU Detected: $NUM_GPUS x $GPU_NAME"

# 5. Launch Training Pipeline
echo "[4/4] Launching Training Pipeline..."
if [ "$NUM_GPUS" -ge 2 ]; then
    RECIPE="recipes/pretrain/runpod_4090_ddp.yaml"
    echo "======================================================================"
    echo " Mode:     Multi-GPU DDP ($NUM_GPUS x $GPU_NAME)"
    echo " Recipe:   $RECIPE"
    echo " Target:   ~110,000 - 120,000 tok/s"
    echo "======================================================================"
    exec python3 -m torch.distributed.run --nproc_per_node="$NUM_GPUS" pipeline/pretrain.py --recipe "$RECIPE" "$@"
else
    RECIPE="recipes/pretrain/runpod_4090.yaml"
    echo "======================================================================"
    echo " Mode:     Single GPU ($GPU_NAME)"
    echo " Recipe:   $RECIPE"
    echo " Target:   ~55,000 - 65,000 tok/s"
    echo "======================================================================"
    exec python3 pipeline/pretrain.py --recipe "$RECIPE" "$@"
fi
