# RunPod Execution & Operations Guide

A practical operational manual for pre-training and managing JerboaLM instances on RunPod.

---

## 1. Quick Start Execution

### Step 1: Repository Sync & Environment Setup
```bash
cd /workspace/jerboa
git pull
bash scripts/setup_runpod.sh
source /workspace/venv/bin/activate
```

### Step 2: 2-Step Sanity Check (Smoke Test)
Verify GPU drivers, CUDA kernels, and data streaming in ~5 seconds:
```bash
python pipeline/pretrain.py --recipe recipes/pretrain/smoke_test.yaml --no_wandb
```

### Step 3: Launch Pre-training in Background (`tmux`)
Keep training active 24/7 even after disconnecting terminal:
```bash
tmux new -s pretrain
source /workspace/venv/bin/activate

# Single GPU (RTX 4090 / 3090)
python pipeline/pretrain.py --recipe recipes/pretrain/phase1_base.yaml

# Dual GPU (2x 3090 DDP)
torchrun --nproc_per_node=2 pipeline/pretrain.py --recipe recipes/pretrain/runpod_ddp.yaml
```

### Step 4: Detach Session
- Press `Ctrl + B`, release, then press `D`.
- Safely close the web browser; training continues unattended.

---

## 2. Real-Time Monitoring & Metrics

### Reconnecting to Session
```bash
tmux attach -t pretrain
```

### Key Health Metrics
- **GPU Utilization (`nvidia-smi`)**: 90%–100% compute load.
- **VRAM Headroom**: ~8 GB to 12 GB / 24 GB (plenty of safety margin).
- **Initial Loss**: Starts at $\sim 10.8$ ($-\ln(1/49152)$ theoretical random baseline).
- **Convergence Target**:
  - Step 100: $\sim 7.0 \rightarrow 5.5$
  - Step 1,000+: $\sim 3.5 \rightarrow 2.8$
- **Expected Throughput (Single 4090)**: ~45,000 to ~65,000 tokens/sec.

---

## 3. Emergency Troubleshooting

### T1: Terminal Disconnected or Browser Closed
The process is running inside `tmux`. Reopen Web Terminal and run:
```bash
tmux attach -t pretrain
```

### T2: WandB Interactive Prompt Blocking Start
Add `--no_wandb` to bypass cloud login and stream logs to stdout:
```bash
python pipeline/pretrain.py --recipe recipes/pretrain/phase1_base.yaml --no_wandb
```

### T3: Out Of Memory (OOM)
Reduce micro-batch size while increasing gradient accumulation to maintain effective batch:
```bash
python pipeline/pretrain.py --recipe recipes/pretrain/phase1_base.yaml --batch_size 2 --grad_accum 8
```

### T4: Pausing & Resuming Training
- **Pause**: Enter `tmux attach -t pretrain`, press `Ctrl + C` (state saved to `checkpoints/pretrain/`). Then click **Stop Pod** on RunPod dashboard.
- **Resume**: Click **Start Pod**, open terminal, and run:
  ```bash
  cd /workspace/jerboa
  source /workspace/venv/bin/activate
  python pipeline/pretrain.py --recipe recipes/pretrain/phase1_base.yaml --resume auto
  ```

---

## 4. Post-Training Roadmap

```
Phase 1 Base (10B) ──▶ Smoke Inference Test ──▶ Phase 2 Long Context (8K) ──▶ SFT / VLM ──▶ Backup Weights
```

1. **Inference Test**:
   ```bash
   python scripts/infer/text.py --model_path checkpoints/pretrain/latest
   ```
2. **Phase 2 Long Context Extension**:
   ```bash
   python pipeline/pretrain.py --recipe recipes/pretrain/phase2_long.yaml --resume checkpoints/pretrain/phase1_final
   ```
3. **Weight Backup & Teardown**:
   - Push checkpoint to Hugging Face Hub (`huggingface-cli upload`) or download `model.safetensors` (~440MB).
   - Terminate the Pod on RunPod dashboard to stop storage billing.
