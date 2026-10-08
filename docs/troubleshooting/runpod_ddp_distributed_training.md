# RunPod Distributed Pre-Training Troubleshooting & Post-Mortem

This document records the full diagnostic investigation, root causes, false leads, and permanent architectural resolutions encountered while launching JerboaLM (221M Deep & Thin, 28 layers, 49k vocab) distributed pre-training on RunPod multi-GPU instances.

---

## 1. Executive Summary

| Issue | Symptom | Initial Hypothesis / False Lead | Actual Root Cause | Resolution |
| :--- | :--- | :--- | :--- | :--- |
| **1. Host Driver Fault** | `cuInit(0) == 999`, `[Errno 5] Input/output error` on `/dev/nvidia-uvm` | PyTorch/CUDA mismatch or missing packages | Community Cloud host kernel corruption after 48-day uptime without reboot | Switched to RunPod Secure Cloud tier with fresh host driver initialization |
| **2. Flash-Attn Wheel Build** | `[Errno 18] Invalid cross-device link` during `pip install flash-attn` | Broken Python build environment | Host `/tmp` (tmpfs) and persistent mount (`/workspace`) cross-device link failure | Rely on PyTorch 2.4+ built-in `F.scaled_dot_product_attention` (SDPA) FlashAttention-2 kernel |
| **3. tqdm Import Failure** | `ModuleNotFoundError: No module named 'tqdm'` on worker launch | Missing package in system environment (`pip install` into `/usr/local/bin/pip`) | `/usr/local/bin/torchrun` shebang invokes `/usr/local/bin/python`, bypassing active virtualenv | Launch with `python3 -m torch.distributed.run`, ensuring all workers run inside `/workspace/venv` |
| **4. Forward Pass CUDA OOM** | `torch.OutOfMemoryError` allocating 24 MiB at Step 1 forward pass | GPU memory too small (24GB insufficient) | 1) SWA redundant float mask disabled SDPA FlashAttention<br>2) Micro-batch 8 exceeded activation budget<br>3) MTP double-logits allocation | 1) Bypass SWA mask when $q\_len \le \text{window}$<br>2) Reduce micro-batch to 4 ($accum=4$)<br>3) Enable activation gradient checkpointing |
| **5. Checkpoint Metadata Mismatch** | `CheckpointError: Recomputed values have different metadata (4096 vs 2048)` | Non-deterministic layers or rotary embedding cache bug | `use_cache=True` was active during training, appending KV caches again during backward pass | Enforce `use_cache=False` whenever `self.training=True` and in `run_pretrain()` |
| **6. Pod Volume Exhaustion** | `SafetensorError: I/O error: No space left on device (os error 28)` at Chunk 12 | Ephemeral text data accumulation | Every chunk saved full weights (845MB) + optimizer state (1.7GB) = 2.54GB without retention limits | Implemented rolling checkpoint pruning (`save_total_limit: 2`), bounding checkpoint storage to ~6.8GB permanently |

---

## 2. Detailed Technical Investigation

### Case 1: Host UVM Driver I/O Error (`CUDA_ERROR_UNKNOWN`)

#### Symptoms
```
cuda.is_available() -> False
os.open('/dev/nvidia-uvm', os.O_RDWR) -> [Errno 5] Input/output error
```
`nvidia-smi` reported hardware status normally, but any CUDA initialization (`cuInit(0)`) failed immediately with error code 999.

#### Root Cause
On RunPod Community Cloud instance `24r6rql9lxcnvz`, the physical host machine had an uptime of over 48 days. The `nvidia-uvm` (Unified Virtual Memory) kernel module state was corrupted, preventing device file descriptors from being opened.

#### Resolution
Terminated the community pod and provisioned a RunPod **Secure Cloud** instance (`47.47.180.21`, 2x RTX 4090 24GB). Secure Cloud nodes run on dedicated data-center infrastructure with monitored host drivers (Driver 595.91.07, CUDA 13.2).

---

### Case 2: Flash-Attention Wheel Build Failure (`[Errno 18]`)

#### Symptoms
```
error: [Errno 18] Invalid cross-device link: 'flash_attn-2.8.3.post1...whl' -> '/workspace/.cache/pip/wheels/...'
Failed building wheel for flash-attn
```

#### Root Cause
The Dao-AILab wheel build script creates temporary files in `/tmp` (mounted as a `tmpfs` RAM disk) and attempts an `os.rename()` / `link()` to `/workspace/.cache` (mounted as an external persistent network storage block). Linux kernels reject hardlinks across different filesystem mount boundaries (`EXDEV: Invalid cross-device link`).

#### Resolution
PyTorch 2.4+ features native hardware-accelerated SDPA (`torch.nn.functional.scaled_dot_product_attention`), which internally invokes FlashAttention-2 CUDA C++ kernels on Ada Lovelace (SM 8.9) without needing the third-party `flash-attn` Python wheel. Verified with `torch.backends.cuda.flash_sdp_enabled() == True`.

---

### Case 3: Distributed Execution Interpreter Desync (`tqdm` Missing)

#### Symptoms
```
ModuleNotFoundError: No module named 'tqdm'
failed (exitcode: 1) local_rank: 0 (pid: 1257) of binary: /usr/local/bin/python
```

#### Analysis of False Lead vs True Root Cause
- **False Lead**: It appeared that `tqdm` or dependencies were missing from the machine, prompting an attempt to run `/usr/local/bin/pip install -r requirements.txt` to sync both system and virtualenv environments.
- **True Root Cause**: `tqdm>=4.66.0` was already present in `requirements.txt` and installed inside `/workspace/venv`. However, invoking bare `torchrun pipeline/pretrain.py` executed `/usr/local/bin/torchrun`. Its shebang script (`#!/usr/local/bin/python`) launched worker subprocesses using the container's base Python interpreter (`/usr/local/bin/python`), completely ignoring `/workspace/venv`.

#### Permanent Resolution
1. Standardized DDP launcher in [scripts/start_runpod.sh](file:///Users/jc/jerboa/scripts/start_runpod.sh) to execute:
   ```bash
   exec python3 -m torch.distributed.run --nproc_per_node="$NUM_GPUS" pipeline/pretrain.py "$@"
   ```
   When the virtualenv is activated (`source /workspace/venv/bin/activate`), `python3` resolves to `/workspace/venv/bin/python`, ensuring 100% of master and worker processes inherit the virtualenv's site-packages.
2. Removed the redundant `/usr/local/bin/pip` installation step to keep container system paths clean.

---

### Case 4: GPU VRAM Out Of Memory (`torch.OutOfMemoryError`)

#### Symptoms
```
[rank0]: torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 24.00 MiB.
GPU 0 has a total capacity of 23.52 GiB of which 19.75 MiB is free.
Allocated: 22.71 GiB, Reserved: 195.96 MiB.
```

#### Root Cause Breakdown
1. **Activation Memory Scaling**: With 28 layers, hidden dimension 768, and sequence length 2,048:
   - At `batch_size: 8`: forward activations consumed ~13.7 GB across transformer layers.
   - Output projection + MTP auxiliary head: Computing full vocab ($49,152$) cross-entropy logits for both primary tokens and $t+2$ prediction tokens required an additional ~6.2 GB.
   - Total uncheckpointed activations: $>19.7 \text{ GB}$.
2. **SDPA FlashAttention Invalidation by SWA**:
   In [model/modeling.py](file:///Users/jc/jerboa/model/modeling.py), Interleaved Sliding Window Attention (SWA) previously generated an explicit 4D float mask tensor (`swa_mask`) for 21 out of 28 layers:
   ```python
   # Prior logic:
   if self.is_sliding and self.sliding_window is not None:
       ...
       attn_mask = swa_mask
       is_causal = False
   ```
   Passing a non-null float `attn_mask` disqualified PyTorch SDPA from using the fused FlashAttention kernel (`UserWarning: Flash Attention does not support non-null attn_mask`), forcing fallback to memory-intensive attention backends.
   Moreover, when sequence length ($2,048$) $\le$ sliding window ($2,048$), all causally valid positions already fall within the window, making the mask mathematically identical to standard causal masking.

#### Resolutions Applied
1. **Optimized SWA Mask Condition**:
   ```python
   # Updated logic in model/modeling.py:
   if self.is_sliding and self.sliding_window is not None and self.sliding_window < q_len:
   ```
   When sequence length $\le$ sliding window, `attn_mask` remains `None` and `is_causal=True`, engaging the fused FlashAttention-2 kernel across all 28 layers.
2. **Activation Gradient Checkpointing**:
   Enabled `torch.utils.checkpoint.checkpoint(..., use_reentrant=False)` in [model/modeling.py](file:///Users/jc/jerboa/model/modeling.py). Peak activation memory for 28 layers plummeted from **13.7 GB to 0.35 GB**.
3. **Hyperparameter Tuning**:
   In [recipes/pretrain/runpod_4090_ddp.yaml](file:///Users/jc/jerboa/recipes/pretrain/runpod_4090_ddp.yaml), tuned micro-batch to `batch_size: 4` and `grad_accum: 4`.
   $$\text{Global Batch} = 4 \times 2\text{ GPUs} \times 4\text{ accum} = 32\text{ sequences} = 65,536\text{ tokens/step}$$
4. **Allocator Anti-Fragmentation**:
   Set `export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"` in [scripts/start_runpod.sh](file:///Users/jc/jerboa/scripts/start_runpod.sh).

---

### Case 5: Gradient Checkpointing Metadata Mismatch (`CheckpointError`)

#### Symptoms
```
torch.utils.checkpoint.CheckpointError: torch.utils.checkpoint: Recomputed values for the following tensors have different metadata than during the forward pass.
tensor at position 28:
saved metadata: {'shape': torch.Size([4, 12, 2048, 64]), 'dtype': torch.bfloat16}
recomputed metadata: {'shape': torch.Size([4, 12, 4096, 64]), 'dtype': torch.bfloat16}
```

#### Root Cause
In [model/config.py](file:///Users/jc/jerboa/model/config.py), `use_cache` defaults to `True` for inference generation. During training, `past_key_values` was initialized as a `DynamicCache` and appended 2,048 tokens during the initial forward pass. When gradient checkpointing recomputed each layer during the backward pass, the recomputation appended *another* 2,048 tokens to the existing cache, expanding the sequence length from $2,048 \rightarrow 4,096$. Autograd detected the shape mismatch between forward and recomputed passes and halted execution.

#### Resolution
1. Added runtime training check in [model/modeling.py](file:///Users/jc/jerboa/model/modeling.py):
   ```python
   use_cache = use_cache if use_cache is not None else (self.config.use_cache if not self.training else False)
   ```
2. Explicitly enforced `model.config.use_cache = False` upon initialization in [pipeline/pretrain.py](file:///Users/jc/jerboa/pipeline/pretrain.py).

---

### Case 6: Persistent Storage Volume Exhaustion (`No space left on device`)

#### Symptoms
```
safetensors._safetensors_rust.SafetensorError: Error while serializing: I/O error: No space left on device (os error 28)
raw_model.save_pretrained(ckpt_dir) FAILED at Chunk 12
df -h -> /workspace: 30G used (99% full, 416MB free)
```

#### Why Did Volume Exhaustion Happen in "Rolling" Mode?
The project's "Rolling-Buffer Pre-training" paradigm originally targeted data ephemeralization:
- Raw downloaded texts (`data/ephemeral_chunk.txt`) were downloaded, trained on, and immediately deleted after each chunk, keeping data storage under ~11 MB.
- **However, checkpoints were NOT rolled**: In [pipeline/pretrain.py](file:///Users/jc/jerboa/pipeline/pretrain.py), every single chunk saved:
  - Model weights (`model.safetensors`): 845 MB
  - Full AdamW optimizer state (`training_state.pt`): 1.70 GB
  - Total per chunk: **2.54 GB** stored permanently into `checkpoints/pretrain/steps/chunk_XXXX/`
- Across 11 completed chunks: $11 \times 2.54\text{ GB} = 27.94\text{ GB}$.
- Adding `latest_state.pt` (1.70 GB) brought total disk usage to $29.6\text{ GB}$, hitting the 30 GB volume quota at Chunk 12.

#### Resolution
1. **Automated Rolling Checkpoint Retention**:
   Implemented `save_total_limit: 2` in [pipeline/pretrain.py](file:///Users/jc/jerboa/pipeline/pretrain.py) and [recipes/pretrain/runpod_4090_ddp.yaml](file:///Users/jc/jerboa/recipes/pretrain/runpod_4090_ddp.yaml).
   After saving each new chunk, older chunk directories are automatically pruned, keeping only the latest $N$ valid step checkpoints.
   $$\text{Permanent Checkpoint Footprint} = (2 \times 2.54\text{ GB}) + 1.70\text{ GB} \approx 6.78\text{ GB}$$
   This bounds disk usage permanently to $<7\text{ GB}$, leaving $>23\text{ GB}$ safe headroom regardless of how many hundreds of chunks are processed.
2. **Immediate Recovery**:
   Pruned historic checkpoints (`chunk_0001` through `chunk_0010` and incomplete `chunk_0012`), preserving `chunk_0011` intact. Volume space instantly recovered to 26 GB free (17% disk utilization).
3. **Lossless Resumption**:
   Pre-training resumes seamlessly from Chunk 12 using `latest_state.json` and `chunk_0011` weights.

---

### Case 7: Rotary Embedding `inv_freq` Desync & Non-finite Loss Propagation

#### Symptoms
```
Training chunk_0188: 29it [01:04, 2.23s/it, loss=nan, speed=56321 tok/s, lr=3.95e-04]
```
Upon resuming pretraining from disk checkpoint, `loss=nan` appeared immediately.

#### Root Cause
1. `inv_freq` in `JerboaRotaryEmbedding` was registered as a non-persistent buffer (`persistent=False`). When models were re-initialized via `from_pretrained`, Hugging Face's buffer initialization left `inv_freq` populated with uninitialized GPU garbage values (e.g. $7.29 \times 10^{22}$), which blew up RoPE position embeddings in Layer 0.
2. The pretraining loop had no guardrail before `loss.backward()` and `optimizer.step()`, so non-finite losses propagated NaNs directly into optimizer momentums and ruined subsequent weights.

#### Resolution
1. **Dynamic RoPE Frequency Validation**:
   In [model/modeling.py](file:///Users/jc/jerboa/model/modeling.py), `_set_cos_sin_cache` automatically validates `self.inv_freq` and recomputes exact frequencies if the buffer is uninitialized, non-finite, or on meta device.
2. **Non-finite Loss & Gradient Guards**:
   In [pipeline/pretrain.py](file:///Users/jc/jerboa/pipeline/pretrain.py), added `torch.isfinite(loss)` and `torch.isfinite(grad_norm)` checks to immediately skip corrupted batches and zero out gradients before they contaminate optimizer state.

---

## 3. Production Verification & Metrics

Following the applied resolutions, training was launched on the 2x RTX 4090 Secure Cloud instance.

### System & Telemetry Metrics
- **Hardware**: 2x NVIDIA GeForce RTX 4090 (24,564 MiB each)
- **Compute Load (`nvidia-smi`)**:
  - GPU 0: **100% Util**, 254W / 450W power draw
  - GPU 1: **97%~100% Util**, 257W / 450W power draw
- **VRAM Utilization**: **15.9 GB / 24.5 GB** per GPU (8.6 GB safe operating buffer)
- **Training Throughput**: ~54,000 tok/s per GPU $\rightarrow$ **~108,000 tokens/sec aggregate**
- **Loss Progression**: Smooth descent ($14.23 \rightarrow 13.45$ over initial steps)
- **W&B Live Run**: `https://wandb.ai/jerboa/jerboa/runs/6o6hp757`

---

## 4. RunPod Operations Checklist

1. **Provisioning**: Choose **Secure Cloud** for high uptime and verified kernel drivers.
2. **Environment**: Run `bash scripts/setup_runpod.sh` once on pod creation.
3. **Execution**: Always use `bash scripts/start_runpod.sh` inside `tmux`. Never invoke `/usr/local/bin/torchrun` directly.
4. **Resumption**: State cursors and optimizer momentums are saved in `checkpoints/pretrain/`. Restart with `bash scripts/start_runpod.sh --resume auto`.
