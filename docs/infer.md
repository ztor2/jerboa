# Inference & Deployment

Evaluation, streaming text generation, multimodal interaction, benchmarking, and post-training compression.

---

## 1. Interactive Terminal CLI (`cli/`) & Text Inference

`cli/` provides a dedicated terminal console (`rich` + `prompt_toolkit`) with live streaming panels, bottom status bar telemetry (`tok/s`, `TTFT`, token count), and slash-command autocompletion (`/mode`, `/model`, `/temp`, `/preset`, `/clear`, `/status`, `/help`).

```bash
# Launch interactive terminal CLI (auto-detects base vs chat mode from checkpoint path)
make cli
# or directly:
python -m cli --model checkpoints/pretrain/model

# Single-shot streaming generation via CLI module
python -m cli --model checkpoints/pretrain/model --prompt "def binary_search(arr: list[int], target: int) -> int:"
```

For lightweight script-based generation without TUI formatting, `scripts/infer/text.py` is also available:

```bash
# Single prompt generation
python scripts/infer/text.py --model checkpoints/sft/model --prompt "Explain grouped-query attention."

# Interactive terminal chat REPL
python scripts/infer/text.py --model checkpoints/sft/model --chat

# Custom generation parameters
python scripts/infer/text.py --model checkpoints/sft/model --chat --temperature 0.3 --max_tokens 256
```

---

## 2. Multimodal Inference

`scripts/infer/multimodal.py` handles vision, audio, and interleaved multimodal reasoning:

```bash
# Vision only
python scripts/infer/multimodal.py --image assets/scene.jpg --prompt "Describe the visual details."

# Audio only
python scripts/infer/multimodal.py --audio assets/speech.wav --prompt "Transcribe and summarize this audio."

# Joint vision + audio reasoning
python scripts/infer/multimodal.py --image assets/scene.jpg --audio assets/sound.wav --prompt "How does the audio relate to the visual scene?"

# Modular checkpoint loading (Vision projector only)
python scripts/infer/multimodal.py --checkpoint checkpoints/multimodal/vision/projector.pt --image assets/scene.jpg --prompt "Describe this."

# Unified end-to-end checkpoint loading
python scripts/infer/multimodal.py --checkpoint checkpoints/multimodal/unified/stage_2.pt --image assets/scene.jpg --audio assets/sound.wav --prompt "Analyze scene."
```

---

## 3. Stage Evaluation & Hardware Benchmarking

Stage-specific evaluation suites live under `tests/eval/` (e.g., `pretrain.py`, with future `sft.py` and `rl.py` modules), while `tests/benchmark.py` measures hardware throughput on Apple Silicon (MPS).

```bash
# Pre-training stage evaluation (Held-out Perplexity, Bits-Per-Byte & Zero-Shot Log-Likelihood MC)
python tests/eval/pretrain.py --model ztor2/jerboa-base

# Or via Makefile shortcut
make eval-pretrain

# Standard hardware throughput benchmark (prompt: 128 tokens, generation: 64 tokens)
python tests/benchmark.py --prompt_len 128 --gen_len 64

# Long-context benchmark
python tests/benchmark.py --prompt_len 512 --gen_len 128
```

---

## 4. Quantization & Compression

`scripts/tools/quantize.py` converts checkpoints for reduced memory bandwidth and edge deployment. Original checkpoints remain full-precision.

```bash
# Convert to FP16 (50% memory reduction, runs on Apple Silicon Metal)
python scripts/tools/quantize.py --model checkpoints/sft/model --precision fp16

# Convert to BF16 (bfloat16 format)
python scripts/tools/quantize.py --model checkpoints/sft/model --precision bf16

# Convert to INT8 dynamic quantization (~70% memory reduction)
python scripts/tools/quantize.py --model checkpoints/sft/model --precision int8

# Run chat REPL with quantized model
python scripts/infer/text.py --model checkpoints/sft/model_fp16 --chat
```
