# Inference & Deployment

Evaluation, streaming text generation, multimodal interaction, benchmarking, and post-training compression.

---

## 1. Text Inference & Interactive Chat

`scripts/infer.py` provides real-time token streaming (`TextStreamer`) and multi-turn REPL capabilities.

```bash
# Single prompt generation
python scripts/infer.py --model checkpoints/sft/model --prompt "Explain grouped-query attention."

# Interactive terminal chat REPL
python scripts/infer.py --model checkpoints/sft/model --chat

# Custom generation parameters
python scripts/infer.py --model checkpoints/sft/model --chat --temperature 0.3 --max_tokens 256
```

---

## 2. Multimodal Inference

`scripts/infer_multimodal.py` handles vision, audio, and interleaved multimodal reasoning:

```bash
# Vision only
python scripts/infer_multimodal.py --image assets/scene.jpg --prompt "Describe the visual details."

# Audio only
python scripts/infer_multimodal.py --audio assets/speech.wav --prompt "Transcribe and summarize this audio."

# Joint vision + audio reasoning
python scripts/infer_multimodal.py --image assets/scene.jpg --audio assets/sound.wav --prompt "How does the audio relate to the visual scene?"
```

---

## 3. Benchmarking

`scripts/benchmark.py` measures prefill throughput, autoregressive decoding speed, and active VRAM usage on Apple Silicon (MPS).

```bash
# Standard benchmark (prompt: 128 tokens, generation: 64 tokens)
python scripts/benchmark.py --prompt_len 128 --gen_len 64

# Long-context benchmark
python scripts/benchmark.py --prompt_len 512 --gen_len 128
```

---

## 4. Quantization & Compression

`scripts/quantize.py` converts checkpoints for reduced memory bandwidth and edge deployment. Original checkpoints remain full-precision.

```bash
# Convert to FP16 (50% memory reduction, runs on Apple Silicon Metal)
python scripts/quantize.py --model checkpoints/sft/model --precision fp16

# Convert to BF16 (bfloat16 format)
python scripts/quantize.py --model checkpoints/sft/model --precision bf16

# Convert to INT8 dynamic quantization (~70% memory reduction)
python scripts/quantize.py --model checkpoints/sft/model --precision int8

# Run chat REPL with quantized model
python scripts/infer.py --model checkpoints/sft/model_fp16 --chat
```
