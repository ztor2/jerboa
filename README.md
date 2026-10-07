<div align="center">

<img src="assets/jerboa_avatar.png" width="160" height="160" alt="JerboaLM Avatar" style="border-radius: 50%; box-shadow: 0 4px 16px rgba(0,0,0,0.15);" />

# Jerboa

In Development

![Python](https://img.shields.io/badge/Python-3.12-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.4+_MPS_%26_CUDA-orange.svg)
![Parameters](https://img.shields.io/badge/Parameters-221M_--_966M-green.svg)
![Context](https://img.shields.io/badge/Context-8K_--_64K-blueviolet.svg)
![Tokenizer](https://img.shields.io/badge/Tokenizer-49.1K_Bilingual_BPE-informational.svg)
![Memory](https://img.shields.io/badge/Active_VRAM-~445MB_--_2.1GB-purple.svg)
![License](https://img.shields.io/badge/License-Apache_2.0-lightgrey.svg)
![Hugging Face](https://img.shields.io/badge/🤗%20Hugging%20Face-ztor2%2Fjerboa--base-ffd21e.svg)

</div>

---

## Model Specifications

| Attribute | Specification |
| :--- | :--- |
| **Language Backbone** | `JerboaForCausalLM` (~221.4M base / ~228.9M with MTP) |
| **Multimodal Model** | `JerboaVLForConditionalGeneration` (~229.8M built-in ViT / ~965.8M with EmbeddingGemma-2) |
| **Trainable Parameters** | ~221.4M (Base) / ~228.9M (with MTP) |
| **Layers & Hidden Dim** | 28 Layers (Deep & Thin), $d_{model}=768$, $d_{ffn}=2048$ (SwiGLU) |
| **Attention Mechanism** | GQA (12 Query : 4 KV Heads), Interleaved SWA (2048 window), QK-Norm |
| **Tokenizer** | 49,152 vocab Byte-Level BPE (Korean, English, Code, ChatML format) |
| **Max Context Length** | **8,192 tokens native** ($\theta=500,000.0$, YaRN scalable to **32K ~ 64K**) |
| **Hardware Target** | Apple Silicon (MPS / Metal) & NVIDIA CUDA (RTX 3090 / 4090 / 5090) |
| **Active VRAM Usage** | ~885 MB (FP32) / ~445 MB (FP16/BF16) / ~2.1 GB (Multimodal FP16) |

---

## Documentation

- [docs/model.md](docs/model.md): Deep & Thin architecture, GQA, SWA, MTP mathematical formulation
- [docs/tokenizer.md](docs/tokenizer.md): 49k BPE tokenizer architecture, corpus breakdown, fertility benchmarks
- [docs/research.md](docs/research.md): Empirical design rationales, SOTA precedents (Llama 3, SmolLM, DoReMi)
- [docs/pretrain_data_plan.md](docs/pretrain_data_plan.md): 10B multilingual token mixture and curriculum strategy
- [docs/train.md](docs/train.md): Pre-training, SFT, DPO, GRPO, and multimodal pipeline guides

---

## Model & Checkpoints

- 🤗 **Repository**: [ztor2/jerboa-base](https://huggingface.co/ztor2/jerboa-base)

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "ztor2/jerboa-base"
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True)
```

---