<div align="center">

<img src="assets/jerboa_avatar.png" width="160" height="160" alt="JerboaLM Avatar" style="border-radius: 50%; box-shadow: 0 4px 16px rgba(0,0,0,0.15);" />

# Jerboa

**Lightweight Multimodal LM**  (in development)

![Python](https://img.shields.io/badge/Python-3.12-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.14_MPS-orange.svg)
![Parameters](https://img.shields.io/badge/Parameters-201M_--_953M-green.svg)
![Memory](https://img.shields.io/badge/Active_VRAM-~420MB_--_2.0GB-purple.svg)
![License](https://img.shields.io/badge/License-Apache_2.0-lightgrey.svg)
[![Hugging Face](https://img.shields.io/badge/🤗%20Hugging%20Face-ztor2%2Fjerboa--base-ffd21e.svg)](https://huggingface.co/ztor2/jerboa-base)

</div>

---

## Model Specifications

| Attribute                   | Specification                                                    |
| :--------------------------- | :---------------------------------------------------------------- |
| Language Backbone       | `JerboaForCausalLM` (~201.4M base / ~208.8M with MTP)             |
| Multimodal Model        | `JerboaVLForConditionalGeneration` (~217.2M with built-in ViT / ~953.2M with EmbeddingGemma-2) |
| Trainable Parameters    | ~201.4M (Base) / ~208.8M (Multimodal with frozen vision backbone) |
| Layers & Hidden Dim     | 28 Layers, $d_{model}=768$, $d_{ffn}=2048$ (SwiGLU)              |
| Attention Configuration | GQA (12 Query Heads : 4 KV Heads), SWA, QK-Norm enabled          |
| Max Context Length      | 4,096 tokens (YaRN scalable to 16K)                               |
| Hardware Target         | Apple Silicon (Mac M-series Metal Performance Shaders / MPS) & CUDA |
| Active VRAM Usage       | ~800 MB (Text FP32) / ~420 MB (Text FP16) / ~2.0 GB (Multimodal FP16) |

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