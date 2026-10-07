<div align="center">

<img src="assets/jerboa_avatar.png" width="160" height="160" alt="JerboaLM Avatar" style="border-radius: 50%; box-shadow: 0 4px 16px rgba(0,0,0,0.15);" />

# Jerboa

**Lightweight Multimodal LM**  (in development)

![Python](https://img.shields.io/badge/Python-3.12-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.14_MPS-orange.svg)
![Parameters](https://img.shields.io/badge/Parameters-133M_--_885M-green.svg)
![Memory](https://img.shields.io/badge/Active_VRAM-~500MB_--_1.8GB-purple.svg)
![License](https://img.shields.io/badge/License-Apache_2.0-lightgrey.svg)
[![Hugging Face](https://img.shields.io/badge/🤗%20Hugging%20Face-ztor2%2Fjerboa--base-ffd21e.svg)](https://huggingface.co/ztor2/jerboa-base)

</div>

---

## Model Specifications

| Attribute                   | Specification                                                    |
| :--------------------------- | :---------------------------------------------------------------- |
| Language Backbone       | `JerboaForCausalLM` (~133.3M base / ~140.8M with MTP)             |
| Multimodal Model        | `JerboaVLForConditionalGeneration` (~141.7M with built-in ViT / ~885.2M with EmbeddingGemma-2) |
| Trainable Parameters    | ~133.3M (Base) / ~140.8M (Multimodal with frozen vision backbone) |
| Layers & Hidden Dim     | 16 Layers, $d_{model}=768$, $d_{ffn}=2048$ (SwiGLU)              |
| Attention Configuration | GQA (12 Query Heads : 4 KV Heads), SWA, QK-Norm enabled          |
| Max Context Length      | 4,096 tokens (YaRN scalable to 16K)                               |
| Hardware Target         | Apple Silicon (Mac M-series Metal Performance Shaders / MPS) & CUDA |
| Active VRAM Usage       | ~487 MB (Text FP32) / ~270 MB (Text FP16) / ~1.8 GB (Multimodal FP16) |

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