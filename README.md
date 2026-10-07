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
| **Architecture** | 28 Layers, 768 Hidden, SwiGLU, GQA |
| **Parameters** | 221M Base / 228M MTP |
| **Vocabulary** | 49,152 Bilingual BPE |
| **Context Length** | 8,192 tokens (Extensible to 64K) |
| **Active Memory** | ~445 MB BF16 |
| **Target Hardware** | Apple Silicon & NVIDIA CUDA |

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