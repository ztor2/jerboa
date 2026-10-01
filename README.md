<div align="center">

<img src="assets/jerboa_avatar.png" width="160" height="160" alt="JerboaLM Avatar" style="border-radius: 50%; box-shadow: 0 4px 16px rgba(0,0,0,0.15);" />

# Jerboa

**Lightweight Multimodal Language Model**  (in development)

![Python](https://img.shields.io/badge/Python-3.12-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.14_MPS-orange.svg)
![Parameters](https://img.shields.io/badge/Parameters-125M_--_145M-green.svg)
![Memory](https://img.shields.io/badge/Active_VRAM-~500MB-purple.svg)
![License](https://img.shields.io/badge/License-Apache_2.0-lightgrey.svg)
[![Hugging Face](https://img.shields.io/badge/🤗%20Hugging%20Face-ztor2%2Fjerboa-ffd21e.svg)](https://huggingface.co/ztor2/jerboa)

</div>

---

## Model Specifications

| Attribute                   | Specification                                                    |
| :--------------------------- | :---------------------------------------------------------------- |
| Language Backbone       | `JerboaForCausalLM` (~138.4M base / ~145.9M with MTP)             |
| Multimodal Model        | `JerboaVLForConditionalGeneration` (~145.2M)                     |
| Layers & Hidden Dim     | 16 Layers, $d_{model}=768$, $d_{ffn}=2048$ (SwiGLU)              |
| Attention Configuration | GQA (12 Query Heads : 4 KV Heads), SWA, QK-Norm enabled          |
| Max Context Length      | 4,096 tokens (YaRN scalable to 16K)                               |
| Hardware Target         | Apple Silicon (Mac M-series Metal Performance Shaders / MPS)     |
| Active VRAM Usage       | ~487.1 MB (Text) / ~600 MB (Multimodal)                          |

---

## Model & Checkpoints

Model weights and tokenizer are hosted on the Hugging Face Hub:
- 🤗 **Hugging Face Repository**: [ztor2/jerboa](https://huggingface.co/ztor2/jerboa)

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "ztor2/jerboa"
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True)
```

---

## Documentation

- **[Model Architecture & Design](docs/model.md)**: Specifications, GQA, SWA, MTP, YaRN, and Multimodal encoder details.
- **[Training & Data Lifecycle](docs/train.md)**: Symmetric data mapping, rolling pre-training, SFT, DPO, GRPO, and Mac long-running guide.
- **[Inference & Deployment](docs/infer.md)**: Streaming chat REPL, multimodal reasoning, benchmarking, and quantization.

