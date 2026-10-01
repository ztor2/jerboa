"""Upload JerboaLM weights and checkpoints to Hugging Face Hub.

Includes automatic remote-code packaging (auto_map) and model card generation.
"""

import argparse
import os
import shutil
import sys
from huggingface_hub import HfApi, whoami


def generate_model_card(repo_id: str, stage: str, num_params_m: float) -> str:
    return f"""---
language:
- en
- ko
license: apache-2.0
tags:
- jerboa
- pytorch
- causal-lm
- edge-ai
- apple-silicon
pipeline_tag: text-generation
---

# {repo_id}

**JerboaLM** is an ultra-lightweight language model (~{num_params_m:.1f}M parameters) optimized for Apple Silicon (MPS / Metal) and edge deployments.

- **Stage**: {stage.upper()}
- **Parameters**: ~{num_params_m:.1f}M
- **Context Length**: 4,096 tokens (extensible to 16K via YaRN)
- **Attention**: Grouped-Query Attention (GQA 3:1) + Interleaved Sliding Window Attention (SWA)
- **Multi-Token Prediction**: Enabled (MTP $t+2$ auxiliary head)
- **Special Tokens**: Native support for `<think>`, `</think>`, `<answer>`, `<tool_call>`, `<|im_start|>`, `<|im_end|>`

---

## Quickstart

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

repo_id = "{repo_id}"

tokenizer = AutoTokenizer.from_pretrained(repo_id, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    repo_id,
    torch_dtype=torch.float32,
    device_map="auto",
    trust_remote_code=True
)

prompt = "<|im_start|>user\\nExplain grouped-query attention.<|im_end|>\\n<|im_start|>assistant\\n"
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

outputs = model.generate(**inputs, max_new_tokens=128, temperature=0.7)
print(tokenizer.decode(outputs[0], skip_special_tokens=False))
```

---

## Architecture Specifications

| Attribute | Specification |
| :--- | :--- |
| **Hidden Dim** | 768 |
| **FFN Intermediate Dim** | 2048 (SwiGLU) |
| **Layers** | 16 |
| **Heads** | 12 Query / 4 KV (GQA) |
| **Vocabulary Size** | 49,164 tokens |
| **Target Hardware** | Apple Silicon MPS / Edge Devices |
"""


def upload_checkpoint(
    checkpoint_dir: str = "checkpoints/sft/model",
    repo_name: str = "jerboa-sft",
    private: bool = False,
    stage: str = "sft",
    include_manifests: bool = True,
):
    if not os.path.exists(checkpoint_dir):
        print(f"❌ Checkpoint directory '{checkpoint_dir}' does not exist.")
        sys.exit(1)

    # 1. Check HF Authentication
    try:
        user_info = whoami()
        username = user_info["name"]
        print(f"🔑 Logged in as Hugging Face user: '{username}'")
    except Exception as e:
        print(f"❌ Not logged in to Hugging Face: {e}")
        print("💡 Please run: huggingface-cli login")
        sys.exit(1)

    repo_id = f"{username}/{repo_name}"
    print(f"\n📦 Preparing model files for '{repo_id}'...")

    # 2. Bundle remote code for Hugging Face Hub trust_remote_code
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    model_dir = os.path.join(project_root, "model")

    for filename in ["config.py", "modeling.py"]:
        src = os.path.join(model_dir, filename)
        dst = os.path.join(checkpoint_dir, filename)
        shutil.copy2(src, dst)
        print(f"  └ Copied remote code: {filename}")

    # 3. Bundle data manifests if present
    if include_manifests:
        manifests_src = os.path.join(project_root, "data", "manifests")
        if os.path.exists(manifests_src) and os.listdir(manifests_src):
            manifests_dst = os.path.join(checkpoint_dir, "manifests")
            os.makedirs(manifests_dst, exist_ok=True)
            for item in os.listdir(manifests_src):
                s = os.path.join(manifests_src, item)
                d = os.path.join(manifests_dst, item)
                if os.path.isfile(s):
                    shutil.copy2(s, d)
                elif os.path.isdir(s):
                    shutil.copytree(s, d, dirs_exist_ok=True)
            print(f"  └ Bundled data provenance manifests from 'data/manifests'")

    # 4. Create Model Card README.md if not present
    readme_path = os.path.join(checkpoint_dir, "README.md")
    if not os.path.exists(readme_path):
        card_content = generate_model_card(repo_id, stage, 145.9)
        with open(readme_path, "w", encoding="utf-8") as f:
            f.write(card_content)
        print("  └ Generated Model Card: README.md")

    # 4. Upload via HfApi
    api = HfApi()
    print(f"\n🚀 Creating repository (if not exists): {repo_id} (private={private})")
    api.create_repo(repo_id=repo_id, repo_type="model", private=private, exist_ok=True)

    print(f"📤 Uploading folder '{checkpoint_dir}' to Hugging Face Hub...")
    api.upload_folder(
        folder_path=checkpoint_dir,
        repo_id=repo_id,
        repo_type="model",
    )

    print(f"\n🎉 Successfully uploaded to Hugging Face Hub!")
    print(f"🔗 URL: https://huggingface.co/{repo_id}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Upload JerboaLM checkpoints to Hugging Face Hub")
    parser.add_argument("--model", type=str, default="checkpoints/sft/model", help="Path to checkpoint folder")
    parser.add_argument("--repo_name", type=str, default="jerboa-sft", help="Repository name on HF Hub")
    parser.add_argument("--private", action="store_true", help="Upload as a private repository")
    parser.add_argument("--stage", type=str, default="sft", choices=["pretrain", "sft", "dpo", "grpo"], help="Training stage")
    parser.add_argument("--no_manifests", action="store_true", help="Do not bundle data manifests")
    args = parser.parse_args()

    upload_checkpoint(
        checkpoint_dir=args.model,
        repo_name=args.repo_name,
        private=args.private,
        stage=args.stage,
        include_manifests=not args.no_manifests,
    )
