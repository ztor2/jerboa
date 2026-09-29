"""Directly initialize and export weights & checkpoints according to DATA_AND_CHECKPOINTS.md

Allows generating complete model weight checkpoints without running full training.
"""

import os
import sys
import json
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.configuration_jerboa import JerboaConfig
from model.modeling_jerboa import JerboaForCausalLM
from model.modeling_multimodal import JerboaVLForConditionalGeneration
from model.tokenizer import get_default_tokenizer


def init_all_checkpoints():
    print("🚀 Initializing tokenizer...")
    tokenizer = get_default_tokenizer()
    vocab_size = len(tokenizer)
    print(f"Tokenizer loaded with vocab_size={vocab_size}")

    print("\n📦 Creating base Jerboa architecture...")
    config = JerboaConfig(vocab_size=vocab_size)
    base_model = JerboaForCausalLM(config)
    num_params = sum(p.numel() for p in base_model.parameters())
    print(f"Base model created with ~{num_params / 1e6:.1f}M parameters")

    stages = [
        ("pretrain", "checkpoints/pretrain/model"),
        ("sft", "checkpoints/sft/model"),
        ("dpo", "checkpoints/dpo/model"),
        ("grpo", "checkpoints/grpo/model"),
    ]

    for stage_name, target_dir in stages:
        print(f"\n💾 Saving [{stage_name.upper()}] checkpoint to '{target_dir}'...")
        os.makedirs(target_dir, exist_ok=True)
        base_model.save_pretrained(target_dir)
        tokenizer.save_pretrained(target_dir)

    # Pretrain steps buffer dir
    os.makedirs("checkpoints/pretrain/steps", exist_ok=True)

    # Multimodal stages
    print("\n📦 Creating Multimodal architecture...")
    mm_output_dir = "checkpoints/multimodal"
    os.makedirs(mm_output_dir, exist_ok=True)

    mm_model = JerboaVLForConditionalGeneration(config)
    
    stage1_path = os.path.join(mm_output_dir, "stage_1.pt")
    torch.save(mm_model.state_dict(), stage1_path)
    print(f"💾 Saved Multimodal Stage 1 checkpoint to '{stage1_path}'")

    stage2_path = os.path.join(mm_output_dir, "stage_2.pt")
    torch.save(mm_model.state_dict(), stage2_path)
    print(f"💾 Saved Multimodal Stage 2 checkpoint to '{stage2_path}'")

    # Multimodal sample data if not present
    mm_sample_path = "data/multimodal/sample.json"
    if not os.path.exists(mm_sample_path):
        os.makedirs(os.path.dirname(mm_sample_path), exist_ok=True)
        sample_mm = [
            {
                "id": "mm_001",
                "image": "data/multimodal/images/scene.jpg",
                "audio": "data/multimodal/audio/clip.wav",
                "conversations": [
                    {"from": "human", "value": "<|image|><|audio|>\nAnalyze this scene."},
                    {"from": "gpt", "value": "The visual and audio cues indicate..."}
                ]
            }
        ]
        with open(mm_sample_path, "w", encoding="utf-8") as f:
            json.dump(sample_mm, f, indent=2, ensure_ascii=False)
        print(f"📄 Created multimodal sample spec at '{mm_sample_path}'")

    # Manifests
    os.makedirs("data/manifests", exist_ok=True)

    print("\n✅ All checkpoints initialized successfully!")


if __name__ == "__main__":
    init_all_checkpoints()
