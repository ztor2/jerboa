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

    # Multimodal stages (Modular & Unified)
    print("\n📦 Creating Multimodal architecture...")
    mm_model = JerboaVLForConditionalGeneration(config)

    # 1. Vision Projector module
    vision_dir = "checkpoints/multimodal/vision"
    os.makedirs(vision_dir, exist_ok=True)
    vision_proj_path = os.path.join(vision_dir, "projector.pt")
    torch.save(mm_model.vision_projector.state_dict(), vision_proj_path)
    print(f"💾 Saved Vision Projector module to '{vision_proj_path}'")

    # 2. Audio Projector module
    audio_dir = "checkpoints/multimodal/audio"
    os.makedirs(audio_dir, exist_ok=True)
    audio_proj_path = os.path.join(audio_dir, "projector.pt")
    torch.save(mm_model.audio_projector.state_dict(), audio_proj_path)
    print(f"💾 Saved Audio Projector module to '{audio_proj_path}'")

    # 3. Unified End-to-End checkpoints
    unified_dir = "checkpoints/multimodal/unified"
    os.makedirs(unified_dir, exist_ok=True)
    stage1_path = os.path.join(unified_dir, "stage_1.pt")
    torch.save(mm_model.state_dict(), stage1_path)
    print(f"💾 Saved Unified Stage 1 checkpoint to '{stage1_path}'")

    stage2_path = os.path.join(unified_dir, "stage_2.pt")
    torch.save(mm_model.state_dict(), stage2_path)
    print(f"💾 Saved Unified Stage 2 checkpoint to '{stage2_path}'")

    # Keep root compatibility links/files
    torch.save(mm_model.state_dict(), "checkpoints/multimodal/stage_1.pt")
    torch.save(mm_model.state_dict(), "checkpoints/multimodal/stage_2.pt")

    # Multimodal Granular Dataset Samples
    mm_datasets = {
        "data/multimodal/image/sample.json": [
            {
                "id": "img_001",
                "image": "data/multimodal/image/sample.jpg",
                "conversations": [
                    {"from": "human", "value": "<|image|>\nDescribe this scene in detail."},
                    {"from": "gpt", "value": "A serene landscape with clear skies and green hills."}
                ]
            }
        ],
        "data/multimodal/audio/sample.json": [
            {
                "id": "aud_001",
                "audio": "data/multimodal/audio/sample.wav",
                "conversations": [
                    {"from": "human", "value": "<|audio|>\nTranscribe and analyze this sound clip."},
                    {"from": "gpt", "value": "Ambient birds chirping in a quiet forest environment."}
                ]
            }
        ],
        "data/multimodal/video/sample.json": [
            {
                "id": "vid_001",
                "video": "data/multimodal/video/sample.mp4",
                "num_frames": 8,
                "conversations": [
                    {"from": "human", "value": "<|image|><|image|><|image|><|image|>\nSummarize the action in this video clip."},
                    {"from": "gpt", "value": "A high-speed train accelerates out of the station across the bridge."}
                ]
            }
        ],
        "data/multimodal/interleaved/sample.json": [
            {
                "id": "intl_001",
                "image": "data/multimodal/interleaved/scene.jpg",
                "audio": "data/multimodal/interleaved/clip.wav",
                "conversations": [
                    {"from": "human", "value": "<|image|><|audio|>\nHow does the audio relate to the visual scene?"},
                    {"from": "gpt", "value": "The visual cues show ocean waves, corresponding to the crashing wave sounds."}
                ]
            }
        ]
    }

    for path, sample_data in mm_datasets.items():
        if not os.path.exists(path):
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "w", encoding="utf-8") as f:
                json.dump(sample_data, f, indent=2, ensure_ascii=False)
            print(f"📄 Created dataset spec at '{path}'")

    # Manifests
    os.makedirs("data/manifests", exist_ok=True)

    print("\n✅ All checkpoints initialized successfully!")


if __name__ == "__main__":
    init_all_checkpoints()
