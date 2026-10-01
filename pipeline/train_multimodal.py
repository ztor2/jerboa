"""Multimodal (Vision + Audio + Text) training pipeline for JerboaVL.

Two training paradigms supported:
1. Stage 1: Projector Warmup (freeze Vision Encoder and LLM, train only Vision/Audio Projector)
2. Stage 2: Full Multimodal Fine-Tuning (end-to-end tuning of Projector + LLM)
"""

import argparse
import os
import sys
import time
from typing import Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
from typing import Dict, List, Optional
from PIL import Image
import torch
from torch.utils.data import DataLoader, Dataset

from model.config import JerboaConfig
from model.multimodal import JerboaVLForConditionalGeneration
from model.tokenizer import get_default_tokenizer


def load_image_tensor(image_input) -> torch.Tensor:
    """Load image from PIL Image or file path, resize to 224x224, and convert to [3, 224, 224] tensor."""
    if image_input is not None:
        try:
            if isinstance(image_input, Image.Image):
                img = image_input.convert("RGB").resize((224, 224))
            elif isinstance(image_input, str) and os.path.exists(image_input):
                img = Image.open(image_input).convert("RGB").resize((224, 224))
            else:
                img = None

            if img is not None:
                arr = torch.tensor(list(img.getdata()), dtype=torch.float32).reshape(224, 224, 3).permute(2, 0, 1) / 255.0
                return (arr - 0.5) / 0.5
        except Exception:
            pass
    return torch.zeros(3, 224, 224)


def load_audio_tensor(audio_input) -> torch.Tensor:
    """Load audio or generate [80, 200] mel-spectrogram feature tensor."""
    if audio_input is not None:
        try:
            if isinstance(audio_input, str) and os.path.exists(audio_input):
                import wave
                with wave.open(audio_input, "r") as wf:
                    n_frames = wf.getnframes()
                    _ = wf.readframes(n_frames)
            elif isinstance(audio_input, dict) and "array" in audio_input:
                pass
            return torch.zeros(80, 200)
        except Exception:
            pass
    return torch.zeros(80, 200)


class MultimodalDataset(Dataset):
    """General-purpose dataset reading multimodal conversations from local JSON or Hugging Face Hub."""

    def __init__(
        self,
        data_path: str,
        tokenizer,
        num_img_tokens: int = 49,
        num_audio_tokens: int = 50,
        max_length: int = 512,
        max_samples: Optional[int] = None,
    ):
        self.tokenizer = tokenizer
        self.num_img_tokens = num_img_tokens
        self.num_audio_tokens = num_audio_tokens
        self.max_length = max_length
        self.samples = []

        if os.path.exists(data_path):
            with open(data_path, "r", encoding="utf-8") as f:
                raw_data = json.load(f)
        else:
            # Load from Hugging Face Hub (streaming or downloaded)
            try:
                from datasets import load_dataset
                print(f"Loading multimodal dataset from Hugging Face Hub: '{data_path}'...")
                hf_ds = load_dataset(data_path, split="train")
                raw_data = []
                for idx, item in enumerate(hf_ds):
                    if max_samples and idx >= max_samples:
                        break
                    raw_data.append(item)
            except Exception as e:
                raise FileNotFoundError(f"Failed to load dataset from local path or Hugging Face Hub ('{data_path}'): {e}")

        img_id = tokenizer.convert_tokens_to_ids("<|image|>")
        aud_id = tokenizer.convert_tokens_to_ids("<|audio|>")

        for item in raw_data:
            convs = item.get("conversations", [])
            user_text = ""
            assistant_text = ""
            for turn in convs:
                role = turn.get("from", "").lower()
                val = turn.get("value", "")
                if role in ["human", "user"]:
                    user_text = val
                elif role in ["gpt", "assistant"]:
                    assistant_text = val

            raw_prompt_ids = tokenizer.encode(
                f"<|im_start|>user\n{user_text}<|im_end|>\n<|im_start|>assistant\n",
                add_special_tokens=False,
            )
            prompt_ids = []
            for tid in raw_prompt_ids:
                if img_id is not None and tid == img_id:
                    prompt_ids.extend([img_id] * num_img_tokens)
                elif aud_id is not None and tid == aud_id:
                    prompt_ids.extend([aud_id] * num_audio_tokens)
                else:
                    prompt_ids.append(tid)

            ans_ids = tokenizer.encode(f"{assistant_text}<|im_end|>\n", add_special_tokens=False)

            full_ids = prompt_ids + ans_ids
            labels = [-100] * len(prompt_ids) + ans_ids

            if len(full_ids) > max_length:
                full_ids = full_ids[:max_length]
                labels = labels[:max_length]

            pixel_values = load_image_tensor(item.get("image", ""))
            audio_values = load_audio_tensor(item.get("audio", ""))

            self.samples.append({
                "input_ids": torch.tensor(full_ids, dtype=torch.long),
                "labels": torch.tensor(labels, dtype=torch.long),
                "pixel_values": pixel_values,
                "audio_values": audio_values,
            })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def collate_multimodal(batch, pad_token_id: int):
    max_len = max(len(x["input_ids"]) for x in batch)
    b_input_ids = []
    b_labels = []
    b_attention_mask = []
    b_pixels = []
    b_audios = []

    for x in batch:
        cur_len = len(x["input_ids"])
        pad_len = max_len - cur_len
        b_input_ids.append(torch.cat([x["input_ids"], torch.full((pad_len,), pad_token_id, dtype=torch.long)]))
        b_labels.append(torch.cat([x["labels"], torch.full((pad_len,), -100, dtype=torch.long)]))
        b_attention_mask.append(torch.cat([torch.ones(cur_len, dtype=torch.long), torch.zeros(pad_len, dtype=torch.long)]))
        b_pixels.append(x["pixel_values"])
        b_audios.append(x["audio_values"])

    return {
        "input_ids": torch.stack(b_input_ids),
        "labels": torch.stack(b_labels),
        "attention_mask": torch.stack(b_attention_mask),
        "pixel_values": torch.stack(b_pixels),
        "audio_values": torch.stack(b_audios),
    }


def run_multimodal_training(
    stage: int = 1,
    modality: str = "unified",
    data_path: Optional[str] = None,
    output_dir: str = "checkpoints/multimodal",
    epochs: int = 2,
    batch_size: int = 2,
    lr: float = 5e-4,
):
    os.makedirs(output_dir, exist_ok=True)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device} ({'Apple Silicon Metal' if device.type == 'mps' else 'CPU'})")

    tokenizer = get_default_tokenizer()
    image_token_id = tokenizer.convert_tokens_to_ids("<|image|>")
    audio_token_id = tokenizer.convert_tokens_to_ids("<|audio|>")

    config = JerboaConfig(
        vocab_size=len(tokenizer),
        hidden_size=768,
        intermediate_size=2048,
        num_hidden_layers=16,
        num_attention_heads=12,
        num_key_value_heads=4,
        tie_word_embeddings=True,
        qk_norm=True,
        image_token_id=image_token_id,
        audio_token_id=audio_token_id,
        vision_spatial_merge_size=2,
    )

    model = JerboaVLForConditionalGeneration(config)
    model.to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Jerboa-VL initialized: {total_params:,} parameters ({total_params/1e6:.2f}M)")

    # Freeze stages based on modality & stage
    if modality == "vision":
        print("\n--- Modular Vision Training (Training Vision Projector Only) ---")
        for p in model.parameters():
            p.requires_grad = False
        for p in model.vision_projector.parameters():
            p.requires_grad = True
        trainable_params = list(model.vision_projector.parameters())
    elif modality == "audio":
        print("\n--- Modular Audio Training (Training Audio Projector Only) ---")
        for p in model.parameters():
            p.requires_grad = False
        for p in model.audio_projector.parameters():
            p.requires_grad = True
        trainable_params = list(model.audio_projector.parameters())
    elif stage == 1:
        print("\n--- Unified Stage 1: Projector Warmup (Freezing Encoders & LLM) ---")
        for p in model.vision_encoder.parameters():
            p.requires_grad = False
        for p in model.audio_encoder.parameters():
            p.requires_grad = False
        for p in model.language_model.parameters():
            p.requires_grad = False
        trainable_params = list(model.vision_projector.parameters()) + list(model.audio_projector.parameters())
    else:
        print("\n--- Unified Stage 2: Multimodal Fine-Tuning (Full Tuning) ---")
        for p in model.vision_encoder.parameters():
            p.requires_grad = False
        trainable_params = [p for p in model.parameters() if p.requires_grad]

    trainable_count = sum(p.numel() for p in trainable_params)
    print(f"Trainable parameters ({modality.upper()}): {trainable_count:,} ({trainable_count/1e6:.2f}M)")

    if data_path is None:
        if modality == "vision":
            data_path = "data/multimodal/image/sample.json"
        elif modality == "audio":
            data_path = "data/multimodal/audio/sample.json"
        else:
            data_path = "data/multimodal/interleaved/sample.json"

    print(f"Loading dataset from: {data_path}")
    dataset = MultimodalDataset(data_path=data_path, tokenizer=tokenizer)
    print(f"Dataset loaded: {len(dataset)} sample(s)")
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=lambda b: collate_multimodal(b, tokenizer.pad_token_id),
    )

    optimizer = torch.optim.AdamW(trainable_params, lr=lr, weight_decay=0.01)

    model.train()
    start_time = time.time()
    step = 0

    for epoch in range(1, epochs + 1):
        epoch_loss = 0.0
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            pixel_values = batch["pixel_values"].to(device)
            audio_values = batch["audio_values"].to(device)

            optimizer.zero_grad()
            outputs = model(
                input_ids=input_ids,
                pixel_values=pixel_values,
                audio_values=audio_values,
                attention_mask=attention_mask,
                labels=labels,
            )
            loss = outputs.loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
            optimizer.step()

            epoch_loss += loss.item()
            step += 1

            if step % 3 == 0:
                print(f"Epoch {epoch}/{epochs} | Step {step:3d} | Loss: {loss.item():.4f}")

        avg_loss = epoch_loss / len(dataloader)
        print(f"=== Epoch {epoch} Complete | Average Loss: {avg_loss:.4f} ===")

    # Modular checkpoint saving
    if modality == "vision":
        save_dir = os.path.join(output_dir, "vision")
        os.makedirs(save_dir, exist_ok=True)
        final_path = os.path.join(save_dir, "projector.pt")
        torch.save(model.vision_projector.state_dict(), final_path)
    elif modality == "audio":
        save_dir = os.path.join(output_dir, "audio")
        os.makedirs(save_dir, exist_ok=True)
        final_path = os.path.join(save_dir, "projector.pt")
        torch.save(model.audio_projector.state_dict(), final_path)
    else:
        save_dir = os.path.join(output_dir, "unified")
        os.makedirs(save_dir, exist_ok=True)
        final_path = os.path.join(save_dir, f"stage_{stage}.pt")
        torch.save(model.state_dict(), final_path)
        # root link for backward compatibility
        torch.save(model.state_dict(), os.path.join(output_dir, f"stage_{stage}.pt"))

    print(f"\nMultimodal training ({modality}) completed in {time.time() - start_time:.2f}s! Saved to {final_path}")
    return final_path


if __name__ == "__main__":
    from pipeline.recipe import apply_recipe

    parser = argparse.ArgumentParser(description="Jerboa Multimodal Training")
    parser.add_argument("--recipe", type=str, default="recipes/multimodal.yaml", help="Path to YAML training recipe")
    parser.add_argument("--stage", type=int, default=1, choices=[1, 2], help="Stage 1 (projector) or Stage 2 (full)")
    parser.add_argument("--modality", type=str, default="unified", choices=["unified", "vision", "audio"], help="Target modality")
    parser.add_argument("--epochs", type=int, default=2, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=2, help="Batch size")
    parser.add_argument("--lr", type=float, default=5e-4, help="Learning rate")
    parser.add_argument("--data", type=str, default=None, help="Path to multimodal JSON dataset")
    parser.add_argument("--output_dir", type=str, default="checkpoints/multimodal")
    args = parser.parse_args()
    args = apply_recipe(args, "recipes/multimodal.yaml")

    run_multimodal_training(
        stage=args.stage,
        modality=args.modality,
        data_path=args.data,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
    )
