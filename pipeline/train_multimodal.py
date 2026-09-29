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

import torch
from torch.utils.data import DataLoader, Dataset

from model.configuration_jerboa import JerboaConfig
from model.modeling_multimodal import JerboaVLForConditionalGeneration
from model.tokenizer import get_default_tokenizer


class SyntheticMultimodalDataset(Dataset):
    """Dataset producing image pixels, audio mel-spectrograms, and conversational text."""

    def __init__(self, tokenizer, num_samples: int = 50, num_img_tokens: int = 49):
        self.samples = []
        self.num_img_tokens = num_img_tokens

        img_placeholder_tokens = [tokenizer.convert_tokens_to_ids("<|image|>")] * num_img_tokens

        sample_qa = [
            ("Describe what is shown in this image.", "This image displays a serene mountain landscape with a calm lake at sunrise."),
            ("What colors are prominent in the picture?", "The prominent colors are deep blue, golden orange, and forest green."),
            ("Can you identify the main object?", "The main subject is a high-speed train traveling along a coastal bridge."),
        ]

        for i in range(num_samples):
            q, a = sample_qa[i % len(sample_qa)]
            prompt_str = f"<|im_start|>user\n"
            q_ids = tokenizer.encode(prompt_str, add_special_tokens=False) + img_placeholder_tokens + tokenizer.encode(f"\n{q}<|im_end|>\n<|im_start|>assistant\n", add_special_tokens=False)
            ans_ids = tokenizer.encode(f"{a}<|im_end|>\n", add_special_tokens=False)

            full_ids = q_ids + ans_ids
            labels = [-100] * len(q_ids) + ans_ids

            self.samples.append({
                "input_ids": torch.tensor(full_ids, dtype=torch.long),
                "labels": torch.tensor(labels, dtype=torch.long),
            })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        return {
            "input_ids": item["input_ids"],
            "labels": item["labels"],
            "pixel_values": torch.randn(3, 224, 224),
            "audio_values": torch.randn(80, 200),
        }


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

    dataset = SyntheticMultimodalDataset(tokenizer, num_samples=30)
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
    parser = argparse.ArgumentParser(description="Jerboa Multimodal Training")
    parser.add_argument("--stage", type=int, default=1, choices=[1, 2], help="Stage 1 (projector) or Stage 2 (full)")
    parser.add_argument("--modality", type=str, default="unified", choices=["unified", "vision", "audio"], help="Target modality")
    parser.add_argument("--epochs", type=int, default=2, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=2, help="Batch size")
    parser.add_argument("--lr", type=float, default=5e-4, help="Learning rate")
    parser.add_argument("--output_dir", type=str, default="checkpoints/multimodal")
    args = parser.parse_args()

    run_multimodal_training(
        stage=args.stage,
        modality=args.modality,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
    )
