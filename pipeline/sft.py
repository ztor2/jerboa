"""Supervised Fine-Tuning (SFT) pipeline for JerboaLM.

Features:
- ChatML formatting (<|im_start|>role ... <|im_end|>)
- Prompt loss masking (loss computed ONLY on assistant responses, prompt tokens set to -100)
- Optimized for Apple Silicon MPS
- Checkpoint saving and interactive testing
"""

import argparse
import json
import os
import sys
import time
from typing import Dict, List, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

import torch
from torch.utils.data import DataLoader, Dataset

from model.config import JerboaConfig
from model.modeling import JerboaForCausalLM
from model.tokenizer import get_default_tokenizer

DEFAULT_SFT_EXAMPLES = [
    {
        "messages": [
            {"role": "system", "content": "You are Jerboa, an intelligent and helpful AI assistant."},
            {"role": "user", "content": "What is Python?"},
            {"role": "assistant", "content": "Python is a high-level, general-purpose programming language known for its readability and simplicity."},
        ]
    },
    {
        "messages": [
            {"role": "system", "content": "You are Jerboa, an intelligent and helpful AI assistant."},
            {"role": "user", "content": "Explain grouped-query attention in one sentence."},
            {"role": "assistant", "content": "Grouped-query attention shares key-value heads across multiple query heads, significantly reducing KV cache memory while maintaining model quality."},
        ]
    },
    {
        "messages": [
            {"role": "system", "content": "You are Jerboa, an intelligent and helpful AI assistant."},
            {"role": "user", "content": "Write a quick hello world function in Python."},
            {"role": "assistant", "content": "def hello_world():\n    print('Hello, World!')\n\nhello_world()"},
        ]
    },
    {
        "messages": [
            {"role": "system", "content": "You are Jerboa, an intelligent and helpful AI assistant."},
            {"role": "user", "content": "What is QK-Norm in transformer architecture?"},
            {"role": "assistant", "content": "QK-Norm applies RMSNorm to the query and key vectors before computing attention scores, preventing attention entropy collapse and stabilizing training."},
        ]
    },
]


class SFTChatDataset(Dataset):
    """Dataset that tokenizes chat messages and masks prompt tokens with -100."""

    def __init__(self, data: List[Dict], tokenizer, max_length: int = 512):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.examples = []

        for item in data:
            messages = item["messages"]
            # Build full text using chat template
            full_prompt = ""
            segments = []

            for msg in messages:
                role = msg["role"]
                content = msg["content"]
                segment_text = f"<|im_start|>{role}\n{content}<|im_end|>\n"
                segments.append((role, segment_text))
                full_prompt += segment_text

            # Tokenize segments to create mask
            input_ids = []
            labels = []

            for role, seg_text in segments:
                seg_tokens = tokenizer.encode(seg_text, add_special_tokens=False)
                input_ids.extend(seg_tokens)
                if role == "assistant":
                    # Only train on assistant response tokens
                    labels.extend(seg_tokens)
                else:
                    # Mask prompt tokens
                    labels.extend([-100] * len(seg_tokens))

            # Truncate if exceeds max_length
            if len(input_ids) > max_length:
                input_ids = input_ids[:max_length]
                labels = labels[:max_length]

            # Pad to max_length
            pad_len = max_length - len(input_ids)
            attention_mask = [1] * len(input_ids) + [0] * pad_len
            input_ids = input_ids + [tokenizer.pad_token_id] * pad_len
            labels = labels + [-100] * pad_len

            self.examples.append({
                "input_ids": torch.tensor(input_ids, dtype=torch.long),
                "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
                "labels": torch.tensor(labels, dtype=torch.long),
            })

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]


def run_sft(
    model_path_or_name: Optional[str] = None,
    output_dir: str = "checkpoints/sft",
    data_path: Optional[str] = None,
    epochs: int = 3,
    batch_size: int = 2,
    lr: float = 2e-4,
    max_length: int = 256,
    use_wandb: bool = False,
    wandb_project: str = "jerboa",
    wandb_run_name: Optional[str] = None,
):
    os.makedirs(output_dir, exist_ok=True)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device} ({'Apple Silicon Metal' if device.type == 'mps' else 'CPU'})")

    # 1. Load Tokenizer & Model
    try:
        print(f"Loading pretrained model from '{model_path_or_name}' (local or Hugging Face Hub)...")
        model = JerboaForCausalLM.from_pretrained(model_path_or_name)
        tokenizer = get_default_tokenizer(model_path_or_name)
    except Exception as e:
        print(f"Notice: Could not load from '{model_path_or_name}' ({e}), initializing base JerboaLM...")
        tokenizer = get_default_tokenizer()
        config = JerboaConfig(
            vocab_size=len(tokenizer),
            hidden_size=768,
            intermediate_size=2048,
            num_hidden_layers=16,
            num_attention_heads=12,
            num_key_value_heads=4,
            tie_word_embeddings=True,
            qk_norm=True,
        )
        model = JerboaForCausalLM(config)

    model.to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    if use_wandb:
        if not WANDB_AVAILABLE:
            print("[Warning] wandb is not installed. Continuing with console logging only.")
            use_wandb = False
        else:
            run_name = wandb_run_name or f"sft-{time.strftime('%Y%m%d-%H%M%S')}"
            wandb.init(
                project=wandb_project,
                name=run_name,
                config={
                    "stage": "sft",
                    "base_model": model_path_or_name,
                    "epochs": epochs,
                    "batch_size": batch_size,
                    "lr": lr,
                    "max_length": max_length,
                    "device": str(device),
                    "parameters": sum(p.numel() for p in model.parameters()),
                },
            )

    # 2. Prepare Data
    if data_path and os.path.exists(data_path):
        with open(data_path, "r", encoding="utf-8") as f:
            raw_data = json.load(f)
    else:
        print("Using built-in multi-turn conversation dataset...")
        raw_data = DEFAULT_SFT_EXAMPLES * 8  # Repeat for multiple training batches

    dataset = SFTChatDataset(raw_data, tokenizer, max_length=max_length)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # 3. Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)

    # 4. Training Loop
    print(f"\n--- Starting Supervised Fine-Tuning ({epochs} Epochs, {len(dataset)} Samples) ---")
    model.train()
    start_time = time.time()
    step = 0

    for epoch in range(1, epochs + 1):
        epoch_loss = 0.0
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            optimizer.zero_grad()
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            step_loss = loss.item()
            epoch_loss += step_loss
            step += 1

            if step % 5 == 0:
                print(f"Epoch {epoch}/{epochs} | Step {step:3d} | Loss: {step_loss:.4f}")

            if use_wandb:
                wandb.log({
                    "train/loss": step_loss,
                    "train/epoch": epoch,
                    "global_step": step,
                })

        avg_loss = epoch_loss / len(dataloader)
        print(f"=== Epoch {epoch} Complete | Average Loss: {avg_loss:.4f} ===")
        if use_wandb:
            wandb.log({
                "train/avg_epoch_loss": avg_loss,
                "epoch": epoch,
            })

    if use_wandb:
        wandb.finish()

    # 5. Save Model
    final_path = os.path.join(output_dir, "model")
    model.save_pretrained(final_path)
    tokenizer.save_pretrained(final_path)
    print(f"\nSFT finished in {time.time() - start_time:.2f}s! Saved to {final_path}")
    return final_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Jerboa Supervised Fine-Tuning (SFT)")
    default_base = "checkpoints/pretrain/model" if os.path.exists("checkpoints/pretrain/model") else "ztor2/jerboa"
    default_data = "data/sft/sample.json" if os.path.exists("data/sft/sample.json") else None
    parser.add_argument("--model", type=str, default=default_base, help="Base model path or HF repo id (default: ztor2/jerboa)")
    parser.add_argument("--epochs", type=int, default=2, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=2, help="Batch size")
    parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate")
    parser.add_argument("--output_dir", type=str, default="checkpoints/sft")
    parser.add_argument("--data", type=str, default=default_data, help="JSON dataset path")
    parser.add_argument("--wandb", action="store_true", help="Enable Weights & Biases experiment tracking")
    parser.add_argument("--wandb_project", type=str, default="jerboa", help="W&B project name (default: jerboa)")
    parser.add_argument("--wandb_run", type=str, default=None, help="W&B run name")
    args = parser.parse_args()

    run_sft(
        model_path_or_name=args.model,
        output_dir=args.output_dir,
        data_path=args.data,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        use_wandb=args.wandb,
        wandb_project=args.wandb_project,
        wandb_run_name=args.wandb_run,
    )
