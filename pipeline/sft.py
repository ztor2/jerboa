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
    WANDB_AVAILABLE = hasattr(wandb, "init")
except ImportError:
    WANDB_AVAILABLE = False

import torch
from torch.utils.data import DataLoader, Dataset

from model.config import JerboaConfig
from model.modeling import JerboaForCausalLM
from model.tokenizer import get_default_tokenizer
from pipeline.checkpoint_manager import CheckpointManager, GracefulInterruptHandler, SleepGuard, get_device

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

    def __init__(self, data: List[Dict], tokenizer, max_length: int = 768):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.examples = []
        skipped_count = 0

        for item in data:
            messages = item.get("messages")
            if not messages and "conversations" in item:
                messages = [
                    {
                        "role": "user" if c.get("from") in ("human", "user") else ("system" if c.get("from") == "system" else "assistant"),
                        "content": c.get("value", c.get("content", "")),
                    }
                    for c in item["conversations"]
                ]

            if not messages or not any(m.get("role") == "assistant" for m in messages):
                continue

            input_ids = []
            labels = []

            for msg in messages:
                role = msg.get("role", "user")
                content = msg.get("content", "")
                if isinstance(content, list):
                    content = " ".join(
                        str(x.get("value", x.get("text", ""))) if isinstance(x, dict) else str(x)
                        for x in content
                    )
                content = str(content).strip()

                header_text = f"<|im_start|>{role}\n"
                body_text = f"{content}<|im_end|>\n"

                header_tokens = tokenizer.encode(header_text, add_special_tokens=False)
                body_tokens = tokenizer.encode(body_text, add_special_tokens=False)

                seg_tokens = header_tokens + body_tokens
                input_ids.extend(seg_tokens)

                # Prompt headers are always masked (-100)
                labels.extend([-100] * len(header_tokens))

                # Assistant response body receives training loss; all others masked
                if role == "assistant":
                    labels.extend(body_tokens)
                else:
                    labels.extend([-100] * len(body_tokens))

            # Skip sequences exceeding max_length to protect completion token (<|im_end|>) learning
            if len(input_ids) > max_length:
                skipped_count += 1
                continue

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

        if skipped_count > 0:
            print(f"[Dataset] Filtered {skipped_count} samples exceeding max_length ({max_length} tokens)")

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
    max_length: int = 768,
    save_steps: int = 50,
    save_total_limit: int = 3,
    resume: Optional[str] = None,
    use_wandb: bool = False,
    wandb_project: str = "jerboa",
    wandb_run_name: Optional[str] = None,
):
    os.makedirs(output_dir, exist_ok=True)
    device = get_device()
    hw_name = torch.cuda.get_device_name(0) if device.type == "cuda" else ("Apple Silicon Metal" if device.type == "mps" else "CPU")
    print(f"Using device: {device} ({hw_name})")

    checkpoint_mgr = CheckpointManager(output_dir=output_dir, max_to_keep=save_total_limit)
    interrupt_handler = GracefulInterruptHandler()
    sleep_guard = SleepGuard()
    sleep_guard.start()

    # Determine resume checkpoint
    ckpt_to_resume = None
    if resume:
        if str(resume).lower() in ("auto", "true", "1"):
            ckpt_to_resume = checkpoint_mgr.find_latest_checkpoint()
            if ckpt_to_resume:
                print(f"[Resume] Detected latest checkpoint: '{ckpt_to_resume}'")
            else:
                print(f"[Resume] No existing checkpoint found in '{output_dir}'. Starting fresh.")
        elif os.path.exists(str(resume)):
            ckpt_to_resume = str(resume)
            print(f"[Resume] Resuming from specified path: '{ckpt_to_resume}'")
        else:
            print(f"[Resume] Warning: Specified checkpoint '{resume}' does not exist. Starting fresh.")

    # 1. Load Tokenizer & Model
    load_source = ckpt_to_resume if ckpt_to_resume else model_path_or_name
    try:
        print(f"Loading model from '{load_source}' (local or Hugging Face Hub)...")
        model = JerboaForCausalLM.from_pretrained(load_source)
        tokenizer = get_default_tokenizer(load_source)
    except Exception as e:
        print(f"Notice: Could not load from '{load_source}' ({e}), initializing base JerboaLM...")
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
        raw_data = []
        if data_path.endswith(".jsonl"):
            with open(data_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        raw_data.append(json.loads(line))
        else:
            with open(data_path, "r", encoding="utf-8") as f:
                loaded = json.load(f)
                if isinstance(loaded, list):
                    raw_data = loaded
                elif isinstance(loaded, dict) and "data" in loaded:
                    raw_data = loaded["data"]
                else:
                    raw_data = [loaded]
    elif data_path:
        # Load from Hugging Face Hub (streaming or standard split)
        try:
            from datasets import load_dataset
            print(f"Loading SFT dataset from Hugging Face Hub: '{data_path}'...")
            hf_ds = load_dataset(data_path, split="train")
            raw_data = []
            for item in hf_ds:
                if "conversations" in item:
                    raw_data.append(item)
                elif "messages" in item:
                    convs = [{"from": "human" if m.get("role") == "user" else "gpt", "value": m.get("content", "")} for m in item["messages"]]
                    raw_data.append({"conversations": convs})
                elif "instruction" in item and "output" in item:
                    user_val = item["instruction"] + (f"\n{item['input']}" if item.get("input") else "")
                    raw_data.append({"conversations": [
                        {"from": "human", "value": user_val},
                        {"from": "gpt", "value": item["output"]},
                    ]})
        except Exception as e:
            print(f"Could not load '{data_path}' from Hugging Face Hub: {e}. Falling back to default.")
            raw_data = DEFAULT_SFT_EXAMPLES * 8
    else:
        print("Using built-in multi-turn conversation dataset...")
        raw_data = DEFAULT_SFT_EXAMPLES * 8  # Repeat for multiple training batches

    dataset = SFTChatDataset(raw_data, tokenizer, max_length=max_length)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # 3. Optimizer & Scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    total_steps = len(dataloader) * epochs
    warmup_steps = max(10, int(total_steps * 0.05))
    from transformers import get_cosine_schedule_with_warmup
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)

    start_epoch = 1
    global_step = 0
    step_in_epoch = 0

    if ckpt_to_resume:
        state = checkpoint_mgr.restore_state(ckpt_to_resume, optimizer=optimizer, scheduler=scheduler)
        global_step = state.get("step", 0)
        saved_epoch = state.get("epoch", 1)
        saved_step_in_epoch = state.get("step_in_epoch", 0)
        if saved_step_in_epoch >= len(dataloader):
            start_epoch = saved_epoch + 1
            step_in_epoch = 0
        else:
            start_epoch = saved_epoch
            step_in_epoch = saved_step_in_epoch
        print(f"[Resume] Restored state -> Step: {global_step}, Epoch: {start_epoch}, Step in Epoch: {step_in_epoch}")

    # 4. Training Loop
    print(f"\n--- Starting Supervised Fine-Tuning ({epochs} Epochs, {len(dataset)} Samples) ---")
    model.train()
    start_time = time.time()
    interrupted = False

    try:
        for epoch in range(start_epoch, epochs + 1):
            epoch_loss = 0.0
            completed_batches = 0
            for batch_idx, batch in enumerate(dataloader):
                if epoch == start_epoch and batch_idx < step_in_epoch:
                    continue

                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)

                optimizer.zero_grad()
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()

                step_loss = loss.item()
                epoch_loss += step_loss
                completed_batches += 1
                global_step += 1

                if global_step % 10 == 0:
                    current_lr = scheduler.get_last_lr()[0]
                    print(f"Epoch {epoch}/{epochs} | Step {global_step:4d} (Batch {batch_idx + 1}/{len(dataloader)}) | Loss: {step_loss:.4f} | LR: {current_lr:.2e}")

                if use_wandb:
                    wandb.log({
                        "train/loss": step_loss,
                        "train/epoch": epoch,
                        "train/lr": scheduler.get_last_lr()[0],
                        "global_step": global_step,
                    })

                # Periodic Step Checkpoint
                if save_steps > 0 and global_step % save_steps == 0:
                    saved_path = checkpoint_mgr.save(
                        model=model,
                        tokenizer=tokenizer,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        step=global_step,
                        epoch=epoch,
                        step_in_epoch=batch_idx + 1,
                        metrics={"loss": step_loss},
                    )
                    print(f"[Checkpoint] Saved step checkpoint: {saved_path}")

                # Interrupt check (Ctrl+C / SIGINT / SIGTERM)
                if interrupt_handler.interrupted:
                    interrupted = True
                    print(f"\n[Interrupt] Saving emergency checkpoint at step {global_step} (Epoch {epoch})...")
                    saved_path = checkpoint_mgr.save(
                        model=model,
                        tokenizer=tokenizer,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        step=global_step,
                        epoch=epoch,
                        step_in_epoch=batch_idx + 1,
                        metrics={"loss": step_loss},
                        is_interrupted=True,
                    )
                    print(f"[Interrupt] Safely saved to: {saved_path}")
                    print(f"[Interrupt] To resume later, run with: --resume auto (or 'make sft RESUME=auto')")
                    break

            if interrupted:
                break

            avg_loss = epoch_loss / max(completed_batches, 1)
            print(f"=== Epoch {epoch} Complete | Average Loss: {avg_loss:.4f} ===")
            if use_wandb:
                wandb.log({
                    "train/avg_epoch_loss": avg_loss,
                    "epoch": epoch,
                })

            # Save epoch checkpoint
            checkpoint_mgr.save(
                model=model,
                tokenizer=tokenizer,
                optimizer=optimizer,
                scheduler=scheduler,
                step=global_step,
                epoch=epoch,
                step_in_epoch=len(dataloader),
                metrics={"loss": avg_loss},
                tag=f"epoch_{epoch:03d}",
            )
            step_in_epoch = 0
    finally:
        sleep_guard.stop()

    if use_wandb:
        wandb.finish()

    if interrupted:
        return None

    # 5. Save Final Model
    final_path = os.path.join(output_dir, "model")
    model.save_pretrained(final_path)
    tokenizer.save_pretrained(final_path)
    print(f"\nSFT finished in {time.time() - start_time:.2f}s! Saved to {final_path}")
    return final_path


if __name__ == "__main__":
    from pipeline.recipe import apply_recipe

    parser = argparse.ArgumentParser(description="Jerboa Supervised Fine-Tuning (SFT)")
    default_base = "checkpoints/pretrain/model" if os.path.exists("checkpoints/pretrain/model") else "ztor2/jerboa"
    default_data = "data/sft/train.jsonl" if os.path.exists("data/sft/train.jsonl") else ("data/sft/sample.json" if os.path.exists("data/sft/sample.json") else None)
    parser.add_argument("--recipe", type=str, default="recipes/sft.yaml", help="Path to YAML training recipe")
    parser.add_argument("--model", type=str, default=default_base, help="Base model path or HF repo id (default: ztor2/jerboa)")
    parser.add_argument("--epochs", type=int, default=2, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--max_length", type=int, default=768, help="Max sequence length")
    parser.add_argument("--save_steps", type=int, default=100, help="Save intermediate checkpoint every N steps")
    parser.add_argument("--save_total_limit", type=int, default=3, help="Max number of intermediate step checkpoints to keep")
    parser.add_argument("--resume", nargs="?", const="auto", default=None, help="Resume training ('auto' or checkpoint path)")
    parser.add_argument("--output_dir", type=str, default="checkpoints/sft")
    parser.add_argument("--data", type=str, default=default_data, help="JSON or JSONL dataset path")
    parser.add_argument("--wandb", action="store_true", help="Enable Weights & Biases experiment tracking")
    parser.add_argument("--wandb_project", type=str, default="jerboa", help="W&B project name (default: jerboa)")
    parser.add_argument("--wandb_run", type=str, default=None, help="W&B run name")
    args = parser.parse_args()
    args = apply_recipe(args, "recipes/sft.yaml")

    run_sft(
        model_path_or_name=args.model,
        output_dir=args.output_dir,
        data_path=args.data,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        max_length=getattr(args, "max_length", 768),
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        resume=args.resume,
        use_wandb=args.wandb,
        wandb_project=args.wandb_project,
        wandb_run_name=args.wandb_run,
    )
