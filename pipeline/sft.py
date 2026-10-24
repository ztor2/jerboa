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
from pipeline.checkpoint_manager import CheckpointManager, GracefulInterruptHandler, SleepGuard, SystemResourceGuard, get_device

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

    def __init__(self, data: List[Dict], tokenizer, max_length: int = 2048):
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

            self.examples.append({
                "input_ids": torch.tensor(input_ids, dtype=torch.long),
                "attention_mask": torch.ones(len(input_ids), dtype=torch.long),
                "labels": torch.tensor(labels, dtype=torch.long),
            })

        if skipped_count > 0:
            print(f"[Dataset] Filtered {skipped_count} samples exceeding max_length ({max_length} tokens)")

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]


class SFTDynamicCollator:
    """Pad each batch dynamically to a coarse bucket (`1024` or `max_length`) to prevent MPSGraph shape-compilation cache bloat."""

    def __init__(self, pad_token_id: int = 0, max_length: int = 2048, buckets: tuple[int, ...] = (1024, 2048)):
        self.pad_token_id = pad_token_id
        self.max_length = max_length
        self.buckets = tuple(sorted(b for b in buckets if b <= max_length)) + (max_length,)

    def __call__(self, batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        max_batch_len = max(item["input_ids"].size(0) for item in batch)
        target_len = self.max_length
        for b in self.buckets:
            if max_batch_len <= b:
                target_len = b
                break

        bsz = len(batch)
        input_ids = torch.full((bsz, target_len), self.pad_token_id, dtype=torch.long)
        attention_mask = torch.zeros((bsz, target_len), dtype=torch.long)
        labels = torch.full((bsz, target_len), -100, dtype=torch.long)

        for i, item in enumerate(batch):
            seq_len = min(item["input_ids"].size(0), target_len)
            input_ids[i, :seq_len] = item["input_ids"][:seq_len]
            attention_mask[i, :seq_len] = 1
            labels[i, :seq_len] = item["labels"][:seq_len]

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }


def run_sft(
    model_path_or_name: Optional[str] = None,
    output_dir: str = "checkpoints/sft",
    data_path: Optional[str] = None,
    val_data_path: Optional[str] = None,
    epochs: int = 2,
    batch_size: int = 4,
    grad_accum_steps: int = 2,
    lr: float = 1e-4,
    max_length: int = 2048,
    save_steps: int = 100,
    save_total_limit: int = 3,
    resume: Optional[str] = None,
    use_wandb: bool = False,
    wandb_project: str = "jerboa",
    wandb_run_name: Optional[str] = None,
    max_mem_fraction: float = 0.6,
    system_ram_limit: float = 85.0,
    gradient_checkpointing: bool = True,
):
    os.makedirs(output_dir, exist_ok=True)
    device = get_device()
    hw_name = torch.cuda.get_device_name(0) if device.type == "cuda" else ("Apple Silicon Metal" if device.type == "mps" else "CPU")
    print(f"Using device: {device} ({hw_name})")

    checkpoint_mgr = CheckpointManager(output_dir=output_dir, max_to_keep=save_total_limit)
    interrupt_handler = GracefulInterruptHandler()
    sleep_guard = SleepGuard()
    sleep_guard.start()

    resource_guard = SystemResourceGuard(max_mps_fraction=max_mem_fraction, system_ram_threshold=system_ram_limit)
    resource_guard.setup()

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
    if load_source == "checkpoints/pretrain/model" and not os.path.exists(load_source):
        print(f"Notice: Local path '{load_source}' not found; trying Hugging Face Hub 'ztor2/jerboa-base'...")
        load_source = "ztor2/jerboa-base"
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
            num_hidden_layers=28,
            num_attention_heads=12,
            num_key_value_heads=4,
            tie_word_embeddings=True,
            qk_norm=True,
        )
        model = JerboaForCausalLM(config)

    model.to(device)
    if gradient_checkpointing and hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
        print("[Memory] Gradient checkpointing enabled (reduced activation memory).")
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

    # CUDA supports fine-grained 64-token dynamic buckets; MPS uses 2 coarse buckets (1024, max_length) to prevent MPSGraph cache bloat
    pad_buckets = (
        tuple(range(64, max_length + 1, 64))
        if device.type == "cuda"
        else (1024, max_length)
    )
    collator = SFTDynamicCollator(
        pad_token_id=tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0,
        max_length=max_length,
        buckets=pad_buckets,
    )
    dataset = SFTChatDataset(raw_data, tokenizer, max_length=max_length)
    use_pin = device.type == "cuda"
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collator, pin_memory=use_pin)

    val_dataloader = None
    if val_data_path and os.path.exists(val_data_path):
        val_raw_data = []
        if val_data_path.endswith(".jsonl"):
            with open(val_data_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        val_raw_data.append(json.loads(line))
        else:
            with open(val_data_path, "r", encoding="utf-8") as f:
                loaded = json.load(f)
                val_raw_data = loaded if isinstance(loaded, list) else [loaded]
        if val_raw_data:
            val_dataset = SFTChatDataset(val_raw_data, tokenizer, max_length=max_length)
            val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collator, pin_memory=use_pin)
            print(f"Loaded validation set: {len(val_dataset)} samples")

    # 3. Optimizer & Scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    effective_steps_per_epoch = max(1, len(dataloader) // grad_accum_steps)
    total_steps = effective_steps_per_epoch * epochs
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
    eff_batch = batch_size * grad_accum_steps
    print(f"\n--- Starting Supervised Fine-Tuning ({epochs} Epochs, {len(dataset)} Samples, Effective Batch: {eff_batch}, Max Length: {max_length}) ---")
    model.train()
    start_time = time.time()
    interrupted = False

    import contextlib
    autocast_context = (
        torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        if device.type == "cuda" and torch.cuda.is_bf16_supported()
        else (torch.autocast(device_type="cuda", dtype=torch.float16) if device.type == "cuda" else contextlib.nullcontext())
    )

    try:
        for epoch in range(start_epoch, epochs + 1):
            epoch_loss = 0.0
            completed_batches = 0
            accum_loss = 0.0
            had_long_batch = False
            optimizer.zero_grad(set_to_none=True)

            for batch_idx, batch in enumerate(dataloader):
                if epoch == start_epoch and batch_idx < step_in_epoch:
                    continue

                batch_seq_len = batch["input_ids"].size(1)
                bsz_cur = batch["input_ids"].size(0)
                if batch_seq_len >= 1024:
                    had_long_batch = True

                # On MPS, split long batches (>1024 tokens) into single-sample micro-passes using only 2 bucket shapes (1024, max_length)
                if device.type == "mps" and batch_seq_len > 1024 and bsz_cur > 1:
                    valid_counts = (batch["labels"][:, 1:] != -100).sum(dim=1).float()
                    total_valid = valid_counts.sum().clamp(min=1.0).item()
                    step_loss = 0.0
                    for micro_i in range(bsz_cur):
                        w_i = float(valid_counts[micro_i].item()) / total_valid
                        if w_i <= 0.0:
                            continue
                        raw_m_len = int(batch["attention_mask"][micro_i].sum().item())
                        m_len = 1024 if raw_m_len <= 1024 else max_length
                        m_ids = batch["input_ids"][micro_i : micro_i + 1, :m_len].to(device)
                        m_mask = batch["attention_mask"][micro_i : micro_i + 1, :m_len].to(device)
                        m_lbls = batch["labels"][micro_i : micro_i + 1, :m_len].to(device)
                        with autocast_context:
                            m_out = model(input_ids=m_ids, attention_mask=m_mask, labels=m_lbls)
                            m_raw = m_out.loss
                            step_loss += m_raw.item() * w_i
                            m_loss = (m_raw * w_i) / grad_accum_steps
                        del m_out, m_raw
                        m_loss.backward()
                        del m_ids, m_mask, m_lbls, m_loss
                else:
                    input_ids = batch["input_ids"].to(device)
                    attention_mask = batch["attention_mask"].to(device)
                    labels = batch["labels"].to(device)

                    with autocast_context:
                        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                        raw_loss = outputs.loss
                        step_loss = raw_loss.item()
                        loss = raw_loss / grad_accum_steps
                    del outputs, raw_loss
                    loss.backward()
                    del input_ids, attention_mask, labels, loss

                epoch_loss += step_loss
                accum_loss += step_loss
                completed_batches += 1

                is_accum_step = ((batch_idx + 1) % grad_accum_steps == 0) or ((batch_idx + 1) == len(dataloader))
                if is_accum_step:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad(set_to_none=True)
                    global_step += 1

                    if global_step % 10 == 0:
                        current_lr = scheduler.get_last_lr()[0]
                        avg_step_loss = accum_loss / (10 * grad_accum_steps)
                        print(f"Epoch {epoch}/{epochs} | Step {global_step:4d} (Batch {batch_idx + 1}/{len(dataloader)}) | Loss: {avg_step_loss:.4f} | LR: {current_lr:.2e}")
                        accum_loss = 0.0

                    if device.type == "mps" and (had_long_batch or global_step % 25 == 0):
                        torch.mps.empty_cache()
                        had_long_batch = False
                        if global_step % 25 == 0:
                            resource_guard.check_and_throttle()

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

            # Validation evaluation
            val_loss_str = ""
            metrics = {"loss": avg_loss}
            if val_dataloader:
                if device.type == "mps":
                    torch.mps.empty_cache()
                model.eval()
                val_loss_sum = 0.0
                val_batches = 0
                with torch.no_grad():
                    for vbatch in val_dataloader:
                        v_ids = vbatch["input_ids"].to(device)
                        v_mask = vbatch["attention_mask"].to(device)
                        v_lbl = vbatch["labels"].to(device)
                        vout = model(input_ids=v_ids, attention_mask=v_mask, labels=v_lbl)
                        val_loss_sum += vout.loss.item()
                        val_batches += 1
                val_loss = val_loss_sum / max(val_batches, 1)
                val_loss_str = f" | Val Loss: {val_loss:.4f}"
                metrics["val_loss"] = val_loss
                if use_wandb:
                    wandb.log({"val/loss": val_loss, "epoch": epoch})
                if device.type == "mps":
                    torch.mps.empty_cache()
                model.train()

            print(f"=== Epoch {epoch} Complete | Train Loss: {avg_loss:.4f}{val_loss_str} ===")
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
                metrics=metrics,
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
    parser.add_argument("--grad_accum", type=int, default=2, help="Gradient accumulation steps")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--max_length", type=int, default=2048, help="Max sequence length")
    parser.add_argument("--save_steps", type=int, default=100, help="Save intermediate checkpoint every N steps")
    parser.add_argument("--save_total_limit", type=int, default=3, help="Max number of intermediate step checkpoints to keep")
    parser.add_argument("--resume", nargs="?", const="auto", default=None, help="Resume training ('auto' or checkpoint path)")
    parser.add_argument("--output_dir", type=str, default="checkpoints/sft")
    parser.add_argument("--data", type=str, default=default_data, help="JSON or JSONL dataset path")
    parser.add_argument("--val_data", type=str, default=None, help="Validation JSON or JSONL dataset path")
    parser.add_argument("--max_mem_fraction", type=float, default=0.72, help="MPS memory limit fraction")
    parser.add_argument("--system_ram_limit", type=float, default=88.0, help="System RAM warning limit percentage")
    parser.add_argument("--gradient_checkpointing", action="store_true", default=True, help="Enable gradient checkpointing")
    parser.add_argument("--no_gradient_checkpointing", action="store_false", dest="gradient_checkpointing", help="Disable gradient checkpointing")
    parser.add_argument("--wandb", action="store_true", help="Enable Weights & Biases experiment tracking")
    parser.add_argument("--wandb_project", type=str, default="jerboa", help="W&B project name (default: jerboa)")
    parser.add_argument("--wandb_run", type=str, default=None, help="W&B run name")
    args = parser.parse_args()
    args = apply_recipe(args, "recipes/sft.yaml")

    run_sft(
        model_path_or_name=args.model,
        output_dir=args.output_dir,
        data_path=args.data,
        val_data_path=getattr(args, "val_data", None),
        epochs=args.epochs,
        batch_size=args.batch_size,
        grad_accum_steps=getattr(args, "grad_accum", 2),
        lr=args.lr,
        max_length=getattr(args, "max_length", 2048),
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        resume=args.resume,
        use_wandb=args.wandb,
        wandb_project=args.wandb_project,
        wandb_run_name=args.wandb_run,
        max_mem_fraction=getattr(args, "max_mem_fraction", 0.72),
        system_ram_limit=getattr(args, "system_ram_limit", 88.0),
        gradient_checkpointing=getattr(args, "gradient_checkpointing", True),
    )
