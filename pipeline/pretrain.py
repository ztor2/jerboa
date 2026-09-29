"""Unified Pre-training Pipeline for JerboaLM on Apple Silicon (MPS).

Default Paradigm: Traceable Rolling-Buffer Pre-training (Ephemeral Chunking)
- Downloads a manageable chunk (e.g. 500-1000 docs) from HuggingFace FineWeb-Edu.
- Computes cryptographic SHA-256 fingerprint & token counts for audit.
- Records data provenance in `data/manifests/dataset_lineage.jsonl`.
- Trains the model on MPS and records step metrics in `data/manifests/training_metrics.jsonl`.
- Embeds state cursor and cumulative token counts in checkpoints.
- Automatically deletes the ephemeral raw text to keep disk usage near zero.

Alternative Modes:
- `--mode stream`: Zero-disk in-memory streaming directly from HuggingFace
- `--mode file` or `--text_file <path>`: Pre-train on an existing local text file
"""

import argparse
import hashlib
import json
import os
import sys
import time
from typing import Dict, Iterator, List, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset, IterableDataset
from tqdm import tqdm
from transformers import get_cosine_schedule_with_warmup

from model.configuration_jerboa import JerboaConfig
from model.modeling_jerboa import JerboaForCausalLM
from model.tokenizer import get_default_tokenizer

MANIFEST_DIR = "data/manifests"
LINEAGE_LOG = os.path.join(MANIFEST_DIR, "dataset_lineage.jsonl")
METRICS_LOG = os.path.join(MANIFEST_DIR, "training_metrics.jsonl")


def compute_file_sha256(filepath: str) -> str:
    """Compute SHA-256 hash of a file for cryptographic data integrity audit."""
    sha256 = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            sha256.update(chunk)
    return sha256.hexdigest()


def log_lineage_record(record: Dict):
    """Append a data provenance record to dataset_lineage.jsonl."""
    os.makedirs(MANIFEST_DIR, exist_ok=True)
    with open(LINEAGE_LOG, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def log_metrics_record(record: Dict):
    """Append a training metrics record to training_metrics.jsonl."""
    os.makedirs(MANIFEST_DIR, exist_ok=True)
    with open(METRICS_LOG, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


class SyntheticOrTextDataset(Dataset):
    """Dataset providing tokenized sequences from a text file or synthetic tokens."""

    def __init__(
        self,
        seq_len: int = 256,
        num_samples: int = 1000,
        vocab_size: int = 32768,
        text_file: Optional[str] = None,
        tokenizer=None,
    ):
        self.seq_len = seq_len
        self.data = []

        if text_file and os.path.exists(text_file) and tokenizer:
            with open(text_file, "r", encoding="utf-8") as f:
                text = f.read()
            tokens = tokenizer.encode(text, add_special_tokens=False)
            total_chunks = len(tokens) // seq_len
            for i in range(total_chunks):
                chunk = tokens[i * seq_len : (i + 1) * seq_len]
                self.data.append(torch.tensor(chunk, dtype=torch.long))
        else:
            for i in range(num_samples):
                seq = torch.randint(4, vocab_size - 10, (seq_len,), dtype=torch.long)
                seq[0] = 1
                seq[-1] = 2
                self.data.append(seq)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        return {"input_ids": item, "labels": item.clone()}


class StreamingPackedDataset(IterableDataset):
    """Streams text directly from HuggingFace, filters quality, and packs sequences in memory."""

    def __init__(self, tokenizer, dataset_name: str = "HuggingFaceFW/fineweb-edu", subset: str = "sample-10BT", seq_len: int = 256, min_score: int = 3):
        self.tokenizer = tokenizer
        self.dataset_name = dataset_name
        self.subset = subset
        self.seq_len = seq_len
        self.min_score = min_score

    def __iter__(self) -> Iterator[dict]:
        dataset = load_dataset(self.dataset_name, name=self.subset, split="train", streaming=True)
        buffer = []
        eos_id = self.tokenizer.eos_token_id

        for item in dataset:
            if item.get("int_score", 0) >= self.min_score:
                text = item.get("text", "").strip()
                if len(text) > 100:
                    tokens = self.tokenizer.encode(text, add_special_tokens=False) + [eos_id]
                    buffer.extend(tokens)
                    while len(buffer) >= self.seq_len:
                        chunk = buffer[: self.seq_len]
                        buffer = buffer[self.seq_len :]
                        tensor_chunk = torch.tensor(chunk, dtype=torch.long)
                        yield {"input_ids": tensor_chunk, "labels": tensor_chunk.clone()}


def create_optimizer_and_scheduler(
    model: torch.nn.Module,
    learning_rate: float,
    weight_decay: float,
    total_steps: int,
    warmup_steps: int,
):
    decay_params = [p for p in model.parameters() if p.requires_grad and p.dim() >= 2]
    no_decay_params = [p for p in model.parameters() if p.requires_grad and p.dim() < 2]

    optimizer = torch.optim.AdamW(
        [
            {"params": decay_params, "weight_decay": weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0},
        ],
        lr=learning_rate,
        betas=(0.9, 0.95),
        eps=1e-8,
    )

    scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )
    return optimizer, scheduler


def download_chunk_with_cursor(
    output_path: str,
    skip_docs: int = 0,
    target_docs: int = 500,
    min_score: int = 3,
) -> Dict:
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    print(f"\n[Data Stream] Seeking to offset {skip_docs:,} and collecting {target_docs} documents...")

    dataset = load_dataset("HuggingFaceFW/fineweb-edu", name="sample-10BT", split="train", streaming=True)

    collected_docs = 0
    passed_docs = 0
    total_tokens = 0

    with open(output_path, "w", encoding="utf-8") as f:
        pbar = tqdm(total=target_docs, desc="Downloading partition")
        for item in dataset:
            if item.get("int_score", 0) >= min_score:
                if passed_docs < skip_docs:
                    passed_docs += 1
                    continue
                text = item.get("text", "").strip()
                if len(text) > 100:
                    f.write(text + "\n<|endoftext|>\n\n")
                    total_tokens += item.get("token_count", int(len(text.split()) * 1.3))
                    collected_docs += 1
                    pbar.update(1)
                    if collected_docs >= target_docs:
                        break
        pbar.close()

    file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
    sha256_hash = compute_file_sha256(output_path)

    return {
        "start_offset": skip_docs,
        "end_offset": skip_docs + collected_docs,
        "document_count": collected_docs,
        "approx_tokens": total_tokens,
        "file_size_mb": round(file_size_mb, 2),
        "sha256": sha256_hash,
    }


def run_pretrain(
    mode: str = "rolling",
    total_chunks: int = 3,
    docs_per_chunk: int = 500,
    steps_per_chunk: int = 15,
    max_steps: int = 100,
    batch_size: int = 2,
    grad_accum_steps: int = 2,
    seq_len: int = 256,
    lr: float = 5e-4,
    output_dir: str = "checkpoints/pretrain",
    text_file: Optional[str] = None,
    resume: Optional[str] = None,
):
    """Unified pre-training entry point."""
    os.makedirs(output_dir, exist_ok=True)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"=== JerboaLM Unified Pre-training on {device} (Mode: {mode.upper()}) ===")

    tokenizer = get_default_tokenizer()

    # 1. Checkpoint resumption or model initialization
    start_chunk = 1
    stream_cursor = 0
    cumulative_tokens = 0
    global_step = 0

    state_file = os.path.join(output_dir, "latest_state.json")
    resume_path = None

    if resume:
        if resume == "auto" and os.path.exists(state_file):
            resume_path = state_file
        elif os.path.exists(resume):
            resume_path = resume

    if resume_path:
        # If path is json state
        if resume_path.endswith(".json"):
            with open(resume_path, "r", encoding="utf-8") as f:
                st = json.load(f)
            last_ckpt = os.path.join(output_dir, "steps", st.get("last_completed_chunk", ""))
            if os.path.exists(last_ckpt):
                model = JerboaForCausalLM.from_pretrained(last_ckpt)
            else:
                model = JerboaForCausalLM.from_pretrained(os.path.join(output_dir, "model"))
            start_chunk = st.get("next_chunk_idx", 1)
            stream_cursor = st.get("stream_cursor", 0)
            cumulative_tokens = st.get("cumulative_tokens", 0)
            global_step = st.get("global_step", 0)
            print(f"Resumed from state: Next Chunk {start_chunk}, Offset: {stream_cursor:,}, Tokens: {cumulative_tokens:,}")
        else:
            print(f"Loading weights from checkpoint '{resume_path}'...")
            model = JerboaForCausalLM.from_pretrained(resume_path)
            ckpt_st_path = os.path.join(resume_path, "training_state.json")
            if os.path.exists(ckpt_st_path):
                with open(ckpt_st_path, "r", encoding="utf-8") as f:
                    st = json.load(f)
                start_chunk = st.get("next_chunk_idx", 1)
                stream_cursor = st.get("stream_cursor", 0)
                cumulative_tokens = st.get("cumulative_tokens", 0)
                global_step = st.get("global_step", 0)
    else:
        print("Initializing new JerboaLM (138M params, GQA, QK-Norm, Tied Embeddings)...")
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

    # 2. Mode execution
    if text_file and os.path.exists(text_file):
        mode = "file"

    if mode == "rolling":
        # Traceable Rolling-Buffer Pre-training
        optimizer, scheduler = create_optimizer_and_scheduler(
            model=model,
            learning_rate=lr,
            weight_decay=0.05,
            total_steps=total_chunks * steps_per_chunk,
            warmup_steps=10,
        )

        temp_chunk_path = "data/ephemeral_chunk.txt"

        for chunk_idx in range(start_chunk, start_chunk + total_chunks):
            chunk_name = f"chunk_{chunk_idx:04d}"
            print(f"\n==================================================")
            print(f"  PROCESSING {chunk_name.upper()} (Offset: {stream_cursor:,})")
            print(f"==================================================")

            chunk_meta = download_chunk_with_cursor(
                output_path=temp_chunk_path,
                skip_docs=stream_cursor,
                target_docs=docs_per_chunk,
                min_score=3,
            )

            stream_cursor = chunk_meta["end_offset"]
            chunk_tokens = chunk_meta["approx_tokens"]

            dataset = SyntheticOrTextDataset(
                seq_len=seq_len,
                text_file=temp_chunk_path,
                tokenizer=tokenizer,
            )
            dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
            data_iter = iter(dataloader)

            model.train()
            chunk_start_time = time.time()
            chunk_losses = []
            step_in_chunk = 0

            while step_in_chunk < steps_per_chunk:
                try:
                    batch = next(data_iter)
                except StopIteration:
                    data_iter = iter(dataloader)
                    batch = next(data_iter)

                input_ids = batch["input_ids"].to(device)
                labels = batch["labels"].to(device)

                outputs = model(input_ids=input_ids, labels=labels)
                loss = outputs.loss / grad_accum_steps
                loss.backward()

                if (step_in_chunk + 1) % grad_accum_steps == 0 or (step_in_chunk + 1) == steps_per_chunk:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
                    global_step += 1

                    step_loss = outputs.loss.item()
                    chunk_losses.append(step_loss)

                    log_metrics_record({
                        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                        "global_step": global_step,
                        "chunk": chunk_name,
                        "loss": round(step_loss, 4),
                        "perplexity": round(float(torch.exp(torch.tensor(step_loss))), 2),
                        "lr": scheduler.get_last_lr()[0],
                        "cumulative_tokens": cumulative_tokens + int((step_in_chunk / steps_per_chunk) * chunk_tokens),
                    })

                step_in_chunk += 1

            cumulative_tokens += chunk_tokens
            chunk_elapsed = time.time() - chunk_start_time
            avg_chunk_loss = sum(chunk_losses) / max(len(chunk_losses), 1)

            print(f"\n[{chunk_name}] Steps: {steps_per_chunk} | Avg Loss: {avg_chunk_loss:.4f} | Time: {chunk_elapsed:.1f}s")

            steps_dir = os.path.join(output_dir, "steps")
            os.makedirs(steps_dir, exist_ok=True)
            ckpt_dir = os.path.join(steps_dir, chunk_name)
            model.save_pretrained(ckpt_dir)
            tokenizer.save_pretrained(ckpt_dir)

            training_state = {
                "last_completed_chunk": chunk_name,
                "next_chunk_idx": chunk_idx + 1,
                "global_step": global_step,
                "stream_cursor": stream_cursor,
                "cumulative_tokens": cumulative_tokens,
                "avg_loss": round(avg_chunk_loss, 4),
                "sha256": chunk_meta["sha256"],
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            }
            with open(os.path.join(ckpt_dir, "training_state.json"), "w", encoding="utf-8") as f:
                json.dump(training_state, f, indent=2)

            with open(state_file, "w", encoding="utf-8") as f:
                json.dump(training_state, f, indent=2)

            log_lineage_record({
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "chunk_id": chunk_name,
                "source": "HuggingFaceFW/fineweb-edu:sample-10BT",
                "document_range": f"{chunk_meta['start_offset']}..{chunk_meta['end_offset']}",
                "doc_count": chunk_meta["document_count"],
                "tokens_in_chunk": chunk_tokens,
                "cumulative_tokens": cumulative_tokens,
                "sha256_hash": chunk_meta["sha256"],
                "initial_loss": round(chunk_losses[0], 4) if chunk_losses else None,
                "final_loss": round(chunk_losses[-1], 4) if chunk_losses else None,
                "checkpoint_saved": ckpt_dir,
                "status": "raw_data_deleted",
            })

            if os.path.exists(temp_chunk_path):
                os.remove(temp_chunk_path)
                print(f"Reclaimed disk space: deleted ephemeral file '{temp_chunk_path}'")

    elif mode == "stream":
        # Pure in-memory streaming (0 MB disk)
        dataset = StreamingPackedDataset(tokenizer, seq_len=seq_len)
        dataloader = DataLoader(dataset, batch_size=batch_size)
        optimizer, scheduler = create_optimizer_and_scheduler(
            model=model, learning_rate=lr, weight_decay=0.05, total_steps=max_steps, warmup_steps=10
        )

        model.train()
        step = 0
        running_loss = 0.0
        accum_count = 0
        start_time = time.time()

        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            outputs = model(input_ids=input_ids, labels=labels)
            loss = outputs.loss / grad_accum_steps
            loss.backward()

            running_loss += loss.item() * grad_accum_steps
            accum_count += 1

            if accum_count % grad_accum_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                step += 1

                if step % 5 == 0 or step == max_steps:
                    elapsed = time.time() - start_time
                    tok_per_sec = (step * batch_size * grad_accum_steps * seq_len) / max(elapsed, 1e-4)
                    print(f"Step {step:3d}/{max_steps} | Loss: {running_loss/accum_count:.4f} | Speed: {tok_per_sec:.1f} tok/s")

                running_loss = 0.0
                accum_count = 0

                if step >= max_steps:
                    break

    else:
        # File mode
        dataset = SyntheticOrTextDataset(seq_len=seq_len, text_file=text_file, tokenizer=tokenizer)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        optimizer, scheduler = create_optimizer_and_scheduler(
            model=model, learning_rate=lr, weight_decay=0.05, total_steps=max_steps, warmup_steps=10
        )
        model.train()
        step = 0
        accum_count = 0
        data_iter = iter(dataloader)

        while step < max_steps:
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(dataloader)
                batch = next(data_iter)

            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            outputs = model(input_ids=input_ids, labels=labels)
            loss = outputs.loss / grad_accum_steps
            loss.backward()
            accum_count += 1

            if accum_count % grad_accum_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                step += 1
                accum_count = 0

    # 3. Base model save to checkpoints/pretrain/model
    final_path = os.path.join(output_dir, "model")
    model.save_pretrained(final_path)
    tokenizer.save_pretrained(final_path)
    print(f"\nPre-training completed! Base model saved to '{final_path}'")
    return final_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="JerboaLM Unified Pre-training")
    parser.add_argument("--mode", type=str, default="rolling", choices=["rolling", "stream", "file"], help="Pre-training mode (default: rolling)")
    parser.add_argument("--chunks", type=int, default=3, help="Number of chunks for rolling mode")
    parser.add_argument("--docs_per_chunk", type=int, default=500, help="Documents per chunk")
    parser.add_argument("--steps_per_chunk", type=int, default=15, help="Steps per chunk")
    parser.add_argument("--steps", type=int, default=50, help="Total steps for streaming/file mode")
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--grad_accum", type=int, default=2)
    parser.add_argument("--seq_len", type=int, default=256)
    parser.add_argument("--lr", type=float, default=5e-4)
    default_text = "data/pretrain/fineweb_edu_sample.txt" if os.path.exists("data/pretrain/fineweb_edu_sample.txt") else None
    parser.add_argument("--text_file", type=str, default=default_text, help="Path to local text file (triggers file mode)")
    parser.add_argument("--output_dir", type=str, default="checkpoints/pretrain", help="Directory to save checkpoints")
    parser.add_argument("--resume", type=str, default=None, help="Checkpoint directory or state file to resume from")
    args = parser.parse_args()

    run_pretrain(
        mode=args.mode,
        total_chunks=args.chunks,
        docs_per_chunk=args.docs_per_chunk,
        steps_per_chunk=args.steps_per_chunk,
        max_steps=args.steps,
        batch_size=args.batch_size,
        grad_accum_steps=args.grad_accum,
        seq_len=args.seq_len,
        lr=args.lr,
        output_dir=args.output_dir,
        text_file=args.text_file,
        resume=args.resume,
    )
