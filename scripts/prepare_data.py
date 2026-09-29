"""Download and prepare high-quality data across all training stages.

Supported Stages:
- pretrain: Streams FineWeb-Edu (score >= 3) or TinyStories into `data/pretrain/`
- sft: Streams conversational instruction data (ChatML) into `data/sft/`
- dpo: Streams pairwise preference pairs (chosen / rejected) into `data/dpo/`
- grpo: Streams verifiable mathematical & logical reasoning tasks into `data/grpo/`
- all: Prepares baseline samples for all stages
"""

import argparse
import json
import os
import sys
from typing import Optional
from tqdm import tqdm
from datasets import load_dataset


def download_fineweb_edu_subset(
    output_path: str = "data/pretrain/fineweb_edu_sample.txt",
    num_samples: int = 1000,
    min_score: int = 3,
):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    print(f"Streaming from HuggingFaceFW/fineweb-edu (target: {num_samples} docs, score >= {min_score})...")

    dataset = load_dataset(
        "HuggingFaceFW/fineweb-edu",
        name="sample-10BT",
        split="train",
        streaming=True,
    )

    collected = 0
    total_tokens_approx = 0

    with open(output_path, "w", encoding="utf-8") as f:
        pbar = tqdm(total=num_samples, desc="Pretrain Docs")
        for item in dataset:
            score = item.get("int_score", 0)
            if score >= min_score:
                text = item["text"].strip()
                if len(text) > 200:
                    f.write(text + "\n<|endoftext|>\n\n")
                    collected += 1
                    total_tokens_approx += item.get("token_count", len(text.split()) * 1.3)
                    pbar.update(1)

                    if collected >= num_samples:
                        break
        pbar.close()

    file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"Saved {collected} docs to '{output_path}' ({file_size_mb:.2f} MB, ~{int(total_tokens_approx):,} tokens)")
    return output_path


def download_tinystories_subset(
    output_path: str = "data/pretrain/tinystories_sample.txt",
    num_samples: int = 2000,
):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    print(f"Streaming from roneneldan/TinyStories (target: {num_samples} stories)...")

    dataset = load_dataset("roneneldan/TinyStories", split="train", streaming=True)
    collected = 0

    with open(output_path, "w", encoding="utf-8") as f:
        pbar = tqdm(total=num_samples, desc="TinyStories")
        for item in dataset:
            text = item["text"].strip()
            if text:
                f.write(text + "\n<|endoftext|>\n\n")
                collected += 1
                pbar.update(1)
                if collected >= num_samples:
                    break
        pbar.close()

    file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"Saved {collected} stories to '{output_path}' ({file_size_mb:.2f} MB)")
    return output_path


def download_sft_subset(
    output_path: str = "data/sft/sample.json",
    num_samples: int = 100,
):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    print(f"Streaming SFT conversation data (target: {num_samples} dialogues)...")

    dataset = load_dataset("HuggingFaceH4/ultrachat_200k", split="train_sft", streaming=True)
    records = []

    pbar = tqdm(total=num_samples, desc="SFT Dialogues")
    for item in dataset:
        messages = item.get("messages", [])
        if len(messages) >= 2:
            formatted_messages = [
                {"role": "system", "content": "You are Jerboa, an intelligent and helpful AI assistant."}
            ]
            for m in messages:
                if m.get("role") in ["user", "assistant"]:
                    formatted_messages.append({"role": m["role"], "content": m["content"]})
            records.append({"messages": formatted_messages})
            pbar.update(1)
            if len(records) >= num_samples:
                break
    pbar.close()

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2, ensure_ascii=False)

    print(f"Saved {len(records)} SFT conversation samples to '{output_path}'")
    return output_path


def download_dpo_subset(
    output_path: str = "data/dpo/sample.json",
    num_samples: int = 100,
):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    print(f"Streaming DPO preference data (target: {num_samples} pairs)...")

    dataset = load_dataset("HuggingFaceH4/ultrafeedback_binarized", split="train_prefs", streaming=True)
    records = []

    pbar = tqdm(total=num_samples, desc="DPO Pairs")
    for item in dataset:
        prompt = item.get("prompt", "")
        chosen = item.get("chosen", [])
        rejected = item.get("rejected", [])

        chosen_text = chosen[-1]["content"] if chosen and isinstance(chosen, list) else ""
        rejected_text = rejected[-1]["content"] if rejected and isinstance(rejected, list) else ""

        if prompt and chosen_text and rejected_text and chosen_text != rejected_text:
            records.append({
                "prompt": prompt,
                "chosen": chosen_text,
                "rejected": rejected_text,
            })
            pbar.update(1)
            if len(records) >= num_samples:
                break
    pbar.close()

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2, ensure_ascii=False)

    print(f"Saved {len(records)} DPO preference pairs to '{output_path}'")
    return output_path


def download_grpo_subset(
    output_path: str = "data/grpo/sample.json",
    num_samples: int = 100,
):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    print(f"Streaming GSM8K verifiable math reasoning data (target: {num_samples} problems)...")

    dataset = load_dataset("openai/gsm8k", "main", split="train", streaming=True)
    records = []

    pbar = tqdm(total=num_samples, desc="GRPO Tasks")
    for item in dataset:
        q = item.get("question", "")
        a = item.get("answer", "")
        # GSM8K format ends with #### <answer>
        if "####" in a:
            parts = a.split("####")
            ans_clean = parts[1].strip()
            prompt = (
                f"Solve the math problem: {q} "
                f"Think step by step inside <think>...</think> and end with <answer>X</answer>."
            )
            records.append({
                "prompt": prompt,
                "expected_answer": ans_clean,
                "domain": "math",
            })
            pbar.update(1)
            if len(records) >= num_samples:
                break
    pbar.close()

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2, ensure_ascii=False)

    print(f"Saved {len(records)} GRPO verifiable reasoning tasks to '{output_path}'")
    return output_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download and prepare training datasets")
    parser.add_argument(
        "--stage",
        type=str,
        default="pretrain",
        choices=["pretrain", "sft", "dpo", "grpo", "all"],
        help="Training stage to prepare data for",
    )
    parser.add_argument(
        "--source",
        type=str,
        default="fineweb-edu",
        choices=["fineweb-edu", "tinystories"],
        help="Source corpus for pre-training",
    )
    parser.add_argument("--samples", type=int, default=500, help="Number of samples to collect")
    parser.add_argument("--min_score", type=int, default=3, help="Minimum score for FineWeb-Edu")
    parser.add_argument("--output", type=str, default=None, help="Custom output path")
    args = parser.parse_args()

    if args.stage == "pretrain" or args.stage == "all":
        if args.source == "fineweb-edu":
            out = args.output or "data/pretrain/fineweb_edu_sample.txt"
            download_fineweb_edu_subset(out, num_samples=args.samples, min_score=args.min_score)
        else:
            out = args.output or "data/pretrain/tinystories_sample.txt"
            download_tinystories_subset(out, num_samples=args.samples)

    if args.stage == "sft" or args.stage == "all":
        out = args.output or "data/sft/sample.json"
        download_sft_subset(out, num_samples=min(args.samples, 200))

    if args.stage == "dpo" or args.stage == "all":
        out = args.output or "data/dpo/sample.json"
        download_dpo_subset(out, num_samples=min(args.samples, 200))

    if args.stage == "grpo" or args.stage == "all":
        out = args.output or "data/grpo/sample.json"
        download_grpo_subset(out, num_samples=min(args.samples, 200))
