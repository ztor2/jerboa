from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from model.tokenizer import get_default_tokenizer

MASK_PATTERN = re.compile(r"#(?:이름|주소|소속|전화번호|주민등록번호|기관|날짜|장소|상호|생년월일|계좌번호)#")
THINK_PATTERN = re.compile(r"<think>.*?</think>\s*", re.DOTALL)
DEFAULT_SYSTEM_PROMPT = "You are Jerboa, a helpful, respectful, and intelligent AI assistant."


def extract_content(content: Any) -> str:
    """Normalize message content to a single clean string."""
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict):
                parts.append(str(item.get("value", item.get("text", ""))))
            elif isinstance(item, str):
                parts.append(item)
            else:
                parts.append(str(item))
        return "\n".join(p for p in parts if p).strip()
    return str(content).strip()


def is_valid_token_sample(messages: List[Dict[str, str]], tokenizer, max_tokens: int = 768) -> bool:
    """Validate message integrity, masking absence, and token length budget."""
    if not messages or len(messages) < 2:
        return False

    has_user = False
    has_assistant = False
    total_chars = 0

    for msg in messages:
        role = msg.get("role")
        content = msg.get("content", "").strip()

        if not content:
            return False

        if MASK_PATTERN.search(content):
            return False

        total_chars += len(content)

        if role == "user":
            if len(content) < 5 or len(content) > 1200:
                return False
            has_user = True
        elif role == "assistant":
            if len(content) < 15 or len(content) > 2000:
                return False
            has_assistant = True

    if not (has_user and has_assistant):
        return False

    # Quick pre-filter: if text is too long, it will definitely exceed max_tokens
    if total_chars > max_tokens * 3:
        return False

    full_prompt = "".join(f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n" for m in messages)
    token_ids = tokenizer.encode(full_prompt, add_special_tokens=False)
    return len(token_ids) <= max_tokens


def extract_from_71890(zip_path: Path, cat_targets: Dict[str, int], tokenizer, max_tokens: int = 768) -> List[Dict[str, Any]]:
    """Extract clean conversations from AI-Hub 71890 labeling zip."""
    samples = []
    if not zip_path.exists():
        print(f"[Warning] 71890 zip not found at {zip_path}")
        return samples

    with zipfile.ZipFile(zip_path, "r") as z:
        for cat_name, target_count in cat_targets.items():
            matching_files = [n for n in z.namelist() if n.endswith(".json") and cat_name in n]
            random.shuffle(matching_files)
            count = 0

            for fname in matching_files:
                if count >= target_count:
                    break
                try:
                    raw_text = z.read(fname).decode("utf-8", errors="ignore")
                    if MASK_PATTERN.search(raw_text):
                        continue

                    data = json.loads(raw_text)
                    raw_msgs = data.get("messages", [])
                    messages = []

                    for m in raw_msgs:
                        role = m.get("role")
                        val = extract_content(m.get("content", ""))
                        if role == "assistant":
                            val = THINK_PATTERN.sub("", val).strip()
                        if val:
                            messages.append({"role": role, "content": val})

                    if not messages:
                        continue

                    if messages[0]["role"] != "system":
                        messages.insert(0, {"role": "system", "content": DEFAULT_SYSTEM_PROMPT})

                    if is_valid_token_sample(messages, tokenizer, max_tokens=max_tokens):
                        samples.append({
                            "source": f"aihub_71890_{cat_name}",
                            "messages": messages,
                        })
                        count += 1
                except Exception:
                    continue

            print(f"  [71890] {cat_name}: {count}/{target_count} extracted")

    return samples


def extract_from_71902(zip_path: Path, cat_targets: Dict[str, int], tokenizer, max_tokens: int = 768) -> List[Dict[str, Any]]:
    """Extract clean conversations from AI-Hub 71902 multi-turn jsonl zip."""
    samples = []
    if not zip_path.exists():
        print(f"[Warning] 71902 zip not found at {zip_path}")
        return samples

    with zipfile.ZipFile(zip_path, "r") as z:
        for cat_key, target_count in cat_targets.items():
            matching_files = [n for n in z.namelist() if n.endswith(".jsonl") and cat_key in n]
            count = 0

            for fname in matching_files:
                if count >= target_count:
                    break
                try:
                    content_str = z.read(fname).decode("utf-8", errors="ignore")
                    lines = content_str.strip().split("\n")
                    random.shuffle(lines)

                    for line in lines:
                        if count >= target_count:
                            break
                        if not line or MASK_PATTERN.search(line):
                            continue

                        item = json.loads(line)
                        qr_list = item.get("query_and_response", [])
                        if not qr_list:
                            continue

                        messages = [{"role": "system", "content": DEFAULT_SYSTEM_PROMPT}]
                        valid_turn = True

                        for turn in qr_list:
                            f = turn.get("from")
                            c = turn.get("content", "").strip()
                            if f == "user":
                                messages.append({"role": "user", "content": c})
                            elif f == "assistant":
                                c_clean = THINK_PATTERN.sub("", c).strip()
                                messages.append({"role": "assistant", "content": c_clean})
                            else:
                                valid_turn = False
                                break

                        if not valid_turn:
                            continue

                        if is_valid_token_sample(messages, tokenizer, max_tokens=max_tokens):
                            samples.append({
                                "source": f"aihub_71902_{cat_key.replace('/', '_')}",
                                "messages": messages,
                            })
                            count += 1
                except Exception:
                    continue

            print(f"  [71902] {cat_key}: {count}/{target_count} extracted")

    return samples


def extract_from_kullm_v2(tokenizer, target_count: int = 3000, max_tokens: int = 768) -> List[Dict[str, Any]]:
    """Stream clean samples from open Korean dataset (nlpai-lab/kullm-v2)."""
    samples = []
    print(f"  [Open Data] Streaming from nlpai-lab/kullm-v2 (target: {target_count})...")
    try:
        from datasets import load_dataset
        ds = load_dataset("nlpai-lab/kullm-v2", split="train", streaming=True)
        count = 0

        for item in ds:
            if count >= target_count:
                break
            try:
                inst = item.get("instruction", "").strip()
                inp = item.get("input", "").strip()
                out = item.get("output", "").strip()

                if not inst or not out:
                    continue

                user_prompt = f"{inst}\n\n{inp}".strip() if inp else inst
                if MASK_PATTERN.search(user_prompt) or MASK_PATTERN.search(out):
                    continue

                messages = [
                    {"role": "system", "content": DEFAULT_SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                    {"role": "assistant", "content": out},
                ]

                if is_valid_token_sample(messages, tokenizer, max_tokens=max_tokens):
                    samples.append({
                        "source": "hf_kullm_v2",
                        "messages": messages,
                    })
                    count += 1
            except Exception:
                continue

        print(f"  [Open Data] kullm-v2: {count}/{target_count} extracted")
    except Exception as e:
        print(f"  [Warning] Could not stream from kullm-v2: {e}")

    return samples


def extract_from_smoltalk(tokenizer, target_count: int = 3000, max_tokens: int = 768) -> List[Dict[str, Any]]:
    """Stream clean English assistant samples from HuggingFaceTB/smoltalk."""
    samples = []
    print(f"  [Open Data] Streaming from HuggingFaceTB/smoltalk (target: {target_count})...")
    try:
        from datasets import load_dataset
        ds = load_dataset("HuggingFaceTB/smoltalk", "all", split="train", streaming=True)
        count = 0

        for item in ds:
            if count >= target_count:
                break
            try:
                raw_msgs = item.get("messages", [])
                if not raw_msgs:
                    continue

                messages = []
                for m in raw_msgs:
                    role = m.get("role")
                    content = extract_content(m.get("content", ""))
                    if role in ("user", "assistant", "system"):
                        messages.append({"role": role, "content": content})

                if not messages:
                    continue

                if messages[0]["role"] != "system":
                    messages.insert(0, {"role": "system", "content": DEFAULT_SYSTEM_PROMPT})

                if is_valid_token_sample(messages, tokenizer, max_tokens=max_tokens):
                    samples.append({
                        "source": "hf_smoltalk",
                        "messages": messages,
                    })
                    count += 1
            except Exception:
                continue

        print(f"  [Open Data] smoltalk: {count}/{target_count} extracted")
    except Exception as e:
        print(f"  [Warning] Could not stream from smoltalk: {e}")

    return samples


def build_trial_sft_dataset(
    output_dir: str = "data/sft",
    total_target: int = 15000,
    val_size: int = 750,
    max_tokens: int = 768,
    seed: int = 42,
    skip_open: bool = False,
) -> None:
    random.seed(seed)
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    print("=== JerboaLM Trial SFT Dataset Builder ===")
    print(f"Target total: {total_target} samples (Val: {val_size}, Train: {total_target - val_size}, Max Tokens: {max_tokens})")

    tok_path = "checkpoints/pretrain/model" if os.path.exists("checkpoints/pretrain/model") else "ztor2/jerboa"
    tokenizer = get_default_tokenizer(tok_path)

    all_samples: List[Dict[str, Any]] = []

    # 1. AI-Hub 71890
    zip_71890 = Path("data/raw/aihub/71890/3.개방데이터/1.데이터/(비식별화완료)01.AI_파운데이션_모델(LLM_LAM)_사후_학습용_데이터/라벨링데이터/라벨링데이터.zip")
    targets_71890 = {
        "03. 상식 문제 및 풀이": 400,
    }
    print("\n[Step 1/4] Extracting from AI-Hub 71890 (LLM/LAM SFT)...")
    samples_71890 = extract_from_71890(zip_71890, targets_71890, tokenizer, max_tokens=max_tokens)
    all_samples.extend(samples_71890)

    # 2. AI-Hub 71902
    zip_71902 = Path("data/raw/aihub/71902/3.개방데이터/1.데이터/(비식별화완료)01.고품질_멀티턴_데이터_집합/원천데이터/원천데이터.zip")
    targets_71902 = {
        "/한국어/일반 대화/": 1300,
        "/한국어/수학/": 600,
        "/한국어/Culture/1T/": 200,
        "/한국어/일반 추론/": 200,
        "/한국어/코드/": 100,
        "/영어/일반 대화/1T/": 3000,
        "/영어/코드/1T/": 1000,
        "/영어/수학/1T/": 1000,
        "/영어/일반 추론/1T/": 1000,
        "/영어/Culture/1T/": 150,
    }
    print("\n[Step 2/4] Extracting from AI-Hub 71902 (Multi-turn Dialogues)...")
    samples_71902 = extract_from_71902(zip_71902, targets_71902, tokenizer, max_tokens=max_tokens)
    all_samples.extend(samples_71902)

    # 3. Open Datasets
    if not skip_open:
        print("\n[Step 3/4] Streaming from Open Hugging Face Datasets...")
        samples_kullm = extract_from_kullm_v2(tokenizer, target_count=3000, max_tokens=max_tokens)
        all_samples.extend(samples_kullm)

        samples_smoltalk = extract_from_smoltalk(tokenizer, target_count=3000, max_tokens=max_tokens)
        all_samples.extend(samples_smoltalk)
    else:
        print("\n[Step 3/4] Skipping open datasets (--skip-open specified)")

    print(f"\nTotal extracted candidate samples: {len(all_samples)}")

    # Deduplication by first user prompt
    seen_prompts = set()
    unique_samples = []
    for s in all_samples:
        msgs = s.get("messages", [])
        user_text = next((m["content"] for m in msgs if m["role"] == "user"), None)
        if not user_text:
            continue
        key = user_text[:120].strip().lower()
        if key in seen_prompts:
            continue
        seen_prompts.add(key)
        unique_samples.append(s)

    print(f"After deduplication: {len(unique_samples)} unique samples")
    random.shuffle(unique_samples)

    if len(unique_samples) > total_target:
        selected_samples = unique_samples[:total_target]
    else:
        selected_samples = unique_samples

    val_count = min(val_size, int(len(selected_samples) * 0.05))
    train_samples = selected_samples[val_count:]
    val_samples = selected_samples[:val_count]

    train_file = out_path / "train.jsonl"
    val_file = out_path / "val.jsonl"

    print(f"\n[Step 4/4] Writing splits to disk: {out_path}...")
    with open(train_file, "w", encoding="utf-8") as f:
        for item in train_samples:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    with open(val_file, "w", encoding="utf-8") as f:
        for item in val_samples:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"\nSuccessfully generated:")
    print(f"  - Train: {train_file} ({len(train_samples):,} samples, {train_file.stat().st_size / (1024*1024):.2f} MB)")
    print(f"  - Val:   {val_file} ({len(val_samples):,} samples, {val_file.stat().st_size / (1024*1024):.2f} MB)")

    # Print summary by source
    counts_by_source: Dict[str, int] = {}
    for s in selected_samples:
        src = s.get("source", "unknown")
        counts_by_source[src] = counts_by_source.get(src, 0) + 1

    print("\nSource distribution:")
    for src, c in sorted(counts_by_source.items(), key=lambda x: -x[1]):
        print(f"  - {src}: {c:,} samples ({c / len(selected_samples) * 100:.1f}%)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build trial SFT dataset from AI-Hub & Open Datasets")
    parser.add_argument("--output-dir", type=str, default="data/sft", help="Output directory")
    parser.add_argument("--total", type=int, default=15000, help="Total target samples")
    parser.add_argument("--val-size", type=int, default=750, help="Validation set size")
    parser.add_argument("--max-tokens", type=int, default=768, help="Maximum sequence token length")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--skip-open", action="store_true", help="Skip streaming open HF datasets")
    args = parser.parse_args()

    build_trial_sft_dataset(
        output_dir=args.output_dir,
        total_target=args.total,
        val_size=args.val_size,
        max_tokens=args.max_tokens,
        seed=args.seed,
        skip_open=args.skip_open,
    )
