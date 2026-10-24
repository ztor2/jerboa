from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
import zipfile
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from model.tokenizer import get_default_tokenizer

MASK_PATTERN = re.compile(r"#(?:이름|주소|소속|전화번호|주민등록번호|기관|날짜|장소|상호|생년월일|계좌번호)#")
THINK_BLOCK_PATTERN = re.compile(r"^\s*<think>(.*?)</think>\s*(.+)$", re.DOTALL)
THINK_STRIP_PATTERN = re.compile(r"<think>.*?</think>\s*", re.DOTALL)
BOXED_END_PATTERN = re.compile(r"\\boxed\{([^{}]+(?:\{[^{}]*\}[^{}]*)*)\}")

DEFAULT_SYSTEM_PROMPT = "You are Jerboa, a helpful, respectful, and intelligent AI assistant."
REASONING_SYSTEM_PROMPT = (
    "You are Jerboa, a helpful, respectful, and intelligent AI assistant. "
    "Reason step-by-step before providing your final answer."
)
TOOL_SYSTEM_HEADER = (
    "You are Jerboa, a helpful, respectful, and intelligent AI assistant with tool-calling capabilities."
)

SPECIAL_TAG_TOKENS = (
    "<think>",
    "</think>",
    "<answer>",
    "</answer>",
    "<tool_call>",
    "</tool_call>",
    "<tool_response>",
    "</tool_response>",
)


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


def strip_all_special_tags(text: str) -> str:
    """Remove all reasoning/tool/answer special tags from plain chat/code text."""
    cleaned = THINK_STRIP_PATTERN.sub("", text)
    for tag in SPECIAL_TAG_TOKENS:
        cleaned = cleaned.replace(tag, "")
    return cleaned.strip()


def format_reasoning_turn(user_text: str, assistant_text: str) -> Optional[Tuple[str, str]]:
    """Format a reasoning turn with balanced <think>...</think> and optional <answer>...</answer>."""
    m = THINK_BLOCK_PATTERN.match(assistant_text)
    if not m:
        return None

    thought = strip_all_special_tags(m.group(1))
    body = strip_all_special_tags(m.group(2))
    if len(thought) < 15 or len(body) < 10:
        return None

    clean_user = strip_all_special_tags(user_text)

    # Check if body concludes with \boxed{...}; if so, wrap the final answer in <answer>...</answer>
    boxed_matches = list(BOXED_END_PATTERN.finditer(body[-250:]))
    if boxed_matches:
        ans_val = boxed_matches[-1].group(1).strip()
        if ans_val and len(ans_val) <= 120:
            body = f"{body}\n<answer>{ans_val}</answer>"

    formatted_assistant = f"<think>\n{thought}\n</think>\n{body}"
    return clean_user, formatted_assistant


def validate_protocol_hygiene(messages: List[Dict[str, str]], protocol: str) -> bool:
    """Ensure special tokens strictly match the declared protocol and are 100% balanced."""
    if not messages or messages[0].get("role") != "system":
        return False

    full_text = "\n".join(m.get("content", "") for m in messages)
    if "<|im_start|>" in full_text or "<|im_end|>" in full_text:
        return False

    if protocol in ("chat", "code"):
        for tag in SPECIAL_TAG_TOKENS:
            if tag in full_text:
                return False
        return True

    if protocol == "reasoning":
        for tag in ("<tool_call>", "</tool_call>", "<tool_response>", "</tool_response>"):
            if tag in full_text:
                return False
        has_think = False
        for m in messages:
            c = m.get("content", "")
            if m.get("role") == "assistant":
                if c.count("<think>") != 1 or c.count("</think>") != 1:
                    return False
                if c.find("<think>") > c.find("</think>"):
                    return False
                if c.count("<answer>") != c.count("</answer>"):
                    return False
                has_think = True
            elif m.get("role") == "user":
                for tag in SPECIAL_TAG_TOKENS:
                    if tag in c:
                        return False
        return has_think

    if protocol == "tool_call":
        sys_content = messages[0].get("content", "")
        if "<tools>" not in sys_content and "function" not in sys_content.lower():
            return False
        has_tool_call = False
        for m in messages[1:]:
            c = m.get("content", "")
            if "<think>" in c or "</think>" in c:
                return False
            if c.count("<tool_call>") != c.count("</tool_call>"):
                return False
            if c.count("<tool_response>") != c.count("</tool_response>"):
                return False
            if m.get("role") == "assistant" and "<tool_call>" in c:
                has_tool_call = True
        return has_tool_call

    return False


def is_valid_token_sample(
    messages: List[Dict[str, str]],
    tokenizer,
    protocol: str = "chat",
    max_tokens: int = 2048,
) -> bool:
    """Validate message integrity, protocol hygiene, masking absence, and token length budget."""
    if not messages or len(messages) < 2:
        return False

    if not validate_protocol_hygiene(messages, protocol):
        return False

    has_user = False
    has_assistant = False
    total_chars = 0
    max_user_chars = max(1800, int(max_tokens * 3.0))
    max_asst_chars = max(3200, int(max_tokens * 4.5))

    for msg in messages:
        role = msg.get("role")
        content = msg.get("content", "").strip()

        if not content:
            return False

        if MASK_PATTERN.search(content):
            return False

        total_chars += len(content)

        if role == "user":
            if len(content) < 5 or len(content) > max_user_chars:
                return False
            has_user = True
        elif role == "assistant":
            if len(content) < 10 or len(content) > max_asst_chars:
                return False
            has_assistant = True

    if not (has_user and has_assistant):
        return False

    if total_chars > max_tokens * 5.0:
        return False

    full_prompt = "".join(f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n" for m in messages)
    token_ids = tokenizer.encode(full_prompt, add_special_tokens=False)
    return len(token_ids) <= max_tokens


def extract_from_71890(zip_path: Path, cat_targets: Dict[str, int], tokenizer, max_tokens: int = 2048) -> List[Dict[str, Any]]:
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
                    messages = [{"role": "system", "content": DEFAULT_SYSTEM_PROMPT}]

                    for m in raw_msgs:
                        role = m.get("role")
                        if role not in ("user", "assistant"):
                            continue
                        val = strip_all_special_tags(extract_content(m.get("content", "")))
                        if val:
                            messages.append({"role": role, "content": val})

                    if is_valid_token_sample(messages, tokenizer, protocol="chat", max_tokens=max_tokens):
                        samples.append({
                            "source": f"aihub_71890_{cat_name}",
                            "protocol": "chat",
                            "messages": messages,
                        })
                        count += 1
                except Exception:
                    continue

            print(f"  [71890] {cat_name}: {count}/{target_count} extracted (protocol=chat)")

    return samples


def extract_from_71902(zip_path: Path, cat_targets: Dict[str, Tuple[int, str]], tokenizer, max_tokens: int = 2048) -> List[Dict[str, Any]]:
    """Extract conversations from AI-Hub 71902 with explicit protocol routing."""
    samples = []
    if not zip_path.exists():
        print(f"[Warning] 71902 zip not found at {zip_path}")
        return samples

    with zipfile.ZipFile(zip_path, "r") as z:
        for cat_key, (target_count, protocol) in cat_targets.items():
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

                        if protocol == "reasoning":
                            if len(qr_list) != 2:
                                continue
                            u_raw = qr_list[0].get("content", "")
                            a_raw = qr_list[1].get("content", "")
                            res = format_reasoning_turn(u_raw, a_raw)
                            if not res:
                                continue
                            u_clean, a_clean = res
                            messages = [
                                {"role": "system", "content": REASONING_SYSTEM_PROMPT},
                                {"role": "user", "content": u_clean},
                                {"role": "assistant", "content": a_clean},
                            ]
                        else:
                            messages = [{"role": "system", "content": DEFAULT_SYSTEM_PROMPT}]
                            valid_turn = True
                            for turn in qr_list:
                                f = turn.get("from")
                                c = strip_all_special_tags(turn.get("content", ""))
                                if f == "user":
                                    messages.append({"role": "user", "content": c})
                                elif f == "assistant":
                                    messages.append({"role": "assistant", "content": c})
                                else:
                                    valid_turn = False
                                    break
                            if not valid_turn:
                                continue

                        if is_valid_token_sample(messages, tokenizer, protocol=protocol, max_tokens=max_tokens):
                            samples.append({
                                "source": f"aihub_71902_{cat_key.strip('/').replace('/', '_')}",
                                "protocol": protocol,
                                "messages": messages,
                            })
                            count += 1
                except Exception:
                    continue

            print(f"  [71902] {cat_key}: {count}/{target_count} extracted (protocol={protocol})")

    return samples


def extract_from_kullm_v2(tokenizer, target_count: int = 3000, max_tokens: int = 2048) -> List[Dict[str, Any]]:
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

                user_prompt = strip_all_special_tags(f"{inst}\n\n{inp}".strip() if inp else inst)
                out_clean = strip_all_special_tags(out)
                if MASK_PATTERN.search(user_prompt) or MASK_PATTERN.search(out_clean):
                    continue

                messages = [
                    {"role": "system", "content": DEFAULT_SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                    {"role": "assistant", "content": out_clean},
                ]

                if is_valid_token_sample(messages, tokenizer, protocol="chat", max_tokens=max_tokens):
                    samples.append({
                        "source": "hf_kullm_v2",
                        "protocol": "chat",
                        "messages": messages,
                    })
                    count += 1
            except Exception:
                continue

        print(f"  [Open Data] kullm-v2: {count}/{target_count} extracted (protocol=chat)")
    except Exception as e:
        print(f"  [Warning] Could not stream from kullm-v2: {e}")

    return samples


def extract_from_smoltalk(
    tokenizer,
    chat_target: int = 2600,
    tool_target: int = 400,
    max_tokens: int = 2048,
) -> List[Dict[str, Any]]:
    """Stream and separate pure conversational samples vs standardized tool-calling samples from smoltalk."""
    samples = []
    print(f"  [Open Data] Streaming from HuggingFaceTB/smoltalk (chat={chat_target}, tool_call={tool_target})...")
    try:
        from datasets import load_dataset
        ds = load_dataset("HuggingFaceTB/smoltalk", "all", split="train", streaming=True)
        chat_count = 0
        tool_count = 0

        for item in ds:
            if chat_count >= chat_target and tool_count >= tool_target:
                break
            try:
                raw_msgs = item.get("messages", [])
                if not raw_msgs:
                    continue

                full_raw = "\n".join(extract_content(m.get("content", "")) for m in raw_msgs)
                is_tool_sample = "<tool_call>" in full_raw or "<tools>" in full_raw

                if is_tool_sample:
                    if tool_count >= tool_target:
                        continue
                    messages = []
                    for m in raw_msgs:
                        role = m.get("role")
                        content = extract_content(m.get("content", ""))
                        if role == "system":
                            if "<tools>" not in content or "</tools>" not in content:
                                break
                            # Standardize header to Jerboa tool-calling identity without raw <tool_call> in system
                            tools_idx = content.find("<tools>")
                            tools_end = content.find("</tools>") + len("</tools>")
                            tools_block = content[tools_idx:tools_end].strip()
                            sys_text = (
                                f"{TOOL_SYSTEM_HEADER}\n\n"
                                f"You have access to the following tools:\n{tools_block}"
                            )
                            messages.append({"role": "system", "content": sys_text})
                        elif role in ("user", "assistant"):
                            messages.append({"role": role, "content": content})
                        elif role == "tool":
                            if "<tool_response>" not in content:
                                content = f"<tool_response>\n{content}\n</tool_response>"
                            messages.append({"role": "tool", "content": content})

                    if is_valid_token_sample(messages, tokenizer, protocol="tool_call", max_tokens=max_tokens):
                        samples.append({
                            "source": "hf_smoltalk_tool",
                            "protocol": "tool_call",
                            "messages": messages,
                        })
                        tool_count += 1
                else:
                    if chat_count >= chat_target:
                        continue
                    messages = [{"role": "system", "content": DEFAULT_SYSTEM_PROMPT}]
                    for m in raw_msgs:
                        role = m.get("role")
                        if role in ("user", "assistant"):
                            content = strip_all_special_tags(extract_content(m.get("content", "")))
                            if content:
                                messages.append({"role": role, "content": content})

                    if is_valid_token_sample(messages, tokenizer, protocol="chat", max_tokens=max_tokens):
                        samples.append({
                            "source": "hf_smoltalk_chat",
                            "protocol": "chat",
                            "messages": messages,
                        })
                        chat_count += 1
            except Exception:
                continue

        print(f"  [Open Data] smoltalk: chat={chat_count}/{chat_target}, tool_call={tool_count}/{tool_target} extracted")
    except Exception as e:
        print(f"  [Warning] Could not stream from smoltalk: {e}")

    return samples


def build_trial_sft_dataset(
    output_dir: str = "data/sft",
    total_target: int = 15000,
    val_size: int = 750,
    max_tokens: int = 2048,
    seed: int = 42,
    skip_open: bool = False,
):
    random.seed(seed)
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    print("=== JerboaLM Harmonized SFT Dataset Builder ===")
    print(f"Target total: {total_target} samples (Val: {val_size}, Max Tokens: {max_tokens})")

    tok_path = "checkpoints/pretrain/model" if os.path.exists("checkpoints/pretrain/model") else "data/tokenizer"
    tokenizer = get_default_tokenizer(tok_path)

    all_samples: List[Dict[str, Any]] = []

    # 1. AI-Hub 71890
    zip_71890 = Path("data/raw/aihub/71890/3.개방데이터/1.데이터/(비식별화완료)01.AI_파운데이션_모델(LLM_LAM)_사후_학습용_데이터/라벨링데이터/라벨링데이터.zip")
    targets_71890 = {
        "03. 상식 문제 및 풀이": 400,
    }
    print("\n[Step 1/4] Extracting from AI-Hub 71890 (Commonsense QA)...")
    all_samples.extend(extract_from_71890(zip_71890, targets_71890, tokenizer, max_tokens=max_tokens))

    # 2. AI-Hub 71902 (Routed by protocol: chat, reasoning, code)
    zip_71902 = Path("data/raw/aihub/71902/3.개방데이터/1.데이터/(비식별화완료)01.고품질_멀티턴_데이터_집합/원천데이터/원천데이터.zip")
    targets_71902: Dict[str, Tuple[int, str]] = {
        "/한국어/일반 대화/": (1500, "chat"),
        "/한국어/Culture/1T/": (300, "chat"),
        "/한국어/수학/1T/": (450, "reasoning"),
        "/한국어/일반 추론/1T/": (450, "reasoning"),
        "/한국어/코드/": (250, "code"),
        "/영어/일반 대화/1T/": (2500, "chat"),
        "/영어/Culture/1T/": (150, "chat"),
        "/영어/수학/1T/": (600, "reasoning"),
        "/영어/일반 추론/1T/": (600, "reasoning"),
        "/영어/코드/1T/": (800, "code"),
    }
    print("\n[Step 2/4] Extracting from AI-Hub 71902 (Chat, <think> Reasoning, Code)...")
    all_samples.extend(extract_from_71902(zip_71902, targets_71902, tokenizer, max_tokens=max_tokens))

    # 3. Open Datasets
    if not skip_open:
        print("\n[Step 3/4] Streaming from Open Hugging Face Datasets...")
        all_samples.extend(extract_from_kullm_v2(tokenizer, target_count=3000, max_tokens=max_tokens))
        all_samples.extend(extract_from_smoltalk(tokenizer, chat_target=2600, tool_target=400, max_tokens=max_tokens))
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

    selected_samples = unique_samples[:total_target] if len(unique_samples) > total_target else unique_samples

    val_count = min(val_size, int(len(selected_samples) * 0.05))
    train_samples = selected_samples[val_count:]
    val_samples = selected_samples[:val_count]

    train_file = out_path / "train.jsonl"
    val_file = out_path / "val.jsonl"
    sample_file = out_path / "sample.json"

    print(f"\n[Step 4/4] Writing splits to disk: {out_path}...")
    with open(train_file, "w", encoding="utf-8") as f:
        for item in train_samples:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    with open(val_file, "w", encoding="utf-8") as f:
        for item in val_samples:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    # Save representative samples per protocol to data/sft/sample.json for inspection
    rep_samples = []
    for proto in ("chat", "reasoning", "code", "tool_call"):
        proto_matches = [s for s in train_samples if s.get("protocol") == proto]
        # Prefer Korean + English mix in sample.json
        ko_matches = [s for s in proto_matches if "한국어" in s.get("source", "") or "kullm" in s.get("source", "") or "71890" in s.get("source", "")]
        en_matches = [s for s in proto_matches if s not in ko_matches]
        if ko_matches:
            rep_samples.append(ko_matches[0])
        if en_matches:
            rep_samples.append(en_matches[0])

    with open(sample_file, "w", encoding="utf-8") as f:
        json.dump(rep_samples, f, indent=2, ensure_ascii=False)

    print(f"\nSuccessfully generated:")
    print(f"  - Train:  {train_file} ({len(train_samples):,} samples, {train_file.stat().st_size / (1024*1024):.2f} MB)")
    print(f"  - Val:    {val_file} ({len(val_samples):,} samples, {val_file.stat().st_size / (1024*1024):.2f} MB)")
    print(f"  - Sample: {sample_file} ({len(rep_samples)} representative samples across protocols)")

    proto_counts = Counter(s.get("protocol", "unknown") for s in selected_samples)
    print("\nProtocol distribution:")
    for p, c in proto_counts.most_common():
        print(f"  - {p:12s}: {c:5,d} samples ({c / len(selected_samples) * 100:.1f}%)")

    source_counts = Counter(s.get("source", "unknown") for s in selected_samples)
    print("\nSource distribution:")
    for src, c in source_counts.most_common():
        print(f"  - {src:35s}: {c:5,d} samples ({c / len(selected_samples) * 100:.1f}%)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build harmonized SFT dataset from AI-Hub & Open Datasets")
    parser.add_argument("--output-dir", type=str, default="data/sft", help="Output directory")
    parser.add_argument("--total", type=int, default=15000, help="Total target samples")
    parser.add_argument("--val-size", type=int, default=750, help="Validation set size")
    parser.add_argument("--max-tokens", type=int, default=2048, help="Maximum sequence token length")
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
    sys.stdout.flush()
    os._exit(0)
