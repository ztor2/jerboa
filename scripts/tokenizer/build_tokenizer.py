"""Jerboa Multi-domain Tokenizer Builder and Benchmark Pipeline."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import zipfile
from pathlib import Path
from typing import Any, Dict, List

# Ensure repo root is on sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from datasets import load_dataset
from tokenizers import Tokenizer, decoders, models, normalizers, pre_tokenizers, trainers
from transformers import AutoTokenizer, PreTrainedTokenizerFast

from model.tokenizer import DEFAULT_CHAT_TEMPLATE, SPECIAL_TOKENS


def collect_aihub_knowledge(raw_dir: Path, output_file: Path, max_lines: int = 150000) -> int:
    """Extract question/answer pairs from AI-Hub 71894 knowledge dataset."""
    zip_candidates = list(raw_dir.glob("**/*지식_지능*/**/라벨링데이터.zip"))
    if not zip_candidates:
        print("[Corpus] AI-Hub 71894 zip not found in raw_dir.")
        return 0

    print(f"[Corpus] Extracting AI-Hub knowledge text from {zip_candidates[0].name}...")
    written = 0
    with zipfile.ZipFile(zip_candidates[0], "r") as z, open(output_file, "w", encoding="utf-8") as out_fp:
        with z.open("/annotation.json") as fp:
            data = json.load(fp)
            for item in data[:max_lines]:
                q = item.get("question", "").strip()
                a = item.get("answer", "").strip()
                if q:
                    out_fp.write(q + "\n")
                    written += 1
                if a:
                    out_fp.write(a + "\n")
                    written += 1

    print(f"[Corpus] AI-Hub knowledge: {written:,} lines written to {output_file.name}")
    return written


def collect_aihub_dialogue(raw_dir: Path, output_file: Path, max_lines: int = 100000) -> int:
    """Extract QA and caption text from AI-Hub 71908 dialogue dataset."""
    zip_candidates = list(raw_dir.glob("**/*대화생성형*/**/라벨링데이터.zip"))
    if not zip_candidates:
        print("[Corpus] AI-Hub 71908 zip not found in raw_dir.")
        return 0

    print(f"[Corpus] Extracting AI-Hub dialogue text from {zip_candidates[0].name}...")
    written = 0
    with zipfile.ZipFile(zip_candidates[0], "r") as z, open(output_file, "w", encoding="utf-8") as out_fp:
        for name in z.namelist()[:max_lines]:
            if not name.endswith(".json"):
                continue
            try:
                with z.open(name) as fp:
                    data = json.load(fp)
                    qa = data.get("annotation_qa", {})
                    q = qa.get("question", "").strip()
                    a = qa.get("response", "").strip()
                    cap = data.get("annotation_caption", {}).get("caption", "").strip()
                    if q:
                        out_fp.write(q + "\n")
                        written += 1
                    if a:
                        out_fp.write(a + "\n")
                        written += 1
                    if cap:
                        out_fp.write(cap + "\n")
                        written += 1
            except Exception:
                continue

    print(f"[Corpus] AI-Hub dialogue: {written:,} lines written to {output_file.name}")
    return written


def collect_korean_colloquial(output_file: Path, max_samples: int = 40000) -> int:
    """Stream open-source Korean colloquial/dialogue datasets."""
    print("[Corpus] Streaming Korean conversational dataset (KoAlpaca / Dialogue)...")
    written = 0
    try:
        ds = load_dataset("beomi/KoAlpaca-v1.1a", split="train", streaming=True)
        with open(output_file, "w", encoding="utf-8") as fp:
            for item in ds:
                instruction = item.get("instruction", "").strip()
                output = item.get("output", "").strip()
                if instruction:
                    fp.write(instruction + "\n")
                    written += 1
                if output:
                    fp.write(output + "\n")
                    written += 1
                if written >= max_samples:
                    break
    except Exception as e:
        print(f"[Corpus] Warning: Failed to stream Korean conversational data: {e}")

    print(f"[Corpus] Korean colloquial: {written:,} lines written to {output_file.name}")
    return written


def collect_english_educational(output_file: Path, max_samples: int = 50000) -> int:
    """Stream FineWeb-Edu high-signal educational samples."""
    print("[Corpus] Streaming English educational dataset (FineWeb-Edu)...")
    written = 0
    try:
        ds = load_dataset("HuggingFaceFW/fineweb-edu", name="sample-10BT", split="train", streaming=True)
        with open(output_file, "w", encoding="utf-8") as fp:
            for item in ds:
                if item.get("score", 0) >= 3:
                    text = item.get("text", "").strip()
                    if text:
                        fp.write(text + "\n\n")
                        written += 1
                if written >= max_samples:
                    break
    except Exception as e:
        print(f"[Corpus] Warning: Failed to stream FineWeb-Edu: {e}")

    print(f"[Corpus] English educational: {written:,} lines written to {output_file.name}")
    return written


def collect_english_wiki(output_file: Path, max_samples: int = 30000) -> int:
    """Stream Wikipedia / Encyclopedic English text for broad named entity coverage."""
    print("[Corpus] Streaming English encyclopedic dataset (Wikitext)...")
    written = 0
    try:
        ds = load_dataset("wikitext", "wikitext-103-raw-v1", split="train", streaming=True)
        with open(output_file, "w", encoding="utf-8") as fp:
            for item in ds:
                text = item.get("text", "").strip()
                if text and len(text) > 40:
                    fp.write(text + "\n")
                    written += 1
                if written >= max_samples:
                    break
    except Exception as e:
        print(f"[Corpus] Warning: Failed to stream Wikitext: {e}")

    print(f"[Corpus] English encyclopedic: {written:,} lines written to {output_file.name}")
    return written


def collect_code_samples(output_file: Path, max_samples: int = 30000) -> int:
    """Stream clean code samples (Python, JS, Markdown, Shell)."""
    print("[Corpus] Streaming code samples (Python codes & CodeParrot)...")
    written = 0
    try:
        ds = load_dataset("flytech/python-codes-25k", split="train", streaming=True)
        with open(output_file, "w", encoding="utf-8") as fp:
            for item in ds:
                text = item.get("text", "") or (item.get("instruction", "") + "\n" + item.get("output", ""))
                if text.strip():
                    fp.write(text.strip() + "\n\n")
                    written += 1
                if written >= max_samples:
                    break
    except Exception as e:
        print(f"[Corpus] Warning: Failed to stream flytech/python-codes-25k: {e}")

    try:
        if written < max_samples:
            ds2 = load_dataset("codeparrot/codeparrot-clean-valid", split="train", streaming=True)
            with open(output_file, "a", encoding="utf-8") as fp:
                for item in ds2:
                    content = item.get("content", "").strip()
                    if content:
                        fp.write(content + "\n\n")
                        written += 1
                    if written >= max_samples:
                        break
    except Exception as e:
        print(f"[Corpus] Warning: Failed to stream CodeParrot: {e}")

    print(f"[Corpus] Code samples: {written:,} lines written to {output_file.name}")
    return written


def train_bpe_tokenizer(
    text_files: List[Path],
    vocab_size: int = 49152,
    save_directory: str = "data/tokenizer",
) -> PreTrainedTokenizerFast:
    """Train a byte-level Byte-Pair Encoding (BPE) tokenizer."""
    os.makedirs(save_directory, exist_ok=True)
    file_paths = [str(p) for p in text_files if p.exists() and p.stat().st_size > 0]
    if not file_paths:
        raise ValueError("No valid corpus files found to train tokenizer.")

    print(f"[Trainer] Training Byte-Level BPE tokenizer across {len(file_paths)} files...")
    start_time = time.time()

    tokenizer = Tokenizer(models.BPE())
    tokenizer.normalizer = normalizers.NFKC()
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()

    all_specials = [
        SPECIAL_TOKENS["pad_token"],
        SPECIAL_TOKENS["bos_token"],
        SPECIAL_TOKENS["eos_token"],
        SPECIAL_TOKENS["unk_token"],
    ] + SPECIAL_TOKENS["additional_special_tokens"]
    # De-duplicate while preserving order
    dedup_specials = list(dict.fromkeys(all_specials))

    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=dedup_specials,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        show_progress=True,
    )

    tokenizer.train(file_paths, trainer)
    train_duration = time.time() - start_time
    print(f"[Trainer] BPE training completed in {train_duration:.2f} seconds!")

    fast_tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        bos_token=SPECIAL_TOKENS["bos_token"],
        eos_token=SPECIAL_TOKENS["eos_token"],
        pad_token=SPECIAL_TOKENS["pad_token"],
        unk_token=SPECIAL_TOKENS["unk_token"],
        additional_special_tokens=SPECIAL_TOKENS["additional_special_tokens"],
    )
    fast_tokenizer.chat_template = DEFAULT_CHAT_TEMPLATE
    fast_tokenizer.save_pretrained(save_directory)
    print(f"[Trainer] Tokenizer saved to: {save_directory}")
    return fast_tokenizer


def run_benchmark(tokenizer: PreTrainedTokenizerFast) -> Dict[str, Any]:
    """Benchmark token compression efficiency (fertility) across multi-domain tests."""
    test_cases = {
        "ko_literary": "브롬톤 런던은 친환경 비건 충전재를 사용하여 자연 순환 기반 지속가능성을 강조합니다.",
        "ko_colloquial": "오늘 날씨 완전 좋은데 이따가 한강 가서 라면 먹을래? 진짜 맛있겠다!",
        "ko_technical": "트랜스포머 아키텍처의 멀티헤드 어텐션과 슬라이딩 윈도우 메커니즘을 결합합니다.",
        "en_academic": "The transformer architecture relies on grouped-query attention to optimize cache efficiency.",
        "en_dialogue": "Hey, what are you doing tonight? Let's grab some coffee and chat!",
        "code_python": "def forward(self, input_ids):\n    x = self.embed(input_ids)\n    return self.layers(x)",
    }

    try:
        baseline_tok = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM-135M")
    except Exception:
        baseline_tok = None

    results = {}
    print("\n" + "=" * 65)
    print(f"{'Domain / Test Phrase':<25} | {'Jerboa (49k)':<15} | {'SmolLM (49k)':<15} | {'Diff':<8}")
    print("-" * 65)

    for name, text in test_cases.items():
        j_tokens = tokenizer.encode(text)
        j_count = len(j_tokens)
        if baseline_tok:
            b_tokens = baseline_tok.encode(text)
            b_count = len(b_tokens)
            diff = f"{j_count - b_count:+d} ({(j_count/b_count - 1)*100:+.1f}%)"
        else:
            b_count = "N/A"
            diff = "N/A"

        results[name] = {
            "text": text,
            "jerboa_tokens": j_count,
            "smollm_tokens": b_count,
        }
        print(f"{name:<25} | {j_count:<15} | {str(b_count):<15} | {diff:<8}")
    print("=" * 65 + "\n")
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Jerboa Multi-domain Tokenizer Builder")
    parser.add_argument("--corpus-dir", default="data/tokenizer_corpus", help="Corpus cache directory")
    parser.add_argument("--raw-dir", default="data/raw/aihub_test", help="AI-Hub raw downloaded files")
    parser.add_argument("--output-dir", default="data/tokenizer", help="Trained tokenizer save directory")
    parser.add_argument("--vocab-size", type=int, default=49152, help="Target vocabulary size")
    parser.add_argument("--skip-collect", action="store_true", help="Skip corpus collection and train directly")
    args = parser.parse_args()

    corpus_dir = Path(args.corpus_dir)
    raw_dir = Path(args.raw_dir)
    save_dir = Path(args.output_dir)
    corpus_dir.mkdir(parents=True, exist_ok=True)

    files_map = {
        "ko_knowledge": corpus_dir / "ko_knowledge.txt",
        "ko_dialogue": corpus_dir / "ko_dialogue.txt",
        "ko_colloquial": corpus_dir / "ko_colloquial.txt",
        "en_educational": corpus_dir / "en_educational.txt",
        "en_wiki": corpus_dir / "en_wiki.txt",
        "code_samples": corpus_dir / "code_samples.txt",
    }

    report: Dict[str, Any] = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "vocab_size": args.vocab_size,
        "corpus_stats": {},
    }

    if not args.skip_collect:
        print("=== Step 1: Multi-domain Corpus Collection ===")
        collect_aihub_knowledge(raw_dir, files_map["ko_knowledge"])
        collect_aihub_dialogue(raw_dir, files_map["ko_dialogue"])
        collect_korean_colloquial(files_map["ko_colloquial"])
        collect_english_educational(files_map["en_educational"])
        collect_english_wiki(files_map["en_wiki"])
        collect_code_samples(files_map["code_samples"])

    # Calculate corpus statistics
    total_bytes = 0
    active_files = []
    for k, p in files_map.items():
        if p.exists() and p.stat().st_size > 0:
            sz = p.stat().st_size
            total_bytes += sz
            active_files.append(p)
            report["corpus_stats"][k] = {
                "file": str(p),
                "size_mb": round(sz / (1024 * 1024), 2),
            }
    report["total_corpus_mb"] = round(total_bytes / (1024 * 1024), 2)
    print(f"[Corpus] Total active corpus: {report['total_corpus_mb']} MB across {len(active_files)} files")

    print("\n=== Step 2: Training Byte-Level BPE Tokenizer ===")
    trained_tok = train_bpe_tokenizer(
        text_files=active_files,
        vocab_size=args.vocab_size,
        save_directory=str(save_dir),
    )

    print("\n=== Step 3: Benchmarking Tokenization Efficiency ===")
    benchmark_results = run_benchmark(trained_tok)
    report["benchmarks"] = benchmark_results

    # Save detailed build report
    report_path = save_dir / "tokenizer_build_report.json"
    with open(report_path, "w", encoding="utf-8") as fp:
        json.dump(report, fp, indent=2, ensure_ascii=False)
    print(f"[Report] Tokenizer build report saved to: {report_path}")


if __name__ == "__main__":
    main()
