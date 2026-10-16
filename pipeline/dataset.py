"""AI-Hub Dataset Loader and In-Memory Archive Streamer for JerboaLM.

Supports:
- Direct in-memory streaming from .zip and .tar archives without extracting to disk
- Seamless parsing of AI-Hub Knowledge (71894), Dialogue (71908), and generic JSON/JSONL/TXT
- Cursor-based chunk extraction (get_aihub_chunk) compatible with rolling-buffer pre-training
- PyTorch IterableDataset interface (AIHubDataset) for in-memory token streaming
"""

from __future__ import annotations

import hashlib
import json
import os
import tarfile
import zipfile
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Union

import torch
from torch.utils.data import IterableDataset


def _extract_text_from_aihub_json(data: Any) -> List[str]:
    """Extract natural language text from various AI-Hub JSON schemas."""
    texts = []
    if isinstance(data, list):
        for item in data:
            texts.extend(_extract_text_from_aihub_json(item))
        return texts

    if not isinstance(data, dict):
        return texts

    # Schema 1: AI-Hub 71894 Knowledge Q&A
    q = data.get("question", "").strip() if isinstance(data.get("question"), str) else ""
    a = data.get("answer", "").strip() if isinstance(data.get("answer"), str) else ""
    if q and a:
        texts.append(f"질문: {q}\n답변: {a}")
    elif q:
        texts.append(q)
    elif a:
        texts.append(a)

    # Schema 2: AI-Hub 71908 Dialogue Q&A & Caption
    qa = data.get("annotation_qa", {})
    if isinstance(qa, dict):
        dq = qa.get("question", "").strip() if isinstance(qa.get("question"), str) else ""
        da = qa.get("response", "").strip() if isinstance(qa.get("response"), str) else ""
        if dq and da:
            texts.append(f"화자1: {dq}\n화자2: {da}")

    cap_obj = data.get("annotation_caption", {})
    if isinstance(cap_obj, dict):
        cap = cap_obj.get("caption", "").strip() if isinstance(cap_obj.get("caption"), str) else ""
        if cap:
            texts.append(cap)

    # Schema 3: AI-Hub 142 Academic & Knowledge Relations (title + sentence)
    title = data.get("title", "").strip() if isinstance(data.get("title"), str) else ""
    sentence = data.get("sentence", "").strip() if isinstance(data.get("sentence"), str) else ""
    if sentence:
        if title and title not in sentence:
            texts.append(f"{title}\n{sentence}")
        else:
            texts.append(sentence)

    # Schema 4: Instruction / Output pairs (e.g. KoAlpaca)
    inst = data.get("instruction", "").strip() if isinstance(data.get("instruction"), str) else ""
    out = data.get("output", "").strip() if isinstance(data.get("output"), str) else ""
    if inst and out:
        texts.append(f"질문: {inst}\n답변: {out}")

    # Schema 5: Generic text fields
    for key in ("text", "content", "body", "document", "passage", "summary"):
        val = data.get(key, "")
        if isinstance(val, str) and len(val.strip()) > 30:
            texts.append(val.strip())

    # Nested container lists
    for list_key in ("data", "documents", "paragraphs", "sentences", "utterances"):
        nested = data.get(list_key)
        if isinstance(nested, list):
            texts.extend(_extract_text_from_aihub_json(nested))

    return texts


class AIHubDataset(IterableDataset):
    """Streams documents directly from AI-Hub zip/tar archives and jsonl files in memory."""

    def __init__(
        self,
        raw_paths: Union[str, Path, List[Union[str, Path]]],
        tokenizer: Any = None,
        seq_len: int = 2048,
        min_chars: int = 40,
    ):
        super().__init__()
        if isinstance(raw_paths, (str, Path)):
            raw_paths = [raw_paths]

        self.archive_files: List[Path] = []
        for p in raw_paths:
            path_obj = Path(p)
            if path_obj.is_dir():
                self.archive_files.extend(sorted(path_obj.glob("**/*.zip")))
                self.archive_files.extend(sorted(path_obj.glob("**/*.tar*")))
                self.archive_files.extend(sorted(path_obj.glob("**/*.jsonl")))
                self.archive_files.extend(sorted(path_obj.glob("**/*.txt")))
            elif path_obj.exists():
                self.archive_files.append(path_obj)

        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.min_chars = min_chars

    def iter_raw_documents(self, skip_docs: int = 0) -> Iterator[str]:
        """Iterate documents with cursor offset without extracting to disk."""
        passed = 0

        for archive in self.archive_files:
            suffix = archive.suffix.lower()

            # 1. Zip Archives
            if suffix == ".zip":
                try:
                    with zipfile.ZipFile(archive, "r") as z:
                        for info in z.infolist():
                            if info.is_dir() or not (info.filename.endswith(".json") or info.filename.endswith(".txt")):
                                continue
                            try:
                                with z.open(info) as fp:
                                    if info.filename.endswith(".json"):
                                        data = json.load(fp)
                                        extracted = _extract_text_from_aihub_json(data)
                                    else:
                                        text = fp.read().decode("utf-8", errors="ignore").strip()
                                        extracted = [text] if len(text) >= self.min_chars else []

                                    for doc in extracted:
                                        if len(doc) < self.min_chars:
                                            continue
                                        if passed < skip_docs:
                                            passed += 1
                                            continue
                                        yield doc
                            except Exception:
                                continue
                except Exception as e:
                    print(f"[AIHubDataset] Warning: Failed to open zip archive {archive.name}: {e}")

            # 2. Tar Archives
            elif ".tar" in archive.name.lower():
                try:
                    with tarfile.open(archive, "r:*") as t:
                        for member in t:
                            if not member.isfile() or not (member.name.endswith(".json") or member.name.endswith(".txt")):
                                continue
                            try:
                                fp = t.extractfile(member)
                                if fp is None:
                                    continue
                                if member.name.endswith(".json"):
                                    data = json.load(fp)
                                    extracted = _extract_text_from_aihub_json(data)
                                else:
                                    text = fp.read().decode("utf-8", errors="ignore").strip()
                                    extracted = [text] if len(text) >= self.min_chars else []

                                for doc in extracted:
                                    if len(doc) < self.min_chars:
                                        continue
                                    if passed < skip_docs:
                                        passed += 1
                                        continue
                                    yield doc
                            except Exception:
                                continue
                except Exception as e:
                    print(f"[AIHubDataset] Warning: Failed to open tar archive {archive.name}: {e}")

            # 3. Line-delimited files (.jsonl, .txt)
            elif suffix in (".jsonl", ".txt"):
                try:
                    with open(archive, "r", encoding="utf-8", errors="ignore") as fp:
                        for line in fp:
                            line = line.strip()
                            if not line:
                                continue
                            if suffix == ".jsonl":
                                try:
                                    item = json.loads(line)
                                    extracted = _extract_text_from_aihub_json(item)
                                except Exception:
                                    continue
                            else:
                                extracted = [line] if len(line) >= self.min_chars else []

                            for doc in extracted:
                                if len(doc) < self.min_chars:
                                    continue
                                if passed < skip_docs:
                                    passed += 1
                                    continue
                                yield doc
                except Exception as e:
                    print(f"[AIHubDataset] Warning: Failed to open file {archive.name}: {e}")

    def __iter__(self) -> Iterator[Dict[str, torch.Tensor]]:
        """PyTorch IterableDataset interface with token packing."""
        if self.tokenizer is None:
            raise ValueError("Tokenizer must be provided to use AIHubDataset as a PyTorch DataLoader.")

        buffer: List[int] = []
        eos_id = self.tokenizer.eos_token_id

        for doc in self.iter_raw_documents():
            tokens = self.tokenizer.encode(doc, add_special_tokens=False) + [eos_id]
            buffer.extend(tokens)
            while len(buffer) >= self.seq_len:
                chunk = buffer[: self.seq_len]
                buffer = buffer[self.seq_len :]
                tensor_chunk = torch.tensor(chunk, dtype=torch.long)
                yield {"input_ids": tensor_chunk, "labels": tensor_chunk.clone()}


def ensure_aihub_cached(
    raw_dir: Union[str, Path],
    cache_dir: str = "data/cache/aihub",
    min_chars: int = 40,
) -> tuple[str, int]:
    """Build or reuse a flat NVMe JSONL cache from AI-Hub archives to eliminate O(N^2) zip re-scanning."""
    abs_cache_dir = os.path.abspath(cache_dir)
    os.makedirs(abs_cache_dir, exist_ok=True)
    jsonl_path = os.path.join(abs_cache_dir, "aihub_cached.jsonl")
    meta_path = os.path.join(abs_cache_dir, "aihub_cached.meta.json")

    dataset = AIHubDataset(raw_paths=raw_dir, min_chars=min_chars)
    archive_sig = [f"{p.name}:{p.stat().st_size}" for p in dataset.archive_files if p.exists()]

    if os.path.exists(jsonl_path) and os.path.exists(meta_path):
        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
            if meta.get("archive_sig") == archive_sig and meta.get("total_docs", 0) > 0:
                return jsonl_path, int(meta["total_docs"])
        except Exception:
            pass

    print(f"[AI-Hub Cache] Indexing Korean archives from '{raw_dir}' into local flat JSONL...")
    total_docs = 0
    with open(jsonl_path, "w", encoding="utf-8") as fout:
        for doc in dataset.iter_raw_documents(skip_docs=0):
            fout.write(json.dumps({"text": doc}, ensure_ascii=False) + "\n")
            total_docs += 1

        if total_docs < 5000:
            try:
                from datasets import load_dataset
                print(f"[AI-Hub Cache] Local archives yielded {total_docs:,} docs; supplementing with curated KoAlpaca-v1.1a...")
                ds = load_dataset("beomi/KoAlpaca-v1.1a", split="train")
                for item in ds:
                    inst = (item.get("instruction") or "").strip()
                    out = (item.get("output") or "").strip()
                    if inst and out:
                        doc = f"질문: {inst}\n답변: {out}"
                        if len(doc) >= min_chars:
                            fout.write(json.dumps({"text": doc}, ensure_ascii=False) + "\n")
                            total_docs += 1
            except Exception as e:
                print(f"[AI-Hub Cache] Notice: Could not supplement from HuggingFace: {e}")

    with open(meta_path, "w", encoding="utf-8") as fmeta:
        json.dump({"archive_sig": archive_sig, "total_docs": total_docs}, fmeta, indent=2)

    print(f"[AI-Hub Cache] Indexed {total_docs:,} Korean documents at '{jsonl_path}'.")
    return jsonl_path, max(total_docs, 1)


def get_aihub_chunk(
    raw_dir: Union[str, Path],
    output_path: str,
    skip_docs: int = 0,
    target_docs: int = 500,
    min_chars: int = 40,
    cache_dir: str = "data/cache/aihub",
) -> Dict[str, Any]:
    """Collect a batch of documents from cached AI-Hub Korean corpora with wrap-around cursor."""
    import time

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    jsonl_path, total_aihub_docs = ensure_aihub_cached(raw_dir=raw_dir, cache_dir=cache_dir, min_chars=min_chars)
    norm_skip = skip_docs % max(total_aihub_docs, 1)

    collected_docs = 0
    total_tokens = 0
    start_time = time.time()

    with open(output_path, "w", encoding="utf-8") as out_fp:
        with open(jsonl_path, "r", encoding="utf-8") as in_fp:
            while collected_docs < target_docs:
                in_fp.seek(0)
                progressed = False
                for line_idx, line in enumerate(in_fp):
                    if norm_skip > 0 and line_idx < norm_skip:
                        continue
                    if collected_docs >= target_docs:
                        break
                    try:
                        data = json.loads(line)
                        doc = data.get("text", "").strip()
                    except Exception:
                        continue
                    if len(doc) >= min_chars:
                        out_fp.write(doc + "\n<|endoftext|>\n\n")
                        total_tokens += int(len(doc.split()) * 1.4)
                        collected_docs += 1
                        progressed = True
                norm_skip = 0
                if not progressed:
                    break

    duration = time.time() - start_time
    file_size_mb = os.path.getsize(output_path) / (1024 * 1024)

    hasher = hashlib.sha256()
    with open(output_path, "rb") as f:
        while chunk := f.read(1024 * 1024):
            hasher.update(chunk)
    sha256_hash = hasher.hexdigest()

    print(
        f"[Korean Stream] Cached extraction: {collected_docs:,} Korean docs ({total_tokens:,} tokens, "
        f"{file_size_mb:.1f} MB) in {duration:.2f}s"
    )

    return {
        "source": "aihub",
        "start_offset": skip_docs,
        "end_offset": skip_docs + target_docs,
        "document_count": collected_docs,
        "approx_tokens": total_tokens,
        "file_size_mb": round(file_size_mb, 2),
        "sha256": sha256_hash,
    }


def get_chunk_source(chunk_idx: int, pattern: str = "fineweb,aihub") -> str:
    """Return the data source name for a given chunk index based on an interleaving pattern string."""
    items = [x.strip() for x in pattern.split(",") if x.strip()]
    if not items:
        return "fineweb"
    return items[(chunk_idx - 1) % len(items)]
