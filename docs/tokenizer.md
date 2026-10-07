# Tokenizer Design & Architecture

JerboaLM uses a custom bilingual (Korean-English-Code) Byte-Level Byte-Pair Encoding (BPE) tokenizer with a vocabulary size of **49,152** tokens.

---

## 1. Motivation & Background

Standard open-source small language models (e.g., SmolLM-135M) allocate almost their entire vocabulary (~49k) to English and common Western characters. When applied to Korean:
- Words decompose into raw UTF-8 byte sequences (3 bytes per character).
- A simple phrase like `"안녕하세요. 제르보아 언어 모델입니다."` consumes **47 tokens** in SmolLM, compared to **11 tokens** in JerboaLM.
- This creates severe token bloat (~4.3x fertility penalty), squandering the model's effective context window and wasting 3x–4x training FLOPs.

Rather than inflating the vocabulary to 70k+ (which would consume >60M parameters in word embeddings alone), JerboaLM trains a balanced 49,152-token Byte-Level BPE from scratch across balanced high-quality Korean, English, and Code corpora.

---

## 2. Vocabulary Specifications

| Item | Specification |
| :--- | :--- |
| **Algorithm** | Byte-Level Byte-Pair Encoding (BPE) |
| **Target Vocab Size** | 49,152 |
| **Normalizer** | NFKC |
| **Pre-tokenizer** | ByteLevel (add_prefix_space=False) |
| **Decoder** | ByteLevel |
| **Special Tokens** | `<|im_start|>`, `<|im_end|>`, `<|pad|>`, `<|unk|>`, `<|endoftext|>`, `<think>`, `</think>`, `<answer>`, `</answer>`, `<|image|>`, `<|audio|>`, `<tool_call>`, `</tool_call>`, `<tool_response>`, `</tool_response>`, `<|quad_space|>` |
| **Chat Template** | ChatML format (`<|im_start|>role\ncontent<|im_end|>\n`) |

---

## 3. Training Corpus Breakdown (Total: 790.34 MB)

The tokenizer was trained on a multi-domain corpus designed to prevent over-indexing on any single genre:

| Category | Source | Samples / Size | Description |
| :--- | :--- | :--- | :--- |
| **Korean Knowledge** | AI-Hub 71894 (지식/지능 데이터) | 300,000 lines (496.06 MB) | Formal Korean, technical, encyclopedic Q&A |
| **Korean Dialogue** | AI-Hub 71908 (대화생성형 AI 학습용) | 132,726 lines (6.14 MB) | Conversational Korean, daily interactions |
| **Korean Colloquial** | `beomi/KoAlpaca-v1.1a` | 40,000 lines (19.48 MB) | Colloquial expressions, informal style |
| **English Educational** | `HuggingFaceFW/fineweb-edu` (score $\ge$ 3) | 50,000 samples (237.15 MB) | High-signal academic, science, educational web |
| **English Encyclopedic** | `wikitext-103-raw-v1` | 30,000 samples (18.04 MB) | General vocabulary, named entities |
| **Code & Markdown** | `flytech/python-codes-25k` & `codeparrot` | 30,000 samples (13.46 MB) | Python syntax, indentations, common symbols |

---

## 4. Benchmark: Token Compression Efficiency (Fertility)

Token counts across representative test phrases compared against `HuggingFaceTB/SmolLM-135M` (49k vocab baseline):

| Domain / Test Phrase | JerboaLM (49k) | SmolLM-135M (49k) | Difference |
| :--- | :---: | :---: | :---: |
| **Korean Literary / Formal**<br>*"브롬톤 런던은 친환경 비건 충전재를 사용하여 자연 순환 기반 지속가능성을 강조합니다."* | **19** | 102 | **-81.4%** |
| **Korean Colloquial**<br>*"오늘 날씨 완전 좋은데 이따가 한강 가서 라면 먹을래? 진짜 맛있겠다!"* | **25** | 80 | **-68.8%** |
| **Korean Technical**<br>*"트랜스포머 아키텍처의 멀티헤드 어텐션과 슬라이딩 윈도우 메커니즘을 결합합니다."* | **18** | 97 | **-81.4%** |
| **English Academic**<br>*"The transformer architecture relies on grouped-query attention to optimize cache efficiency."* | **14** | 14 | **0.0%** |
| **English Dialogue**<br>*"Hey, what are you doing tonight? Let's grab some coffee and chat!"* | **18** | 16 | **+12.5%** (+2 tokens) |
| **Python Code**<br>*"def forward(self, input_ids):\n    x = self.embed(input_ids)\n    return self.layers(x)"* | **31** | 28 | **+10.7%** (+3 tokens) |

### Key Takeaways
1. **Dramatic Korean Compression (3.2x–5.4x improvement)**: Korean text requires 68%–81% fewer tokens. This directly quadruples the effective context length for Korean conversations and saves 3x–5x compute during pre-training.
2. **Preserved English & Code Efficiency**: English academic text matches SmolLM token-for-token (14 vs 14), while conversational English and Python code show minimal variation (+2 to +3 tokens).

---

## 5. Pipeline & Reproducibility

### Quick Build
```bash
make tokenizer
```

### Script Execution
```bash
python3 scripts/tokenizer/build_tokenizer.py \
    --corpus-dir data/tokenizer_corpus \
    --raw-dir data/raw/aihub_test \
    --output-dir data/tokenizer \
    --vocab-size 49152
```

Build artifacts generated in `data/tokenizer/`:
- `tokenizer.json`: Full serialized tokenizer model.
- `tokenizer_config.json`: HuggingFace tokenizer configuration.
- `special_tokens_map.json`: Special token mappings.
- `tokenizer_build_report.json`: Build metadata, corpus breakdown, and benchmark log.
