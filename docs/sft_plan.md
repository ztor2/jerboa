# JerboaLM Supervised Fine-Tuning (SFT) Plan & Dataset Design

This document details the data curation strategy, pipeline architecture, and recipe for the trial Supervised Fine-Tuning (SFT) of JerboaLM (~221.4M parameters).

---

## 1. Objectives & Scope

- **Primary Goal**: Transition JerboaLM from a raw pre-trained causal base model (`checkpoints/pretrain/model`) into an instruction-following assistant.
- **Focus**: Natural Korean and English turn-taking, respectful polite tone (해요체/하십시오체), clean markdown formatting, and reliable sequence termination (`<|im_end|>`).
- **Scope Restriction**: Non-conversational creative/aesthetic data is excluded from the trial run; emphasis is placed strictly on instruction compliance, general reasoning, common sense, and programming assistance.

---

## 2. Dataset Composition (Total: 13,560 Samples)

The trial SFT dataset is built by combining verified clean subsets of 2025/2026 AI-Hub data with high-signal open instruction datasets:

| Source | Category / Split | Count | Share (%) | Key Purpose |
| :--- | :--- | :---: | :---: | :--- |
| **Hugging Face (`nlpai-lab/kullm-v2`)** | Korean General QA & Instructions | 3,000 | 22.1% | Clear, concise Korean assistant responses, no masking issues |
| **Hugging Face (`HuggingFaceTB/smoltalk`)** | English Conversational Assistant | 2,905 | 21.4% | Modern conversational English, markdown bullet formatting |
| **AI-Hub `71902` (멀티턴 대화)** | English General Dialogue (1T) | 2,073 | 15.3% | Natural English dialogue turn-taking |
| **AI-Hub `71902` (멀티턴 대화)** | Korean General Dialogue (1T/2T) | 1,297 | 9.6% | Natural native Korean honorifics and turn-taking |
| **AI-Hub `71902` (멀티턴 대화)** | English Math & Reasoning (1T) | 1,989 | 14.7% | STEM reasoning and problem solving |
| **AI-Hub `71902` (멀티턴 대화)** | English Code (1T) | 677 | 5.0% | Concise Python and algorithmic functions |
| **AI-Hub `71902` (멀티턴 대화)** | Korean Math & Reasoning (1T) | 800 | 5.9% | Korean mathematical logic and deductive reasoning |
| **AI-Hub `71890` (사후학습용)** | Korean Commonsense (03) | 400 | 2.9% | Filtered, clean historical & cultural commonsense QA |
| **AI-Hub `71902` (멀티턴 대화)** | Korean & English Culture | 320 | 2.4% | Cultural knowledge without verbosity |
| **AI-Hub `71902` (멀티턴 대화)** | Korean Code | 99 | 0.7% | Korean Python coding instruction |

- **Language Balance**: Korean 42.7% (5,796 samples) / English 57.3% (7,764 samples).
- **Split**: 12,882 Train (`data/sft/train.jsonl`, 18.04 MB) / 678 Val (`data/sft/val.jsonl`, 0.95 MB).

---

## 3. Data Cleaning & Tokenizer Safeguards

1. **Strict Masking Regex Filter**:
   - Historical `#이름#`, `#주소#`, `#소속#` anomalies are completely eliminated via `MASK_PATTERN = re.compile(r"#(?:이름|주소|소속|전화번호|주민등록번호|기관|날짜|장소|상호|생년월일|계좌번호)#")`.
   - Samples containing any matching pattern are discarded.
2. **CoT Reasoning Tag Removal**:
   - AI-Hub 71902 `<think>...</think>` internal monologues are stripped from assistant responses to retain direct, concise answers suitable for a 221M parameter model.
3. **Hard Token Length Constraint ($\le 768$ Tokens)**:
   - 100% of samples are pre-verified with the canonical 49,152 Jerboa tokenizer to fit within 768 tokens.
   - Prevents truncation mid-generation and guarantees that the model always learns the termination token (`<|im_end|>`).

---

## 4. Training Pipeline Enhancements (`pipeline/sft.py`)

- **ChatML Response Loss Masking**:
  - The role header (`<|im_start|>{role}\n`) is masked with `labels = -100` across all turns.
  - Non-assistant turns (`system`, `user`, `tool`) are completely masked with `-100`.
  - Only the assistant response body and its trailing `<|im_end|>\n` receive gradient updates.
- **Cosine Learning Rate Schedule**:
  - `get_cosine_schedule_with_warmup` with 5% warmup steps and cosine decay down to 0.
  - Default learning rate: `1.0e-4` (conservative to protect pre-trained base representations).
- **Apple Silicon (MPS) Resource Guard & Stability**:
  - `grad_accum: 2` with `batch_size: 4` -> effective batch size of 8 (~1,610 optimizer steps/epoch).
  - `SystemResourceGuard`: Enforces 60% MPS allocation limit and 85% system RAM threshold; renices training process to background priority to prevent macOS UI stuttering.
  - Periodic MPS memory cleanup: `torch.mps.empty_cache()` every 50 global steps.
  - `SleepGuard`: Invokes `caffeinate` to prevent system sleep during long training runs.
  - Validation set loss evaluation performed at epoch completion.
- **Resume & Fault Tolerance**:
  - Full optimizer and scheduler state preservation in step/epoch checkpoints.
