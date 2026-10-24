# JerboaLM Supervised Fine-Tuning (SFT) Plan & Dataset Design

This document details the special token taxonomy, data harmonization strategy, pipeline architecture, and recipe for Supervised Fine-Tuning (SFT) of JerboaLM (~221.4M parameters).

---

## 1. Objectives & Scope

- **Primary Goal**: Transition JerboaLM from a pre-trained causal base model (`checkpoints/pretrain/model`) into an instruction-following assistant supporting general conversation, step-by-step reasoning (`<think>`), code generation, and function calling (`<tool_call>`).
- **Focus**: Natural Korean and English turn-taking, polite tone (해요체/하십시오체), clean markdown formatting, strict special token hygiene, and reliable sequence termination (`<|im_end|>`).

---

## 2. Canonical Special Token Taxonomy (IDs 0–15)

The Jerboa 49,152-token vocabulary reserves IDs `0–15` for structural, reasoning, tool-use, and multimodal control:

| ID | Token | Category | Role & Protocol Specification |
| :---: | :--- | :--- | :--- |
| **0** | `<|pad|>` | Control | Right-padding token (masked with `-100` in labels) |
| **1** | `<|im_start|>` | ChatML | Turn start delimiter (`<|im_start|>{role}\n`) |
| **2** | `<|im_end|>` | ChatML | Turn end delimiter (`<|im_end|>\n`, EOS target) |
| **3** | `<|unk|>` | Control | Unknown fallback token |
| **4** | `<|endoftext|>` | Control | Document boundary token |
| **5** | `<think>` | Reasoning | Chain-of-Thought (CoT) internal reasoning block start |
| **6** | `</think>` | Reasoning | Chain-of-Thought (CoT) internal reasoning block end |
| **7** | `<answer>` | Reasoning | Final boxed/extracted answer start (aligned with GRPO verifier) |
| **8** | `</answer>` | Reasoning | Final boxed/extracted answer end |
| **9** | `<|image|>` | Multimodal | Vision encoder patch placeholder (`config.image_token_id = 9`) |
| **10** | `<|audio|>` | Multimodal | Audio encoder frame placeholder (`config.audio_token_id = 10`) |
| **11** | `<tool_call>` | Tool Use | Assistant JSON function/tool invocation block start |
| **12** | `</tool_call>` | Tool Use | Assistant JSON function/tool invocation block end |
| **13** | `<tool_response>` | Tool Use | Environment/tool execution output start (`role == "tool"`) |
| **14** | `</tool_response>` | Tool Use | Environment/tool execution output end |
| **15** | `<|quad_space|>` | Code | 4-space indentation token |

---

## 3. Four-Protocol Dataset Harmonization (`scripts/data/build_trial_sft.py`)

Rather than discarding reasoning or tool-use samples—or mixing unconditioned special tags into plain chat—all sources are deterministically routed into four conditioned protocols:

| Protocol | System Prompt Conditioning | Assistant Target Structure | Share (%) |
| :--- | :--- | :--- | :---: |
| **`chat`** | `You are Jerboa, a helpful, respectful, and intelligent AI assistant.` | Pure conversational response (zero `<think>` or `<tool_call>` tags) | ~78.7% |
| **`reasoning`** | `...Reason step-by-step before providing your final answer.` | `<think>\n{CoT}\n</think>\n{Solution}` (+ `<answer>{X}</answer>` on math) | ~11.0% |
| **`code`** | `You are Jerboa, a helpful, respectful, and intelligent AI assistant.` | Concise explanation + Markdown fenced code blocks | ~7.1% |
| **`tool_call`** | `...with tool-calling capabilities.\n\nYou have access to the following tools:\n<tools>...</tools>` | `<tool_call>[...]</tool_call>` and `<tool_response>...</tool_response>` | ~3.3% |

- **Dataset Split (`data/sft/`)**:
  - `train.jsonl`: 11,655 unique samples ($\le 768$ tokens, 100% tag-balanced).
  - `val.jsonl`: 613 unique samples.
  - `sample.json`: Representative Korean/English samples across all four protocols.
- **Strict Hygiene Checks**:
  - Regex filtering eliminates PII masking placeholders (`#이름#`, `#주소#`, `#소속#`).
  - `validate_protocol_hygiene()` verifies 100% opening/closing tag symmetry (`<think>` == `</think>`, `<answer>` == `</answer>`, `<tool_call>` == `</tool_call>`) and ensures zero cross-protocol tag leakage.

---

## 4. Model & Training Pipeline Safeguards (`model/modeling.py`, `pipeline/sft.py`)

- **Causal + Padding Attention Masking (`model/modeling.py`)**:
  - When a 2D right-padded `attention_mask` is supplied during SFT, `JerboaAttention.forward` explicitly combines the lower-triangular causal mask (`k_pos <= q_pos`), optional Sliding Window Attention mask, and 2D padding mask so padded batches strictly preserve autoregressive causality.
- **ChatML Response Loss Masking (`pipeline/sft.py`)**:
  - Role headers (`<|im_start|>{role}\n`) and non-assistant turns (`system`, `user`, `tool`) are masked with `labels = -100`.
  - Only `assistant` response bodies (including `<think>...</think>`, `<answer>...</answer>`, `<tool_call>...</tool_call>`, and trailing `<|im_end|>\n`) receive gradient updates.
- **Apple Silicon (MPS) Stability**:
  - Gradient checkpointing enabled (`gradient_checkpointing: true`), reducing 28-layer activation memory from ~13 GB to ~2.0 GB without affecting gradient precision.
  - `batch_size: 2`, `grad_accum: 4` (Effective Batch Size = 8).
  - `SystemResourceGuard` (`max_mem_fraction: 0.7`, `system_ram_limit: 85.0%`) and `SleepGuard` (`caffeinate`).
