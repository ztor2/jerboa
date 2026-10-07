# Jerboa Research Notes

---

## 1. Model Scaling & Depth Allocation (Deep & Thin Paradigm)

### Decision
- **Architecture**: 28 Layers, $d_{model}=768$, $d_{ffn}=2048$ (SwiGLU), $H_q=12$, $H_{kv}=4$ (GQA 3:1).
- **Active Parameters**: ~221.4M base / ~228.9M with Multi-Token Prediction (MTP).

### Theoretical & Empirical Basis
- **Deep & Thin vs. Wide & Shallow**:
  - Classical scaling (GPT-2, early Hugging Face SLMs) favored wider hidden states with fewer layers (e.g., 12–16 layers, $d_{model}=1024$).
  - Recent breakthroughs (**MobileLLM** [Meta, 2024], **SmolLM** [Hugging Face, 2024]) proved that in the sub-1B parameter regime, **layer depth dominates reasoning capacity and compositional capability**.
  - Deeper models enable more sequential reasoning steps in the residual stream without increasing per-step activation memory.
- **Tied Word Embeddings ($W_{lm\_head} = W_{embed}^T$)**:
  - With a 49,152 vocabulary, an untied output head consumes $49,152 \times 768 \approx 37.7\text{M}$ additional parameters (~17% of total parameter budget).
  - Tying embeddings reallocates those 37.7M parameters into transformer depth (+5 layers), providing significantly higher marginal intelligence.

---

## 2. Bilingual Tokenizer & Vocabulary Allocation

### Decision
- **Algorithm**: Byte-Level Byte-Pair Encoding (BPE), NFKC normalization, ChatML format.
- **Vocab Size**: **49,152** (Option B).
- **Corpus Mixture**: 790.34 MB (AI-Hub Knowledge 71894, Dialogue 71908, KoAlpaca, FineWeb-Edu, Wikitext, Python Code).

### Engineering Rationale & Industry Precedents
1. **The SmolLM Bottleneck**:
   - SmolLM-135M allocated >99% of its 49k vocabulary to English and Western European text.
   - For Korean, lack of subwords forced text into 3-byte UTF-8 sequences. A single sentence consumed **102 tokens** in SmolLM vs. **19 tokens** in JerboaLM (**-81.4% reduction, 5.4x compression**).
   - This 5.4x token bloat squanders effective context window and multiplies training FLOPs on Korean by 4x–5x.
2. **Vocab Budgeting for SLMs**:
   - While frontier models (Llama 3 @ 128k, Gemma 2 @ 256k) afford giant vocabularies because embeddings represent <6% of an 8B+ model, in a 200M model a 128k vocab would consume ~98M parameters (>45% of the model).
   - 49,152 is the exact sweet spot: large enough to capture rich Korean morphemes and Python syntax, small enough to keep embedding footprint under 17% of parameters.
3. **Lossless Fallback (0% Out-of-Vocabulary / OOV)**:
   - Root alphabet initializes with all 256 raw byte values (`0x00`–`0xFF`).
   - Unfamiliar words, typos, ancient glyphs (e.g., `𪚥`), or new emojis (`🫠`) decompose into UTF-8 bytes gracefully without producing `<unk>` tokens. Original strings are 100% losslessly reconstructed upon decoding.

---

## 3. Context Length & RoPE Theta ($\theta$) Scaling

### Decision
- **Native Context Window**: 8,192 tokens.
- **RoPE Base Frequency ($\theta$)**: $500,000.0$.
- **Long-context Extrapolation**: YaRN scaling (32K via $4\times$, 64K via $8\times$).

### Research References
1. **CodeLlama & Llama-3 Frequency Precedents**:
   - Meta originally trained Llama 2 with $\theta = 10,000$ (4K context). For **CodeLlama** [Rozière et al., 2023], they scaled $\theta$ from $10,000 \rightarrow 1,000,000$ ($100\times$) and continued pre-training for just 20B tokens, expanding the context to 100K with zero regression on short sequences.
   - **Llama 3** [Meta, 2024] adopted $\theta = 500,000.0$ natively from the first pre-training token, demonstrating that a high base frequency does not hurt short-context modeling while natively supporting long sequences.
   - **Qwen 2.5** [Alibaba, 2024] uses $\theta = 1,000,000.0$ natively for 128K context.
2. **Interleaved Sliding Window Attention (SWA)**:
   - At 32K or 64K, full attention KV cache becomes prohibitive.
   - JerboaLM restricts 21 of 28 layers to a 2,048-token sliding window; every 4th layer (7 global layers) maintains full context.
   - Cuts KV cache memory by **>60%** while retaining global recall across document boundaries.

---

## 4. Quantization Strategy & QAT Timing

### Decision
- **Pre-training from scratch**: Pure **BF16 (Bfloat16)**.
- **Post-training QAT**: Applied during Supervised Fine-Tuning (SFT) or final calibration.

### Trade-Off Analysis
| Method | Pre-training from Scratch | Post-training (SFT / Calibration) |
| :--- | :--- | :--- |
| **Quantization-Aware Training (QAT)** | **Not recommended**: Straight-Through Estimator (STE) rounding noise destabilizes initial random weight convergence, causes gradient spikes, and adds 20%–30% fake quantization kernel overhead. | **Highly recommended**: Model already has stable semantic representations; adapts smoothly to 4-bit quantization grid within a few thousand steps, preserving 98%–99% of FP16 accuracy. |
| **Hardware Native FP8** | Viable on RTX 5090/Hopper, but 221M static memory is only 3.5 GB and already finishes in ~20 hours. FP8 dynamic scaling complexity yields marginal practical benefit for sub-500M models. | Excellent for high-concurrency inference serving engines. |

---

## 5. Multilingual Interleaving & The "Goldilocks" Switching Granularity

### Problem: Catastrophic Forgetting vs. Momentum Jitter
When training multiple disparate distributions (English web, Korean conversational, Python code):
- **Sequential Training** (e.g. 5B English $\rightarrow$ 5B Korean) triggers **Catastrophic Forgetting**: gradients for Korean overwrite English attention pathways and code syntax.
- **Micro-switching too fast** (e.g. switching every 10–20 steps): AdamW optimizer momentum ($\beta_1=0.9, \beta_2=0.95$) oscillates between token embedding spaces, causing optimization jitter and high I/O overhead.

### The "Goldilocks" Zone
- **Empirical Reference**: *DoReMi: Optimizing Data Mixtures for Language Modeling* [Xie et al., NeurIPS 2023] and MosaicML MPT training curricula demonstrate that the ideal single-domain persistence block is **3M to 10M tokens (~100 to 300 optimizer steps, 1–3 minutes)**.
- **JerboaLM Blocked Schedule**:
  - `docs_per_chunk: 1500` ($\approx 2.5\text{M}$ tokens per chunk).
  - `interleave_pattern`: `"fineweb,fineweb,aihub,aihub,fineweb,code"`.
  - **FineWeb Block** (2 chunks $\approx$ 5.0M tokens, ~150 steps): Builds sustained English semantic momentum.
  - **AI-Hub Block** (2 chunks $\approx$ 5.0M tokens, ~150 steps): Deep Korean morpheme and cultural grounding.
  - **Code Block** (1 chunk $\approx$ 2.5M tokens, ~75 steps): Syntactic structure and causal chain reinforcement.
  - Overall ratio: **50% English, 35% Korean, 15% Code**.

---

## 6. Compute & Hardware Scaling (RTX 5090 Baseline)

### Theoretical Calculation (Chinchilla $6N$ Rule)
- For $N = 221.4\text{M}$ parameters and $D = 10\text{ Billion tokens}$:
  $$\text{Total Compute} \approx 6 \times (2.214 \times 10^8) \times (10 \times 10^9) \approx \mathbf{1.33 \times 10^{19}\text{ FLOPs}}$$

### Hardware Realization
- **RTX 5090 Specs**: 32 GB GDDR7 (~1,792 GB/s bandwidth), ~450 TFLOPS BF16 Tensor Core peak.
- **Model FLOPs Utilization (MFU)**: 35% ~ 45% (conservative to tuned with PyTorch SDPA).
- **Throughput & Wall-Clock Estimates**:
  - **Single RTX 5090**: ~55,000 to ~75,000 tok/s $\rightarrow$ **~37 to 50 hours** (~$30–$40 cloud cost).
  - **Dual RTX 5090 (2x DDP)**: ~105,000 to ~140,000 tok/s $\rightarrow$ **~20 to 27 hours (< 1 day)** (~$35–$45 cloud cost).
- **VRAM Headroom**:
  - Static model state (BF16 weights + BF16 grads + FP32 master weights + AdamW states) = **~3.5 GB**.
  - **>28 GB of free VRAM** remains for sequence length 2,048 activation caching, allowing large micro-batches (8–16) without activation checkpointing.
