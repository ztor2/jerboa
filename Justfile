# Jerboa Command Runner

default:
    @just --list

# Run Rolling-Buffer Pre-training
pretrain *args:
    python pipeline/pretrain.py --recipe recipes/pretrain.yaml {{args}}

# Run Supervised Fine-Tuning
sft *args:
    python pipeline/sft.py --recipe recipes/sft.yaml {{args}}

# Run Direct Preference Optimization
dpo *args:
    python pipeline/rl_dpo.py --recipe recipes/dpo.yaml {{args}}

# Run Group Relative Policy Optimization
grpo *args:
    python pipeline/rl_grpo.py --recipe recipes/grpo.yaml {{args}}

# Run Multimodal Alignment
multimodal *args:
    python pipeline/train_multimodal.py --recipe recipes/multimodal.yaml {{args}}

# Run Apple Silicon (MPS) Benchmark
bench *args:
    python tests/benchmark.py {{args}}

# Launch Interactive Text Chat
chat *args:
    python scripts/infer/text.py --chat {{args}}

# View Training History Ledger
history *args:
    python scripts/tools/history.py {{args}}

# Clean Python caches
clean:
    find . -type d -name "__pycache__" -exec rm -rf {} +
    find . -type f -name "*.pyc" -delete
