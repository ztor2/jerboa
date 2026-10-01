.PHONY: help pretrain sft dpo grpo multimodal bench chat history clean

PYTHON ?= python

help:
	@echo "======================================================================"
	@echo "                       JERBOA WORKFLOW RUNNER                         "
	@echo "======================================================================"
	@echo "  make pretrain    - Run Rolling-Buffer Pre-training (recipes/pretrain.yaml)"
	@echo "  make sft         - Run Supervised Fine-Tuning (recipes/sft.yaml)"
	@echo "  make dpo         - Run Direct Preference Optimization (recipes/dpo.yaml)"
	@echo "  make grpo        - Run Group Relative Policy Optimization (recipes/grpo.yaml)"
	@echo "  make multimodal  - Run Multimodal Alignment (recipes/multimodal.yaml)"
	@echo "  make bench       - Run Apple Silicon (MPS) Throughput Benchmark"
	@echo "  make chat        - Launch Interactive Text Chat REPL"
	@echo "  make history     - View Training Provenance Ledger & Loss Curves"
	@echo "  make clean       - Remove Python cache files"
	@echo "======================================================================"

pretrain:
	$(PYTHON) pipeline/pretrain.py --recipe recipes/pretrain.yaml

sft:
	$(PYTHON) pipeline/sft.py --recipe recipes/sft.yaml

dpo:
	$(PYTHON) pipeline/rl_dpo.py --recipe recipes/dpo.yaml

grpo:
	$(PYTHON) pipeline/rl_grpo.py --recipe recipes/grpo.yaml

multimodal:
	$(PYTHON) pipeline/train_multimodal.py --recipe recipes/multimodal.yaml

bench:
	$(PYTHON) tests/benchmark.py

chat:
	$(PYTHON) scripts/infer/text.py --chat

history:
	$(PYTHON) scripts/tools/history.py

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
