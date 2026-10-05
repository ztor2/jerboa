.PHONY: help pretrain sft dpo grpo multimodal bench chat history clean upload upload-pretrain upload-sft upload-dpo upload-grpo

PYTHON ?= python
STAGE ?= sft
MODEL ?= checkpoints/$(STAGE)/model
REPO ?= jerboa
PRIVATE ?= 0
DRY_RUN ?= 0
RESUME ?=
CAFFEINATE ?= $(shell command -v caffeinate 2>/dev/null)
RUNNER ?= $(if $(CAFFEINATE),caffeinate -i -s -m,)

help:
	@echo "======================================================================"
	@echo "                       JERBOA WORKFLOW RUNNER                         "
	@echo "======================================================================"
	@echo "  make pretrain    - Run Rolling-Buffer Pre-training (RESUME=auto supported)"
	@echo "  make sft         - Run Supervised Fine-Tuning (RESUME=auto supported)"
	@echo "  make dpo         - Run Direct Preference Optimization (recipes/dpo.yaml)"
	@echo "  make grpo        - Run Group Relative Policy Optimization (recipes/grpo.yaml)"
	@echo "  make multimodal  - Run Multimodal Alignment (recipes/multimodal.yaml)"
	@echo "  make bench       - Run Apple Silicon (MPS) Throughput Benchmark"
	@echo "  make chat        - Launch Interactive Text Chat REPL"
	@echo "  make history     - View Training Provenance Ledger & Loss Curves"
	@echo "  make upload      - Upload checkpoint to HF Hub (STAGE=pretrain|sft|dpo|grpo, default: sft)"
	@echo "  make clean       - Remove Python cache files"
	@echo "======================================================================"

pretrain:
	$(RUNNER) $(PYTHON) pipeline/pretrain.py --recipe recipes/pretrain.yaml $(if $(RESUME),--resume $(RESUME),)

sft:
	$(RUNNER) $(PYTHON) pipeline/sft.py --recipe recipes/sft.yaml $(if $(RESUME),--resume $(RESUME),)

dpo:
	$(RUNNER) $(PYTHON) pipeline/rl_dpo.py --recipe recipes/dpo.yaml

grpo:
	$(RUNNER) $(PYTHON) pipeline/rl_grpo.py --recipe recipes/grpo.yaml

multimodal:
	$(RUNNER) $(PYTHON) pipeline/train_multimodal.py --recipe recipes/multimodal.yaml

bench:
	$(PYTHON) tests/benchmark.py

chat:
	$(PYTHON) scripts/infer/text.py --chat

history:
	$(PYTHON) scripts/tools/history.py

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

upload:
	$(PYTHON) scripts/tools/upload_hub.py \
		--model $(MODEL) \
		--repo_name $(REPO) \
		--stage $(STAGE) \
		$(if $(filter 1 true,$(PRIVATE)),--private,) \
		$(if $(filter 1 true,$(DRY_RUN)),--dry_run,)

upload-pretrain:
	$(MAKE) upload STAGE=pretrain

upload-sft:
	$(MAKE) upload STAGE=sft

upload-dpo:
	$(MAKE) upload STAGE=dpo

upload-grpo:
	$(MAKE) upload STAGE=grpo
