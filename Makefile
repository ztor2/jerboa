.PHONY: help pretrain sft dpo grpo multimodal tokenizer bench chat history clean upload upload-pretrain upload-sft upload-dpo upload-grpo

VENV_PYTHON := $(shell if [ -f .venv/bin/python ]; then echo .venv/bin/python; elif command -v python3 >/dev/null 2>&1; then echo python3; else echo python; fi)
PYTHON ?= $(VENV_PYTHON)
STAGE ?= sft
MODEL ?= checkpoints/$(STAGE)/model
REPO ?= jerboa
PRIVATE ?= 0
DRY_RUN ?= 0
RESUME ?=
CAFFEINATE ?= $(shell command -v caffeinate 2>/dev/null)
RUNNER ?= $(if $(CAFFEINATE),caffeinate -i -s -m,)
TORCHRUN := $(PYTHON) -m torch.distributed.run --master_addr=127.0.0.1
GPUS ?= 2

help:
	@echo "======================================================================"
	@echo "                       JERBOA WORKFLOW RUNNER                         "
	@echo "======================================================================"
	@echo "  make pretrain         - Run Pre-training (balanced multitasking mode, recipes/pretrain.yaml)"
	@echo "  make pretrain-fast    - Run High-Throughput Pre-training (dedicated mode, batch=8, recipes/pretrain_fast.yaml)"
	@echo "  make pretrain-ddp     - Run Distributed Multi-GPU Pre-training (GPUS=2, torchrun)"
	@echo "  make sft              - Run Supervised Fine-Tuning (RESUME=auto supported)"
	@echo "  make dpo              - Run Direct Preference Optimization (recipes/dpo.yaml)"
	@echo "  make grpo             - Run Group Relative Policy Optimization (recipes/grpo.yaml)"
	@echo "  make multimodal       - Run Multimodal Alignment (recipes/multimodal.yaml)"
	@echo "  make bench            - Run Apple Silicon (MPS) Throughput Benchmark"
	@echo "  make chat             - Launch Interactive Text Chat REPL"
	@echo "  make history          - View Training Provenance Ledger & Loss Curves"
	@echo "  make upload           - Upload checkpoint to HF Hub (STAGE=pretrain|sft|dpo|grpo, default: sft)"
	@echo "  make clean            - Remove Python cache files"
	@echo "======================================================================"

pretrain:
	$(RUNNER) $(PYTHON) pipeline/pretrain.py --recipe $(or $(RECIPE),recipes/pretrain.yaml) $(if $(RESUME),--resume $(RESUME),)

pretrain-fast:
	$(RUNNER) $(PYTHON) pipeline/pretrain.py --recipe $(or $(RECIPE),recipes/pretrain_fast.yaml) $(if $(RESUME),--resume $(RESUME),)

pretrain-ddp:
	$(TORCHRUN) --nproc_per_node=$(GPUS) pipeline/pretrain.py --recipe $(or $(RECIPE),recipes/pretrain.yaml) $(if $(RESUME),--resume $(RESUME),)

pretrain-runpod:
	$(TORCHRUN) --nproc_per_node=$(GPUS) pipeline/pretrain.py --recipe recipes/pretrain_runpod.yaml $(if $(RESUME),--resume $(RESUME),)

sft:
	$(RUNNER) $(PYTHON) pipeline/sft.py --recipe recipes/sft.yaml $(if $(RESUME),--resume $(RESUME),)

dpo:
	$(RUNNER) $(PYTHON) pipeline/rl_dpo.py --recipe recipes/dpo.yaml

grpo:
	$(RUNNER) $(PYTHON) pipeline/rl_grpo.py --recipe recipes/grpo.yaml

multimodal:
	$(RUNNER) $(PYTHON) pipeline/train_multimodal.py --recipe recipes/multimodal.yaml

tokenizer:
	$(PYTHON) scripts/tokenizer/build_tokenizer.py --vocab-size $(or $(VOCAB_SIZE),49152)

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
	$(MAKE) upload STAGE=pretrain REPO=$(or $(REPO),jerboa-base)

upload-sft:
	$(MAKE) upload STAGE=sft REPO=$(or $(REPO),jerboa-sft)

upload-dpo:
	$(MAKE) upload STAGE=dpo REPO=$(or $(REPO),jerboa-dpo)

upload-grpo:
	$(MAKE) upload STAGE=grpo REPO=$(or $(REPO),jerboa-grpo)
