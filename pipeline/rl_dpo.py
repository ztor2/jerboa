"""Direct Preference Optimization (DPO) pipeline for JerboaLM.

Implements pairwise preference alignment without requiring a separate reward model or critic:
Loss = -log(sigmoid(beta * (log_pi(y_w|x)/log_ref(y_w|x) - log_pi(y_l|x)/log_ref(y_l|x))))
"""

import argparse
import copy
import os
import sys
import time
from typing import Dict, List, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from model.configuration_jerboa import JerboaConfig
from model.modeling_jerboa import JerboaForCausalLM
from model.tokenizer import get_default_tokenizer

DEFAULT_DPO_PAIRS = [
    {
        "prompt": "<|im_start|>user\nWhat is the best way to write clean code?<|im_end|>\n<|im_start|>assistant\n",
        "chosen": "Write simple, self-explanatory code with clear variable names, modular functions, and concise comments where necessary.<|im_end|>",
        "rejected": "Just put everything in one big function and don't bother naming variables well since nobody reads them.<|im_end|>",
    },
    {
        "prompt": "<|im_start|>user\nWhy do we use RMSNorm instead of LayerNorm in modern LLMs?<|im_end|>\n<|im_start|>assistant\n",
        "chosen": "RMSNorm simplifies computation by discarding the mean-centering step, reducing training latency by 10-50% while preserving normalization quality.<|im_end|>",
        "rejected": "RMSNorm is completely identical to LayerNorm with no speed or architectural differences.<|im_end|>",
    },
    {
        "prompt": "<|im_start|>user\nCan you explain what QK-Norm does?<|im_end|>\n<|im_start|>assistant\n",
        "chosen": "QK-Norm normalizes Query and Key vectors before the dot product, preventing logit blowup and stabilizing attention entropy during long-context training.<|im_end|>",
        "rejected": "QK-Norm is a technique used for audio compression that has nothing to do with attention.<|im_end|>",
    },
]


def compute_log_probs(model: torch.nn.Module, input_ids: torch.Tensor, attention_mask: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Compute per-token log-probabilities for the response tokens (where labels != -100)."""
    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    logits = outputs.logits  # [B, seq_len, vocab_size]

    # Shift logits and labels for next-token prediction
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()

    log_probs = F.log_softmax(shift_logits, dim=-1)
    # Gather log prob of true tokens
    per_token_logps = torch.gather(log_probs, dim=2, index=shift_labels.unsqueeze(2).clamp(min=0)).squeeze(2)

    loss_mask = shift_labels != -100
    sum_logps = (per_token_logps * loss_mask).sum(dim=-1)
    return sum_logps


class DPODataset(Dataset):
    """Dataset producing chosen and rejected token tensors with prompt masking."""

    def __init__(self, data: List[Dict], tokenizer, max_length: int = 256):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.pairs = []

        for item in data:
            prompt = item["prompt"]
            chosen = item["chosen"]
            rejected = item["rejected"]

            p_tokens = tokenizer.encode(prompt, add_special_tokens=False)
            c_tokens = tokenizer.encode(chosen, add_special_tokens=False)
            r_tokens = tokenizer.encode(rejected, add_special_tokens=False)

            def prepare_seq(resp_tokens):
                full_ids = p_tokens + resp_tokens
                labels = [-100] * len(p_tokens) + resp_tokens
                if len(full_ids) > max_length:
                    full_ids = full_ids[:max_length]
                    labels = labels[:max_length]
                pad_len = max_length - len(full_ids)
                attn_mask = [1] * len(full_ids) + [0] * pad_len
                full_ids = full_ids + [tokenizer.pad_token_id] * pad_len
                labels = labels + [-100] * pad_len
                return (
                    torch.tensor(full_ids, dtype=torch.long),
                    torch.tensor(attn_mask, dtype=torch.long),
                    torch.tensor(labels, dtype=torch.long),
                )

            c_ids, c_mask, c_labels = prepare_seq(c_tokens)
            r_ids, r_mask, r_labels = prepare_seq(r_tokens)

            self.pairs.append({
                "chosen_ids": c_ids,
                "chosen_mask": c_mask,
                "chosen_labels": c_labels,
                "rejected_ids": r_ids,
                "rejected_mask": r_mask,
                "rejected_labels": r_labels,
            })

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        return self.pairs[idx]


def run_dpo(
    model_path: Optional[str] = None,
    data_path: Optional[str] = None,
    output_dir: str = "checkpoints/dpo",
    beta: float = 0.1,
    steps: int = 30,
    batch_size: int = 2,
    lr: float = 5e-6,
):
    os.makedirs(output_dir, exist_ok=True)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device} ({'Apple Silicon Metal' if device.type == 'mps' else 'CPU'})")

    tokenizer = get_default_tokenizer(model_path if model_path and os.path.exists(model_path) else None)

    # 1. Initialize Policy Model
    if model_path and os.path.exists(model_path):
        print(f"Loading policy model from {model_path}...")
        policy_model = JerboaForCausalLM.from_pretrained(model_path)
    else:
        print("Initializing Jerboa model for DPO alignment demonstration...")
        config = JerboaConfig(
            vocab_size=len(tokenizer),
            hidden_size=768,
            intermediate_size=2048,
            num_hidden_layers=16,
            num_attention_heads=12,
            num_key_value_heads=4,
            tie_word_embeddings=True,
            qk_norm=True,
        )
        policy_model = JerboaForCausalLM(config)

    policy_model.to(device)

    # 2. Reference Model (frozen copy)
    print("Creating frozen reference model (π_ref)...")
    ref_model = copy.deepcopy(policy_model)
    ref_model.eval()
    for p in ref_model.parameters():
        p.requires_grad = False
    ref_model.to(device)

    # 3. Dataset
    import json
    if data_path and os.path.exists(data_path):
        with open(data_path, "r", encoding="utf-8") as f:
            pairs = json.load(f)
        print(f"Loaded {len(pairs)} preference pairs from '{data_path}'")
        pairs = pairs * max(1, (steps * batch_size // len(pairs)) + 1)
    else:
        pairs = DEFAULT_DPO_PAIRS * 10

    dataset = DPODataset(pairs, tokenizer)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    optimizer = torch.optim.AdamW(policy_model.parameters(), lr=lr)

    # 4. DPO Training Loop
    print(f"\n--- Starting Direct Preference Optimization (DPO, Beta={beta}, Steps={steps}) ---")
    policy_model.train()
    step = 0
    start_time = time.time()
    data_iter = iter(dataloader)

    while step < steps:
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            batch = next(data_iter)

        c_ids = batch["chosen_ids"].to(device)
        c_mask = batch["chosen_mask"].to(device)
        c_lbls = batch["chosen_labels"].to(device)

        r_ids = batch["rejected_ids"].to(device)
        r_mask = batch["rejected_mask"].to(device)
        r_lbls = batch["rejected_labels"].to(device)

        # Policy logprobs
        pi_chosen_logps = compute_log_probs(policy_model, c_ids, c_mask, c_lbls)
        pi_rejected_logps = compute_log_probs(policy_model, r_ids, r_mask, r_lbls)

        # Reference logprobs
        with torch.no_grad():
            ref_chosen_logps = compute_log_probs(ref_model, c_ids, c_mask, c_lbls)
            ref_rejected_logps = compute_log_probs(ref_model, r_ids, r_mask, r_lbls)

        # DPO Bradley-Terry Loss: -log(sigmoid(beta * (pi_diff - ref_diff)))
        pi_diff = pi_chosen_logps - pi_rejected_logps
        ref_diff = ref_chosen_logps - ref_rejected_logps
        logits = beta * (pi_diff - ref_diff)
        dpo_loss = -F.logsigmoid(logits).mean()

        # Compute implicit reward margins
        chosen_rewards = (beta * (pi_chosen_logps - ref_chosen_logps)).detach()
        rejected_rewards = (beta * (pi_rejected_logps - ref_rejected_logps)).detach()
        reward_margin = (chosen_rewards - rejected_rewards).mean().item()

        optimizer.zero_grad()
        dpo_loss.backward()
        torch.nn.utils.clip_grad_norm_(policy_model.parameters(), 1.0)
        optimizer.step()

        step += 1
        if step % 5 == 0 or step == steps:
            print(
                f"DPO Step {step:3d}/{steps} | "
                f"Loss: {dpo_loss.item():.4f} | "
                f"Reward Margin: {reward_margin:+.4f} | "
                f"Accuracy: {(logits > 0).float().mean().item() * 100:.1f}%"
            )

    # 5. Save Final DPO Model
    final_path = os.path.join(output_dir, "model")
    policy_model.save_pretrained(final_path)
    tokenizer.save_pretrained(final_path)
    print(f"\nDPO alignment finished in {time.time() - start_time:.2f}s! Saved to {final_path}")
    return final_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Jerboa DPO Alignment")
    default_sft = "checkpoints/sft/model" if os.path.exists("checkpoints/sft/model") else None
    default_data = "data/dpo/sample.json" if os.path.exists("data/dpo/sample.json") else None
    parser.add_argument("--model", type=str, default=default_sft, help="SFT model path")
    parser.add_argument("--data", type=str, default=default_data, help="JSON preference pairs path")
    parser.add_argument("--steps", type=int, default=25, help="Number of DPO steps")
    parser.add_argument("--beta", type=float, default=0.1, help="DPO Beta parameter")
    parser.add_argument("--lr", type=float, default=5e-6, help="Learning rate")
    parser.add_argument("--output_dir", type=str, default="checkpoints/dpo")
    args = parser.parse_args()

    run_dpo(
        model_path=args.model,
        data_path=args.data,
        output_dir=args.output_dir,
        beta=args.beta,
        steps=args.steps,
        lr=args.lr,
    )
