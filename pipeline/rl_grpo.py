"""Group Relative Policy Optimization (GRPO) pipeline for JerboaLM.

State-of-the-art RL alignment method used in DeepSeek-R1 / DeepSeekMath:
- Samples G candidate responses per prompt
- Computes automated verifier rewards (format, correctness, reasoning length)
- Calculates relative advantage within each group: A_i = (r_i - mean(r)) / (std(r) + eps)
- Eliminates the need for a separate Critic / Value network, saving 50%+ memory!
- Perfectly optimized for Apple Silicon (Mac M-series) unified memory
"""

import argparse
import copy
import os
import re
import sys
import time
from typing import Callable, List, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F

from model.config import JerboaConfig
from model.modeling import JerboaForCausalLM
from model.multimodal import JerboaVLForConditionalGeneration
from model.tokenizer import get_default_tokenizer
from pipeline.checkpoint_manager import GracefulInterruptHandler, SleepGuard, get_device

# Example reasoning prompts for GRPO
SAMPLE_PROMPTS = [
    "Solve the math problem: If a train travels 60 miles in 45 minutes, what is its speed in miles per hour? Think step by step and end with <answer>X</answer>.",
    "Solve the math problem: What is 15% of 240? Think step by step and end with <answer>X</answer>.",
    "Solve the puzzle: Alice has 3 brothers and each brother has 2 sisters. How many sisters does Alice have? Think step by step and end with <answer>X</answer>.",
    "Solve the math problem: What is the prime factorization of 60? Think step by step and end with <answer>X</answer>.",
]

GROUND_TRUTHS = {
    0: "80",
    1: "36",
    2: "1",
    3: "2^2 * 3 * 5",
}

SAMPLE_MULTIMODAL_PROMPTS = [
    "Look at this image. How many green circles are visible? Think step by step and end with <answer>X</answer>.",
    "Examine the geometric figure: What is the area of a right-angled triangle with base 8 and height 5? Think step by step and end with <answer>X</answer>.",
    "Inspect the chart: What was the total score recorded in Round 3? Think step by step and end with <answer>X</answer>.",
]

GROUND_TRUTHS_MULTIMODAL = {
    0: "3",
    1: "20",
    2: "150",
}


def rule_based_verifier(completion: str, target: str) -> float:
    """Automated rule-based verifier giving format reward and accuracy reward."""
    reward = 0.0

    # 1. Format reward: Did it follow reasoning structure and use <answer>...</answer>?
    match = re.search(r"<answer>(.*?)</answer>", completion, re.DOTALL)
    if match:
        reward += 0.5
        extracted_answer = match.group(1).strip()
        # 2. Correctness reward
        if target.lower() in extracted_answer.lower():
            reward += 1.0

    # 3. Soft length penalty / bonus (encourages structured reasoning without infinite loop)
    if 20 <= len(completion) <= 500:
        reward += 0.2

    return reward


def compute_sequence_log_prob(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    prompt_len: int,
    pixel_values: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Compute per-token log prob on the generated completion tokens."""
    kwargs = {}
    if pixel_values is not None:
        kwargs["pixel_values"] = pixel_values

    outputs = model(input_ids=input_ids, **kwargs)
    logits = outputs.logits[:, :-1, :]
    targets = input_ids[:, 1:]

    # Shifted prompt boundary
    log_probs = F.log_softmax(logits, dim=-1)
    target_logps = torch.gather(log_probs, dim=2, index=targets.unsqueeze(2)).squeeze(2)

    # Return sum of log probs for response tokens
    resp_logps = target_logps[:, prompt_len - 1 :].sum(dim=-1)
    return resp_logps


def run_grpo(
    model_path: Optional[str] = None,
    data_path: Optional[str] = None,
    output_dir: str = "checkpoints/grpo",
    group_size: int = 4,
    steps: int = 20,
    lr: float = 1e-5,
    clip_eps: float = 0.2,
    beta_kl: float = 0.04,
    multimodal: bool = False,
):
    os.makedirs(output_dir, exist_ok=True)
    device = get_device()
    hw_name = torch.cuda.get_device_name(0) if device.type == "cuda" else ("Apple Silicon Metal" if device.type == "mps" else "CPU")
    print(f"Using device: {device} ({hw_name}) | Multimodal: {multimodal}")

    tokenizer = get_default_tokenizer(model_path if model_path and os.path.exists(model_path) else None)

    # 1. Policy Model
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

    if multimodal:
        print("Initializing Jerboa-VL model for Multimodal GRPO reasoning...")
        policy_model = JerboaVLForConditionalGeneration(config)
        if model_path and os.path.exists(model_path):
            state_dict = None
            if os.path.isdir(model_path):
                st_file = os.path.join(model_path, "model.safetensors")
                pt_file = os.path.join(model_path, "model.pt")
                if os.path.exists(st_file):
                    from safetensors.torch import load_file
                    state_dict = load_file(st_file)
                elif os.path.exists(pt_file):
                    state_dict = torch.load(pt_file, map_location=device, weights_only=True)
            elif os.path.isfile(model_path):
                state_dict = torch.load(model_path, map_location=device, weights_only=True)

            if state_dict is not None:
                if not any(k.startswith("vision_") or k.startswith("audio_") or k.startswith("language_model.") for k in state_dict.keys()):
                    print(f"Loading base text weights into multimodal language backbone from '{model_path}'...")
                    policy_model.language_model.load_state_dict(state_dict, strict=False)
                else:
                    print(f"Loading multimodal weights from '{model_path}'...")
                    policy_model.load_state_dict(state_dict, strict=False)
    else:
        if model_path and os.path.exists(model_path):
            print(f"Loading policy model from {model_path}...")
            policy_model = JerboaForCausalLM.from_pretrained(model_path)
        else:
            print("Initializing Jerboa model for GRPO alignment...")
            policy_model = JerboaForCausalLM(config)

    policy_model.to(device)

    # 2. Frozen Reference Model for KL divergence regularization
    print("Creating frozen reference model (π_ref)...")
    ref_model = copy.deepcopy(policy_model)
    ref_model.eval()
    for p in ref_model.parameters():
        p.requires_grad = False
    ref_model.to(device)

    # 3. Tasks / Prompts
    import json
    if data_path and os.path.exists(data_path):
        with open(data_path, "r", encoding="utf-8") as f:
            tasks = json.load(f)
        task_prompts = [t["prompt"] for t in tasks]
        task_targets = [t.get("expected_answer", "") for t in tasks]
        print(f"Loaded {len(task_prompts)} verifiable tasks from '{data_path}'")
    else:
        task_prompts = SAMPLE_MULTIMODAL_PROMPTS if multimodal else SAMPLE_PROMPTS
        task_targets = GROUND_TRUTHS_MULTIMODAL if multimodal else GROUND_TRUTHS

    optimizer = torch.optim.AdamW(policy_model.parameters(), lr=lr)

    sleep_guard = SleepGuard()
    sleep_guard.start()
    interrupt_handler = GracefulInterruptHandler()

    stage_desc = "Multimodal GRPO (Visual Reasoning)" if multimodal else "GRPO"
    print(f"\n--- Starting {stage_desc} (Group Size G={group_size}) ---")
    start_time = time.time()

    img_id = tokenizer.convert_tokens_to_ids("<|image|>")
    img_tokens = [img_id] * 49 if multimodal and img_id is not None else []

    try:
        for step in range(1, steps + 1):
            prompt_idx = (step - 1) % len(task_prompts)
            curr_prompt = task_prompts[prompt_idx]
            target = task_targets[prompt_idx]

            if multimodal and "<|image|>" not in curr_prompt:
                p_prefix = tokenizer.encode("<|im_start|>user\n", add_special_tokens=False) + img_tokens
                p_suffix = tokenizer.encode(f"{curr_prompt}<|im_end|>\n<|im_start|>assistant\n", add_special_tokens=False)
                prompt_ids = torch.tensor([p_prefix + p_suffix], dtype=torch.long, device=device)
            else:
                prompt_text = f"<|im_start|>user\n{curr_prompt}<|im_end|>\n<|im_start|>assistant\n"
                prompt_ids = tokenizer.encode(prompt_text, return_tensors="pt").to(device)

            prompt_len = prompt_ids.shape[1]
            pixel_vals = torch.randn(group_size, 3, 224, 224, device=device) if multimodal else None

            # Generate G candidate completions with sampling
            policy_model.eval()
            with torch.no_grad():
                expanded_prompt = prompt_ids.repeat(group_size, 1)
                gen_kwargs = {"pixel_values": pixel_vals} if multimodal else {}
                generated_ids = policy_model.generate(
                    expanded_prompt,
                    max_new_tokens=40,
                    do_sample=True,
                    temperature=0.8,
                    top_p=0.9,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                    **gen_kwargs,
                )

            # Decode candidates and compute verifier rewards
            rewards = []
            for i in range(group_size):
                completion = tokenizer.decode(generated_ids[i, prompt_len:], skip_special_tokens=True)
                r = rule_based_verifier(completion, target)
                rewards.append(r)

            rewards_tensor = torch.tensor(rewards, dtype=torch.float32, device=device)

            # Normalize rewards across the group: A_i = (r_i - mean) / (std + eps)
            r_mean = rewards_tensor.mean()
            r_std = rewards_tensor.std()
            if r_std < 1e-4:
                advantages = torch.zeros_like(rewards_tensor)
            else:
                advantages = (rewards_tensor - r_mean) / (r_std + 1e-4)

            # Train policy on candidate rollouts
            policy_model.train()
            optimizer.zero_grad()

            # Compute log probs under current policy π_θ and ref π_ref
            pi_logps = compute_sequence_log_prob(policy_model, generated_ids, prompt_len, pixel_values=pixel_vals)
            with torch.no_grad():
                ref_logps = compute_sequence_log_prob(ref_model, generated_ids, prompt_len, pixel_values=pixel_vals)

            # Compute ratio and KL penalty
            log_ratio = pi_logps - pi_logps.detach()
            ratio = torch.exp(log_ratio)

            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * advantages
            policy_loss = -torch.min(surr1, surr2).mean()

            # Approximate KL divergence: KL(π || π_ref) = exp(ref_logps - pi_logps) - (ref_logps - pi_logps) - 1
            kl_div = (torch.exp(ref_logps - pi_logps) - (ref_logps - pi_logps) - 1.0).mean()
            total_loss = policy_loss + beta_kl * kl_div

            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(policy_model.parameters(), 1.0)
            optimizer.step()

            if step % 2 == 0 or step == steps:
                print(
                    f"GRPO Step {step:2d}/{steps} | "
                    f"Avg Reward: {r_mean.item():.3f} | "
                    f"Max Reward: {rewards_tensor.max().item():.3f} | "
                    f"Policy Loss: {policy_loss.item():.4f} | "
                    f"KL: {kl_div.item():.4f}"
                )

            # Graceful Interrupt Handling
            if interrupt_handler.interrupted:
                print(f"\n[Interrupt] Saving emergency GRPO model at step {step}...")
                interrupted_path = os.path.join(output_dir, f"interrupted_step_{step:04d}")
                os.makedirs(interrupted_path, exist_ok=True)
                if multimodal:
                    torch.save(policy_model.state_dict(), os.path.join(interrupted_path, "multimodal_grpo.pt"))
                else:
                    policy_model.save_pretrained(interrupted_path)
                    tokenizer.save_pretrained(interrupted_path)
                print(f"[Interrupt] Safely saved to {interrupted_path}. GRPO paused.")
                return interrupted_path

        final_path = os.path.join(output_dir, "model")
        os.makedirs(final_path, exist_ok=True)
        if multimodal:
            torch.save(policy_model.state_dict(), os.path.join(final_path, "multimodal_grpo.pt"))
            print(f"\nMultimodal GRPO alignment finished in {time.time() - start_time:.2f}s! Saved weights to {final_path}/multimodal_grpo.pt")
        else:
            policy_model.save_pretrained(final_path)
            tokenizer.save_pretrained(final_path)
            print(f"\nGRPO alignment finished in {time.time() - start_time:.2f}s! Saved to {final_path}")
        return final_path
    finally:
        sleep_guard.stop()


if __name__ == "__main__":
    from pipeline.recipe import apply_recipe

    parser = argparse.ArgumentParser(description="Jerboa GRPO Alignment")
    default_sft = "checkpoints/sft/model" if os.path.exists("checkpoints/sft/model") else None
    default_data = "data/grpo/sample.json" if os.path.exists("data/grpo/sample.json") else None
    parser.add_argument("--recipe", type=str, default="recipes/grpo.yaml", help="Path to YAML training recipe")
    parser.add_argument("--model", type=str, default=default_sft, help="Base/SFT model path")
    parser.add_argument("--data", type=str, default=default_data, help="JSON verifiable tasks path")
    parser.add_argument("--steps", type=int, default=10, help="GRPO steps")
    parser.add_argument("--group_size", type=int, default=4, help="Group size G")
    parser.add_argument("--lr", type=float, default=1e-5, help="Learning rate")
    parser.add_argument("--output_dir", type=str, default="checkpoints/grpo")
    parser.add_argument("--multimodal", action="store_true", help="Enable Multimodal GRPO (visual reasoning)")
    args = parser.parse_args()
    args = apply_recipe(args, "recipes/grpo.yaml")

    run_grpo(
        model_path=args.model,
        data_path=args.data,
        output_dir=args.output_dir,
        group_size=args.group_size,
        steps=args.steps,
        lr=args.lr,
        multimodal=args.multimodal,
    )
