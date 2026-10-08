"""Production-grade Text Inference & Interactive Chat CLI for JerboaLM.

Features:
- Single-prompt generation & Interactive multi-turn Chat REPL
- Token-by-token streaming output (typing effect via TextStreamer)
- Configurable sampling parameters (temperature, top_p, max_new_tokens, system prompt)
"""

import argparse
import os
import sys
import torch
from transformers import TextStreamer

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from model import JerboaConfig, JerboaForCausalLM, get_default_tokenizer


def load_model_and_tokenizer(model_path: str, device: torch.device):
    print(f"Loading checkpoint from '{model_path}' (local or Hugging Face Hub)...")
    tokenizer = None
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    local_tok = os.path.join(project_root, "data", "tokenizer")

    # Prefer local tokenizer if available to ensure exact 49k vocab match
    if os.path.exists(local_tok):
        try:
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(local_tok)
        except Exception:
            pass

    if tokenizer is None:
        try:
            tokenizer = get_default_tokenizer(model_path)
        except Exception:
            tokenizer = get_default_tokenizer()

    model = JerboaForCausalLM.from_pretrained(model_path, trust_remote_code=True)
    model.to(device)
    model.eval()
    return model, tokenizer


def generate_single_prompt(
    model: JerboaForCausalLM,
    tokenizer,
    prompt: str,
    system_prompt: str = "You are Jerboa, an intelligent and helpful AI assistant.",
    max_new_tokens: int = 80,
    temperature: float = 0.4,
    top_p: float = 0.9,
    repetition_penalty: float = 1.15,
    raw_mode: bool = True,
    stream: bool = True,
):
    device = next(model.parameters()).device
    if raw_mode:
        formatted_prompt = prompt
        print(f"\n[Prompt]: {prompt}\n[Completion]: ", end="", flush=True)
    else:
        formatted_prompt = (
            f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
            f"<|im_start|>user\n{prompt}<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )
        print(f"\n[Prompt]: {prompt}\n[Assistant]: ", end="", flush=True)

    input_ids = tokenizer.encode(formatted_prompt, return_tensors="pt").to(device)
    attention_mask = torch.ones_like(input_ids)
    streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True) if stream else None

    with torch.no_grad():
        output_ids = model.generate(
            input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            do_sample=temperature > 0,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            streamer=streamer,
        )

    if not stream:
        resp = tokenizer.decode(output_ids[0, input_ids.shape[1] :], skip_special_tokens=True)
        print(resp)
    print()
    print()


def interactive_chat_repl(
    model: JerboaForCausalLM,
    tokenizer,
    system_prompt: str = "You are Jerboa, an intelligent and helpful AI assistant.",
    max_new_tokens: int = 150,
    temperature: float = 0.7,
    top_p: float = 0.9,
):
    device = next(model.parameters()).device
    print("=" * 70)
    print("       JERBOA INTERACTIVE CHAT (Type 'exit' or 'quit' to quit)")
    print("=" * 70)

    conversation_history = f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
    streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

    while True:
        try:
            user_input = input("\nYou: ").strip()
            if not user_input:
                continue
            if user_input.lower() in ["exit", "quit", "q"]:
                print("Exiting chat. Goodbye!")
                break

            conversation_history += f"<|im_start|>user\n{user_input}<|im_end|>\n<|im_start|>assistant\n"
            input_ids = tokenizer.encode(conversation_history, return_tensors="pt").to(device)
            attention_mask = torch.ones_like(input_ids)

            print("Jerboa: ", end="", flush=True)
            with torch.no_grad():
                output_ids = model.generate(
                    input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    do_sample=temperature > 0,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                    streamer=streamer,
                )

            assistant_reply = tokenizer.decode(output_ids[0, input_ids.shape[1] :], skip_special_tokens=True)
            conversation_history += f"{assistant_reply}<|im_end|>\n"

        except KeyboardInterrupt:
            print("\nExiting chat session.")
            break


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Jerboa Inference CLI")
    default_model = "ztor2/jerboa-pretrain-checkpoints"
    parser.add_argument("--model", type=str, default=default_model, help="Path to local checkpoint or HF Hub repo id (default: ztor2/jerboa-pretrain-checkpoints)")
    parser.add_argument("--prompt", type=str, default=None, help="Input prompt text")
    parser.add_argument("--chat", action="store_true", help="Launch interactive multi-turn chat session with ChatML format")
    parser.add_argument("--instruct", action="store_true", help="Format prompt as instruction (ChatML)")
    parser.add_argument("--max_tokens", type=int, default=80)
    parser.add_argument("--temperature", type=float, default=0.4)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--repetition_penalty", type=float, default=1.15)
    parser.add_argument("--system", type=str, default="You are Jerboa, an intelligent and helpful AI assistant.")
    parser.add_argument("--device", type=str, default=None, help="Device to run inference on (cuda, mps, cpu, or auto)")
    args = parser.parse_args()

    if args.device:
        dev = torch.device(args.device)
    elif torch.cuda.is_available():
        dev = torch.device("cuda:0")
    elif torch.backends.mps.is_available():
        dev = torch.device("mps")
    else:
        dev = torch.device("cpu")
    loaded_model, loaded_tokenizer = load_model_and_tokenizer(args.model, dev)

    if args.chat:
        interactive_chat_repl(
            loaded_model,
            loaded_tokenizer,
            system_prompt=args.system,
            max_new_tokens=args.max_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
        )
    else:
        sample_prompt = args.prompt or "Artificial intelligence is a branch of computer science that"
        generate_single_prompt(
            loaded_model,
            loaded_tokenizer,
            prompt=sample_prompt,
            system_prompt=args.system,
            max_new_tokens=args.max_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            repetition_penalty=args.repetition_penalty,
            raw_mode=not args.instruct,
        )
