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
    try:
        print(f"Loading checkpoint from '{model_path}' (local or Hugging Face Hub)...")
        tokenizer = get_default_tokenizer(model_path)
        model = JerboaForCausalLM.from_pretrained(model_path)
    except Exception as e:
        print(f"Notice: Could not load from '{model_path}' ({e}), initializing base JerboaLM...")
        tokenizer = get_default_tokenizer()
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
        model = JerboaForCausalLM(config)

    model.to(device)
    model.eval()
    return model, tokenizer


def generate_single_prompt(
    model: JerboaForCausalLM,
    tokenizer,
    prompt: str,
    system_prompt: str = "You are Jerboa, an intelligent and helpful AI assistant.",
    max_new_tokens: int = 120,
    temperature: float = 0.7,
    top_p: float = 0.9,
    stream: bool = True,
):
    device = next(model.parameters()).device
    formatted_prompt = (
        f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
        f"<|im_start|>user\n{prompt}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )
    input_ids = tokenizer.encode(formatted_prompt, return_tensors="pt").to(device)

    print(f"\n[Prompt]: {prompt}\n")
    print("[Assistant]: ", end="", flush=True)

    streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True) if stream else None

    attention_mask = torch.ones_like(input_ids)

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

    if not stream:
        resp = tokenizer.decode(output_ids[0, input_ids.shape[1] :], skip_special_tokens=True)
        print(resp)
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
    default_model = "checkpoints/sft/model" if os.path.exists("checkpoints/sft/model") else "ztor2/jerboa"
    parser.add_argument("--model", type=str, default=default_model, help="Path to local checkpoint or HF Hub repo id (default: ztor2/jerboa)")
    parser.add_argument("--prompt", type=str, default=None, help="Single prompt mode")
    parser.add_argument("--chat", action="store_true", help="Launch interactive multi-turn chat session")
    parser.add_argument("--max_tokens", type=int, default=100)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--system", type=str, default="You are Jerboa, an intelligent and helpful AI assistant.")
    args = parser.parse_args()

    dev = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
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
        sample_prompt = args.prompt or "Explain how grouped-query attention works in modern LLMs."
        generate_single_prompt(
            loaded_model,
            loaded_tokenizer,
            prompt=sample_prompt,
            system_prompt=args.system,
            max_new_tokens=args.max_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
        )
