"""Benchmark throughput (tokens/sec) and memory on Apple Silicon (MPS)."""

import argparse
import os
import sys
import time
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.configuration_jerboa import JerboaConfig
from model.modeling_jerboa import JerboaForCausalLM


def benchmark(batch_size: int = 1, prompt_len: int = 128, gen_len: int = 64, num_layers: int = 16):
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"=== Jerboa Benchmark on {device} ({'Apple Silicon Metal' if device.type == 'mps' else 'CPU'}) ===")

    config = JerboaConfig(
        vocab_size=32768,
        hidden_size=768,
        intermediate_size=2048,
        num_hidden_layers=num_layers,
        num_attention_heads=12,
        num_key_value_heads=4,
        tie_word_embeddings=True,
        qk_norm=True,
    )

    model = JerboaForCausalLM(config).to(device)
    model.eval()

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,} ({total_params/1e6:.2f}M)")

    # Warmup
    dummy_input = torch.randint(0, config.vocab_size, (batch_size, 32), device=device)
    with torch.no_grad():
        for _ in range(3):
            _ = model.generate(dummy_input, max_new_tokens=5, do_sample=False)

    # 1. Prefill Latency Test
    prompt = torch.randint(0, config.vocab_size, (batch_size, prompt_len), device=device)
    if device.type == "mps":
        torch.mps.synchronize()
    t0 = time.perf_counter()
    with torch.no_grad():
        for _ in range(5):
            _ = model(input_ids=prompt)
    if device.type == "mps":
        torch.mps.synchronize()
    prefill_time = (time.perf_counter() - t0) / 5
    prefill_tok_per_sec = (batch_size * prompt_len) / prefill_time
    print(f"\n[Prefill] Length: {prompt_len} tokens | Latency: {prefill_time*1000:.2f} ms | Throughput: {prefill_tok_per_sec:.1f} tokens/s")

    # 2. Decoding Generation Test
    if device.type == "mps":
        torch.mps.synchronize()
    t0 = time.perf_counter()
    with torch.no_grad():
        out = model.generate(prompt, max_new_tokens=gen_len, do_sample=False)
    if device.type == "mps":
        torch.mps.synchronize()
    decode_time = time.perf_counter() - t0
    decode_tok_per_sec = (batch_size * gen_len) / decode_time
    print(f"[Decoding] New tokens: {gen_len} | Total time: {decode_time:.3f} s | Speed: {decode_tok_per_sec:.1f} tokens/s")

    # Memory info
    if device.type == "mps" and hasattr(torch.mps, "current_allocated_memory"):
        mem_mb = torch.mps.current_allocated_memory() / (1024 * 1024)
        print(f"[MPS Memory] Allocated: {mem_mb:.1f} MB")

    print("\nBenchmark completed successfully!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Jerboa MPS Benchmark")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--prompt_len", type=int, default=128)
    parser.add_argument("--gen_len", type=int, default=64)
    parser.add_argument("--layers", type=int, default=16)
    args = parser.parse_args()

    benchmark(batch_size=args.batch_size, prompt_len=args.prompt_len, gen_len=args.gen_len, num_layers=args.layers)
