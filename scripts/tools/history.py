"""Inspect training history, data lineage, and loss curves for rolling-buffer training."""

import json
import os
import sys

LINEAGE_LOG = "data/manifests/dataset_lineage.jsonl"
METRICS_LOG = "data/manifests/training_metrics.jsonl"


def display_history():
    print("=" * 80)
    print("                 JERBOA PRE-TRAINING DATA PROVENANCE & HISTORY")
    print("=" * 80)

    if not os.path.exists(LINEAGE_LOG):
        print(f"\nNo lineage records found at '{LINEAGE_LOG}'. Run rolling pre-training first:")
        print("  python pipeline/pretrain.py --chunks 3\n")
        return

    records = []
    with open(LINEAGE_LOG, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))

    print(f"\n[1] DATASET LINEAGE & PROVENANCE LEDGER ({len(records)} Chunks Processed)")
    print("-" * 80)
    print(f"{'Chunk':<12} | {'Doc Range':<15} | {'Tokens':<10} | {'Cumul. Tokens':<14} | {'Loss (Init->End)':<18}")
    print("-" * 80)

    total_tokens = 0
    total_docs = 0

    for r in records:
        chunk_id = r.get("chunk_id", "N/A")
        doc_range = r.get("document_range", "N/A")
        tokens = r.get("tokens_in_chunk", 0)
        cumul = r.get("cumulative_tokens", 0)
        i_loss = r.get("initial_loss", 0.0)
        f_loss = r.get("final_loss", 0.0)
        loss_str = f"{i_loss:.2f} -> {f_loss:.2f}"
        total_tokens = max(total_tokens, cumul)
        total_docs += r.get("doc_count", 0)

        print(f"{chunk_id:<12} | {doc_range:<15} | {tokens:<10,} | {cumul:<14,} | {loss_str:<18}")

    print("-" * 80)
    print(f"Total Documents Consumed: {total_docs:,}")
    print(f"Total Tokens Trained:    {total_tokens:,}")

    print("\n[2] CRYPTOGRAPHIC DATA INTEGRITY AUDIT (SHA-256 Hashes Before Deletion)")
    print("-" * 80)
    for r in records:
        chunk_id = r.get("chunk_id", "N/A")
        sha = r.get("sha256_hash", "N/A")
        status = r.get("status", "unknown")
        print(f"  {chunk_id}: SHA-256 = {sha} [{status.upper()}]")

    print("\n[3] RECENT TRAINING STEP METRICS")
    print("-" * 80)
    if os.path.exists(METRICS_LOG):
        steps = []
        with open(METRICS_LOG, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    steps.append(json.loads(line))

        print(f"{'Step':<6} | {'Chunk':<10} | {'Loss':<8} | {'Perplexity':<12} | {'LR':<10} | {'Timestamp':<20}")
        print("-" * 80)
        for s in steps[-8:]:
            print(
                f"{s['global_step']:<6} | {s['chunk']:<10} | {s['loss']:<8.4f} | "
                f"{s['perplexity']:<12.2f} | {s['lr']:<10.2e} | {s['timestamp']:<20}"
            )
    print("=" * 80)


if __name__ == "__main__":
    display_history()
