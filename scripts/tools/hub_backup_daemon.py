"""Periodic Hugging Face Hub backup daemon for Jerboa training checkpoints.

Monitors checkpoints/pretrain/steps/ and safely uploads intermediate
checkpoints to a designated Hugging Face Hub repository without blocking training.
"""

import argparse
import glob
import json
import os
import re
import shutil
import sys
import time
from typing import Optional, Set

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from huggingface_hub import HfApi, whoami


def inject_remote_code_and_automap(ckpt_dir: str, project_root: str):
    """Bundles modeling code and auto_map to allow direct loading via trust_remote_code."""
    model_src_dir = os.path.join(project_root, "model")
    for filename in ["__init__.py", "config.py", "modeling.py", "multimodal.py"]:
        src = os.path.join(model_src_dir, filename)
        if os.path.exists(src):
            dst = os.path.join(ckpt_dir, filename)
            shutil.copy2(src, dst)

    manifests_src = os.path.join(project_root, "data", "manifests")
    if os.path.isdir(manifests_src):
        manifests_dst = os.path.join(ckpt_dir, "manifests")
        os.makedirs(manifests_dst, exist_ok=True)
        for fname in ["dataset_lineage.jsonl", "training_metrics.jsonl"]:
            m_src = os.path.join(manifests_src, fname)
            if os.path.exists(m_src):
                shutil.copy2(m_src, os.path.join(manifests_dst, fname))

    config_path = os.path.join(ckpt_dir, "config.json")
    if os.path.exists(config_path):
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                cfg_data = json.load(f)
            if "auto_map" not in cfg_data:
                cfg_data["auto_map"] = {
                    "AutoConfig": "config.JerboaConfig",
                    "AutoModelForCausalLM": "modeling.JerboaForCausalLM",
                }
                with open(config_path, "w", encoding="utf-8") as f:
                    json.dump(cfg_data, f, indent=2)
        except Exception as e:
            print(f"[Backup] Warning: could not update config auto_map: {e}")


def get_chunk_number(folder_name: str) -> Optional[int]:
    """Extract integer chunk index from folder name like 'chunk_0025'."""
    m = re.search(r"chunk_(\d+)", folder_name)
    if m:
        return int(m.group(1))
    return None


def run_backup_daemon(
    steps_dir: str = "checkpoints/pretrain/steps",
    repo_id: str = "ztor2/jerboa-base",
    interval_chunks: int = 50,
    poll_sec: int = 30,
    private: bool = False,
    once: bool = False,
):
    print("==================================================")
    print("  JERBOA HF HUB CHECKPOINT BACKUP DAEMON")
    print("==================================================")
    print(f"Source Directory: {steps_dir}")
    print(f"Target Repository: {repo_id} (private={private})")
    print(f"Backup Interval: Every {interval_chunks} chunks (once={once})")
    print("==================================================")

    token = os.getenv("HF_TOKEN")
    try:
        user_info = whoami(token=token)
        username = user_info["name"]
        print(f"Authenticated as HF user: '{username}'")
    except Exception as e:
        print(f"Error: Not authenticated with Hugging Face Hub: {e}")
        sys.exit(1)

    api = HfApi(token=token)
    try:
        api.create_repo(repo_id=repo_id, repo_type="model", private=private, exist_ok=True)
        print(f"Verified repository: https://huggingface.co/{repo_id}")
    except Exception as e:
        print(f"Notice on repo check: {e}")

    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    staging_dir = os.path.join(os.path.dirname(steps_dir), ".backup_staging")
    history_file = os.path.join(steps_dir, ".backup_history.json")
    uploaded_chunks: Set[str] = set()

    if os.path.exists(history_file):
        try:
            with open(history_file, "r", encoding="utf-8") as f:
                uploaded_chunks = set(json.load(f))
        except Exception:
            pass

    last_uploaded_idx = 0
    for item in uploaded_chunks:
        c_num = get_chunk_number(item)
        if c_num and c_num > last_uploaded_idx:
            last_uploaded_idx = c_num

    print(f"Previously backed up: {len(uploaded_chunks)} checkpoints (highest: chunk_{last_uploaded_idx:04d})")

    try:
        while True:
            if not os.path.exists(steps_dir):
                if once:
                    break
                time.sleep(poll_sec)
                continue

            candidates = sorted([
                d for d in os.listdir(steps_dir)
                if os.path.isdir(os.path.join(steps_dir, d)) and (d.startswith("chunk_") or d.startswith("interrupted_"))
            ])

            target_to_upload: Optional[str] = None

            # Check for emergency interrupt checkpoint first
            for cand in candidates:
                if cand.startswith("interrupted_") and cand not in uploaded_chunks:
                    target_to_upload = cand
                    break

            # If no emergency, check for scheduled interval chunk
            if not target_to_upload:
                for cand in reversed(candidates):
                    if cand in uploaded_chunks:
                        continue
                    c_num = get_chunk_number(cand)
                    if c_num is not None:
                        if once or c_num % interval_chunks == 0 or (c_num - last_uploaded_idx >= interval_chunks):
                            target_to_upload = cand
                            break

            if target_to_upload:
                full_path = os.path.join(steps_dir, target_to_upload)
                pt_file = os.path.join(full_path, "training_state.pt")
                meta_file = os.path.join(full_path, "training_state.json")

                if not (os.path.exists(pt_file) and os.path.exists(meta_file)):
                    if once:
                        break
                    time.sleep(10)
                    continue

                commit_msg = f"Backup {target_to_upload}"
                try:
                    with open(meta_file, "r", encoding="utf-8") as f:
                        meta = json.load(f)
                    avg_loss = meta.get("avg_loss")
                    tokens = meta.get("cumulative_tokens")
                    commit_msg = f"Backup {target_to_upload} | Loss: {avg_loss} | Tokens: {tokens:,}"
                except Exception:
                    pass

                print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Staging '{target_to_upload}'...")
                if os.path.exists(staging_dir):
                    shutil.rmtree(staging_dir, ignore_errors=True)
                os.makedirs(staging_dir, exist_ok=True)
                for fname in os.listdir(full_path):
                    s_p = os.path.join(full_path, fname)
                    d_p = os.path.join(staging_dir, fname)
                    if os.path.isfile(s_p):
                        try:
                            os.link(s_p, d_p)
                        except OSError:
                            shutil.copy2(s_p, d_p)

                inject_remote_code_and_automap(staging_dir, project_root)

                print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Uploading '{target_to_upload}' to '{repo_id}'...")
                upload_start = time.time()
                try:
                    api.upload_folder(
                        folder_path=staging_dir,
                        repo_id=repo_id,
                        repo_type="model",
                        commit_message=commit_msg,
                    )
                    upload_duration = time.time() - upload_start
                    print(f"Successfully uploaded '{target_to_upload}' in {upload_duration:.1f}s!")

                    uploaded_chunks.add(target_to_upload)
                    c_num = get_chunk_number(target_to_upload)
                    if c_num and c_num > last_uploaded_idx:
                        last_uploaded_idx = c_num

                    with open(history_file, "w", encoding="utf-8") as f:
                        json.dump(list(uploaded_chunks), f, indent=2)

                except Exception as e:
                    print(f"Upload failed for '{target_to_upload}': {e}. Will retry in next cycle.")
                finally:
                    shutil.rmtree(staging_dir, ignore_errors=True)

            if once:
                break
            time.sleep(poll_sec)

    except KeyboardInterrupt:
        print("\n[Backup Daemon] Gracefully stopped by user.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Jerboa Pretrain Checkpoint HF Hub Backup Daemon")
    parser.add_argument("--steps_dir", type=str, default="checkpoints/pretrain/steps", help="Path to checkpoint steps directory")
    parser.add_argument("--repo_id", type=str, default="ztor2/jerboa-base", help="Hugging Face Hub repository ID")
    parser.add_argument("--interval_chunks", type=int, default=50, help="Upload every N chunks (default: 50 chunks)")
    parser.add_argument("--poll_sec", type=int, default=30, help="Directory polling interval in seconds")
    parser.add_argument("--public", action="store_true", default=True, help="Set repository to public")
    parser.add_argument("--private", action="store_false", dest="public", help="Set repository to private")
    parser.add_argument("--once", action="store_true", help="Run a single backup check and exit")
    args = parser.parse_args()

    run_backup_daemon(
        steps_dir=args.steps_dir,
        repo_id=args.repo_id,
        interval_chunks=args.interval_chunks,
        poll_sec=args.poll_sec,
        private=not args.public,
        once=args.once,
    )
