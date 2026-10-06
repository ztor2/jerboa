import json
import os
import random
import shutil
import signal
import subprocess
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import torch


class SleepGuard:
    """Prevents macOS from entering sleep mode while training is active."""

    def __init__(self, reason: str = "Jerboa Training in progress"):
        self.reason = reason
        self.process: Optional[subprocess.Popen] = None

    def start(self):
        if sys.platform == "darwin" and self.process is None:
            try:
                # -d: prevent display sleep
                # -i: prevent system idle sleep
                # -s: prevent system sleep on AC power
                # -w <pid>: wait for this process PID to exit
                self.process = subprocess.Popen(
                    ["caffeinate", "-d", "-i", "-s", "-w", str(os.getpid())],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
            except Exception as e:
                print(f"[SleepGuard] Warning: could not launch caffeinate: {e}")

    def stop(self):
        if self.process is not None:
            try:
                self.process.terminate()
                self.process.wait(timeout=1.0)
            except Exception:
                pass
            self.process = None

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()


class GracefulInterruptHandler:
    """Catches SIGINT (Ctrl+C) and SIGTERM to allow saving checkpoints before exiting."""

    def __init__(self):
        self._interrupted = False
        self._original_sigint = signal.getsignal(signal.SIGINT)
        self._original_sigterm = signal.getsignal(signal.SIGTERM)

        signal.signal(signal.SIGINT, self._handle_signal)
        signal.signal(signal.SIGTERM, self._handle_signal)

    def _handle_signal(self, signum, frame):
        if not self._interrupted:
            self._interrupted = True
            sig_name = "SIGINT (Ctrl+C)" if signum == signal.SIGINT else "SIGTERM"
            print(f"\n[Interrupt] Caught {sig_name}! Saving state before exit... (Press again to force quit)")
            # Revert to original handler so a second press aborts immediately
            signal.signal(signal.SIGINT, self._original_sigint)
            signal.signal(signal.SIGTERM, self._original_sigterm)
        else:
            if callable(self._original_sigint):
                self._original_sigint(signum, frame)
            sys.exit(130)

    @property
    def interrupted(self) -> bool:
        return self._interrupted

    def reset(self):
        self._interrupted = False


class SystemResourceGuard:
    """Protects macOS system stability and prevents OOM/swapping during background training."""

    def __init__(
        self,
        max_mps_fraction: float = 0.25,
        system_ram_threshold: float = 85.0,
        nice_priority: int = 10,
    ):
        self.max_mps_fraction = max_mps_fraction
        self.system_ram_threshold = system_ram_threshold
        self.nice_priority = nice_priority
        self._initialized = False

    def setup(self):
        """Configure MPS memory ceilings and process priority."""
        if self._initialized:
            return

        # 1. Lower process priority so foreground UI apps (Chrome, VSCode, Slack) stay smooth
        if hasattr(os, "nice") and sys.platform == "darwin":
            try:
                os.nice(self.nice_priority)
            except Exception:
                pass

        # 2. Limit MPS process memory fraction if on Apple Silicon
        if torch.backends.mps.is_available() and hasattr(torch.mps, "set_per_process_memory_fraction"):
            try:
                torch.mps.set_per_process_memory_fraction(self.max_mps_fraction)
            except Exception:
                pass

        self._initialized = True

    def check_and_throttle(self) -> bool:
        """Checks current system memory pressure and throttles if memory is constrained."""
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()

        try:
            import psutil
            mem = psutil.virtual_memory()
            if mem.percent >= self.system_ram_threshold:
                print(f"\n[ResourceGuard] High system memory pressure: {mem.percent:.1f}% >= {self.system_ram_threshold}%.")
                print("[ResourceGuard] Pausing 5s and freeing MPS cache to protect foreground apps...")
                import gc
                gc.collect()
                if torch.backends.mps.is_available():
                    torch.mps.empty_cache()
                time.sleep(5.0)
                return True
        except ImportError:
            pass
        return False


class CheckpointManager:
    """Manages saving, rotating, and restoring training checkpoints and states."""

    def __init__(self, output_dir: str, max_to_keep: int = 3):
        self.output_dir = output_dir
        self.max_to_keep = max_to_keep
        self.meta_file = os.path.join(output_dir, "checkpoint_meta.json")
        os.makedirs(output_dir, exist_ok=True)

    def save(
        self,
        model: torch.nn.Module,
        tokenizer: Any = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[Any] = None,
        step: int = 0,
        epoch: int = 0,
        step_in_epoch: int = 0,
        metrics: Optional[Dict[str, Any]] = None,
        tag: Optional[str] = None,
        is_interrupted: bool = False,
        custom_state: Optional[Dict[str, Any]] = None,
    ) -> str:
        tag_str = tag if tag else f"step_{step:06d}"
        if is_interrupted:
            tag_str = f"interrupted_{tag_str}"

        ckpt_dir = os.path.join(self.output_dir, tag_str)
        os.makedirs(ckpt_dir, exist_ok=True)

        # 1. Save model weights and tokenizer
        if hasattr(model, "save_pretrained"):
            model.save_pretrained(ckpt_dir)
        else:
            torch.save(model.state_dict(), os.path.join(ckpt_dir, "model.pt"))

        if tokenizer is not None and hasattr(tokenizer, "save_pretrained"):
            tokenizer.save_pretrained(ckpt_dir)

        # 2. Save training state (optimizer, scheduler, RNGs)
        state_dict: Dict[str, Any] = {
            "step": step,
            "epoch": epoch,
            "step_in_epoch": step_in_epoch,
            "metrics": metrics or {},
            "custom_state": custom_state or {},
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "rng_state": {
                "python": random.getstate(),
                "torch": torch.get_rng_state(),
            },
        }

        if torch.cuda.is_available():
            state_dict["rng_state"]["cuda"] = torch.cuda.get_rng_state()

        if optimizer is not None:
            state_dict["optimizer_state"] = optimizer.state_dict()
        if scheduler is not None and hasattr(scheduler, "state_dict"):
            state_dict["scheduler_state"] = scheduler.state_dict()

        state_pt_path = os.path.join(ckpt_dir, "training_state.pt")
        torch.save(state_dict, state_pt_path)

        # 3. Save readable JSON summary
        readable_summary = {
            "checkpoint_dir": ckpt_dir,
            "step": step,
            "epoch": epoch,
            "step_in_epoch": step_in_epoch,
            "metrics": metrics or {},
            "custom_state": custom_state or {},
            "is_interrupted": is_interrupted,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        with open(os.path.join(ckpt_dir, "training_state.json"), "w", encoding="utf-8") as f:
            json.dump(readable_summary, f, indent=2)

        # 4. Update latest metadata pointer
        self._update_meta(ckpt_dir, readable_summary)

        # 5. Rotate old step checkpoints if max_to_keep > 0
        if not is_interrupted and self.max_to_keep > 0:
            self._rotate_checkpoints()

        return ckpt_dir

    def _update_meta(self, latest_dir: str, summary: Dict[str, Any]):
        meta = {
            "latest_checkpoint": latest_dir,
            "step": summary.get("step", 0),
            "epoch": summary.get("epoch", 0),
            "timestamp": summary.get("timestamp", ""),
            "is_interrupted": summary.get("is_interrupted", False),
        }
        with open(self.meta_file, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

    def _rotate_checkpoints(self):
        """Keep only the most recent `max_to_keep` step checkpoints."""
        if not os.path.exists(self.output_dir):
            return

        candidates = []
        for name in os.listdir(self.output_dir):
            if name.startswith("step_") or name.startswith("interrupted_step_"):
                full_path = os.path.join(self.output_dir, name)
                if os.path.isdir(full_path):
                    candidates.append((os.path.getmtime(full_path), full_path))

        # Sort by modification time ascending (oldest first)
        candidates.sort(key=lambda x: x[0])
        while len(candidates) > self.max_to_keep:
            _, old_dir = candidates.pop(0)
            try:
                shutil.rmtree(old_dir)
            except Exception as e:
                print(f"[CheckpointManager] Notice: failed to remove old checkpoint {old_dir}: {e}")

    def find_latest_checkpoint(self) -> Optional[str]:
        """Find the latest valid checkpoint directory."""
        if os.path.exists(self.meta_file):
            try:
                with open(self.meta_file, "r", encoding="utf-8") as f:
                    meta = json.load(f)
                latest = meta.get("latest_checkpoint")
                if latest and os.path.isdir(latest):
                    return latest
            except Exception:
                pass

        # Fallback: scan directories
        if not os.path.exists(self.output_dir):
            return None

        dirs = []
        for item in os.listdir(self.output_dir):
            p = os.path.join(self.output_dir, item)
            if os.path.isdir(p) and (item.startswith("step_") or item.startswith("interrupted_") or item.startswith("chunk_") or item.startswith("epoch_")):
                dirs.append((os.path.getmtime(p), p))

        if dirs:
            dirs.sort(key=lambda x: x[0], reverse=True)
            return dirs[0][1]

        # Check if output_dir/model exists
        model_sub = os.path.join(self.output_dir, "model")
        if os.path.isdir(model_sub):
            return model_sub

        return None

    def restore_state(
        self,
        checkpoint_dir: str,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """Restores optimizer, scheduler, and RNG state from training_state.pt."""
        state_file = os.path.join(checkpoint_dir, "training_state.pt")
        summary_file = os.path.join(checkpoint_dir, "training_state.json")

        if not os.path.exists(state_file):
            # Fallback to json summary if pt doesn't exist
            if os.path.exists(summary_file):
                with open(summary_file, "r", encoding="utf-8") as f:
                    return json.load(f)
            return {}

        state = torch.load(state_file, map_location="cpu")

        if optimizer is not None and "optimizer_state" in state:
            try:
                optimizer.load_state_dict(state["optimizer_state"])
            except Exception as e:
                print(f"[CheckpointManager] Warning: could not restore optimizer state: {e}")

        if scheduler is not None and "scheduler_state" in state and hasattr(scheduler, "load_state_dict"):
            try:
                scheduler.load_state_dict(state["scheduler_state"])
            except Exception as e:
                print(f"[CheckpointManager] Warning: could not restore scheduler state: {e}")

        # Restore RNG states
        if "rng_state" in state:
            rng = state["rng_state"]
            if "python" in rng:
                random.setstate(rng["python"])
            if "torch" in rng:
                torch.set_rng_state(rng["torch"])
            if torch.cuda.is_available() and "cuda" in rng:
                torch.cuda.set_rng_state(rng["cuda"])

        return {
            "step": state.get("step", 0),
            "epoch": state.get("epoch", 0),
            "step_in_epoch": state.get("step_in_epoch", 0),
            "metrics": state.get("metrics", {}),
            "custom_state": state.get("custom_state", {}),
            "timestamp": state.get("timestamp", ""),
        }
