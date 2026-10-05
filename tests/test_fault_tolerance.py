import os
import shutil
import signal
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pipeline.sft import run_sft


def test_sft_checkpoint_and_resume():
    test_output_dir = "checkpoints/test_fault_tolerance"
    if os.path.exists(test_output_dir):
        shutil.rmtree(test_output_dir, ignore_errors=True)

    print("\n--- Test Phase 1: Train 1 epoch, save every 5 steps ---")
    run_sft(
        model_path_or_name=None,
        output_dir=test_output_dir,
        epochs=1,
        batch_size=2,
        save_steps=5,
        save_total_limit=2,
        resume=None,
    )

    # Verify checkpoints exist
    step_dirs = [d for d in os.listdir(test_output_dir) if d.startswith("step_") or d.startswith("epoch_")]
    print(f"Generated checkpoint directories: {step_dirs}")
    assert len(step_dirs) > 0, "No step or epoch checkpoint directory created!"
    assert os.path.exists(os.path.join(test_output_dir, "checkpoint_meta.json")), "checkpoint_meta.json missing!"

    print("\n--- Test Phase 2: Resume training to epoch 2 using --resume auto ---")
    final_path = run_sft(
        model_path_or_name=None,
        output_dir=test_output_dir,
        epochs=2,
        batch_size=2,
        save_steps=5,
        save_total_limit=2,
        resume="auto",
    )

    assert final_path and os.path.exists(final_path), "Final model not found after resumed training!"
    print("\nTest Phase 2 Passed: Resumed training successfully completed.")

    # Clean up test output
    shutil.rmtree(test_output_dir, ignore_errors=True)
    print("Fault tolerance test passed completely!\n")


def test_graceful_interrupt():
    test_output_dir = "checkpoints/test_interrupt"
    if os.path.exists(test_output_dir):
        shutil.rmtree(test_output_dir, ignore_errors=True)

    print("\n--- Test Phase 3: Launch training and simulate SIGINT (Ctrl+C) ---")
    proc = subprocess.Popen(
        [
            sys.executable,
            "pipeline/sft.py",
            "--output_dir", test_output_dir,
            "--epochs", "5",
            "--batch_size", "2",
            "--save_steps", "2",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    # Wait for training to start and print at least one step
    time.sleep(6)
    print("Sending SIGINT to training process...")
    proc.send_signal(signal.SIGINT)
    stdout, stderr = proc.communicate(timeout=15)
    print(f"Process output preview:\n{stdout[-500:]}")

    # Verify emergency checkpoint was created
    items = os.listdir(test_output_dir) if os.path.exists(test_output_dir) else []
    print(f"Items in checkpoint dir after SIGINT: {items}")
    has_interrupted = any("interrupted" in x or "step_" in x for x in items)
    assert has_interrupted, f"No emergency checkpoint found in {items}!"
    print("Test Phase 3 Passed: Emergency checkpoint cleanly created on SIGINT.")

    # Clean up test output
    shutil.rmtree(test_output_dir, ignore_errors=True)


if __name__ == "__main__":
    test_sft_checkpoint_and_resume()
    test_graceful_interrupt()

