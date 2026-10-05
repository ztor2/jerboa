import os
import shutil
import sys
import tempfile
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pipeline.checkpoint_manager import CheckpointManager, SleepGuard, GracefulInterruptHandler


class DummyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(4, 2)

    def forward(self, x):
        return self.fc(x)


def test_checkpoint_save_and_restore():
    tmp_dir = tempfile.mkdtemp()
    try:
        model = DummyModel()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10)

        # Do a dummy forward/backward to populate optimizer momentum
        x = torch.randn(2, 4)
        loss = model(x).sum()
        loss.backward()
        optimizer.step()
        scheduler.step()

        mgr = CheckpointManager(output_dir=tmp_dir, max_to_keep=2)

        # Save checkpoint 1
        ckpt1 = mgr.save(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            step=10,
            epoch=1,
            step_in_epoch=5,
            metrics={"loss": 0.5},
        )
        assert os.path.exists(ckpt1)
        assert mgr.find_latest_checkpoint() == ckpt1

        # Save checkpoint 2
        ckpt2 = mgr.save(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            step=20,
            epoch=1,
            step_in_epoch=15,
            metrics={"loss": 0.3},
        )
        assert mgr.find_latest_checkpoint() == ckpt2

        # Save checkpoint 3 (should rotate ckpt1 away)
        ckpt3 = mgr.save(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            step=30,
            epoch=2,
            step_in_epoch=5,
            metrics={"loss": 0.2},
        )
        assert mgr.find_latest_checkpoint() == ckpt3
        assert not os.path.exists(ckpt1)  # Rotated out
        assert os.path.exists(ckpt2)
        assert os.path.exists(ckpt3)

        # Test restore
        new_model = DummyModel()
        new_opt = torch.optim.AdamW(new_model.parameters(), lr=1e-3)
        new_sched = torch.optim.lr_scheduler.StepLR(new_opt, step_size=10)

        state = mgr.restore_state(ckpt3, optimizer=new_opt, scheduler=new_sched)
        assert state["step"] == 30
        assert state["epoch"] == 2
        assert state["step_in_epoch"] == 5
        assert state["metrics"]["loss"] == 0.2
        print("CheckpointManager tests passed successfully!")
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


if __name__ == "__main__":
    test_checkpoint_save_and_restore()
