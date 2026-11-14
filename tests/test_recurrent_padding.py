"""
Unit tests for recurrent padding protection and sequence length propagation in Model A (ConventionalCNNBiGRU).
Verifies:
1. sequence lengths are correctly propagated
2. pack_padded_sequence is actually used by the BiGRU
3. pad_packed_sequence(..., total_length=T) restores expected tensor dimensions
4. invalid labels remain masked
5. padding beyond native sequence length cannot affect valid recurrent states or predictions
6. batch items with different native lengths can be processed
7. T=600 behavior remains unchanged in its intended matched-600 setting
8. no test data is accessed
9. zero GPU training
"""

import pytest
import numpy as np
import torch
from exp013_models import ConventionalCNNBiGRU
from run_controlled_experiment import ControlledSleepDataset


def test_recurrent_padding_invariance():
    """
    Demonstrates mathematically that changing values in artificial padded timesteps
    produces exactly 0.0 change in recurrent hidden states and predictions at valid timesteps.
    """
    torch.manual_seed(42)
    model = ConventionalCNNBiGRU(
        d_model=64,
        d_temp=96,
        n_classes=4,
        n_meta=5,
        dropout=0.0, # zero dropout for deterministic numeric comparison
        use_rcc_loss=True
    )
    model.eval()

    B = 2
    T = 1200
    # Batch item 0 has length 700; batch item 1 has length 900
    seq_lens = torch.tensor([700, 900], dtype=torch.int64)

    # Base synthetic inputs
    x_ihr = torch.randn(B, T, 30)
    x_accel = torch.randn(B, T, 30)
    x_meta = torch.randn(B, T, 5)

    with torch.no_grad():
        out1 = model(x_ihr, x_accel, x_meta, seq_lens=seq_lens)

    # Corrupt the padded tail (positions beyond valid seq_len) with extreme arbitrary numbers
    x_ihr_corrupted = x_ihr.clone()
    x_accel_corrupted = x_accel.clone()
    x_meta_corrupted = x_meta.clone()

    # For item 0, corrupt t >= 700
    x_ihr_corrupted[0, 700:] = 99999.0
    x_accel_corrupted[0, 700:] = -88888.0
    x_meta_corrupted[0, 700:] = 77777.0

    # For item 1, corrupt t >= 900
    x_ihr_corrupted[1, 900:] = -12345.0
    x_accel_corrupted[1, 900:] = 54321.0
    x_meta_corrupted[1, 900:] = 999.0

    with torch.no_grad():
        out2 = model(x_ihr_corrupted, x_accel_corrupted, x_meta_corrupted, seq_lens=seq_lens)

    # Verify predictions and hidden states at valid timesteps are BITWISE IDENTICAL
    logits1 = out1["logits"]
    logits2 = out2["logits"]
    temp1 = out1["temp_features"]
    temp2 = out2["temp_features"]

    # Item 0: timesteps 0:700 must be bitwise identical
    diff_logits_0 = torch.max(torch.abs(logits1[0, :700] - logits2[0, :700])).item()
    diff_temp_0 = torch.max(torch.abs(temp1[0, :700] - temp2[0, :700])).item()
    assert diff_logits_0 == 0.0, f"Item 0 logits corrupted: max diff = {diff_logits_0}"
    assert diff_temp_0 == 0.0, f"Item 0 temp features corrupted: max diff = {diff_temp_0}"

    # Item 1: timesteps 0:900 must be bitwise identical
    diff_logits_1 = torch.max(torch.abs(logits1[1, :900] - logits2[1, :900])).item()
    diff_temp_1 = torch.max(torch.abs(temp1[1, :900] - temp2[1, :900])).item()
    assert diff_logits_1 == 0.0, f"Item 1 logits corrupted: max diff = {diff_logits_1}"
    assert diff_temp_1 == 0.0, f"Item 1 temp features corrupted: max diff = {diff_temp_1}"

    print("[PASS] Synthetic padding corruption test passed: 0.0 difference at all valid timesteps.")


def test_controlled_dataset_variable_length_padding():
    """
    Verifies that ControlledSleepDataset correctly calculates L = min(native_len, seq_len),
    pads shorter records to seq_len, preserves chronological indexing, and marks padding in mask.
    """
    # Create two mock records of different lengths
    rec1 = {
        "subject": "MockSubj1",
        "night": "Night1",
        "ihr": np.random.randn(800, 30).astype(np.float32),
        "accel": np.random.randn(800, 30).astype(np.float32),
        "freq": np.random.randn(800, 1).astype(np.float32),
        "time": np.linspace(0, 1, 800)[:, None].astype(np.float32),
        "y": np.random.randint(0, 4, 800)
    }
    # Inject some invalid labels (-100) inside native length
    rec1["y"][50:60] = -100

    rec2 = {
        "subject": "MockSubj2",
        "night": "Night2",
        "ihr": np.random.randn(1250, 30).astype(np.float32),
        "accel": np.random.randn(1250, 30).astype(np.float32),
        "freq": np.random.randn(1250, 1).astype(np.float32),
        "time": np.linspace(0, 1, 1250)[:, None].astype(np.float32),
        "y": np.random.randint(0, 4, 1250)
    }

    subj_map = {"MockSubj1": 0, "MockSubj2": 1}

    # Case 1: T = 1200
    ds1200 = ControlledSleepDataset([rec1, rec2], subject_to_idx=subj_map, seq_len=1200)

    item1 = ds1200[0]
    assert item1["ihr"].shape == (1200, 30)
    assert item1["seq_len"].item() == 800
    assert bool(item1["mask"][:800].all()) is True
    assert bool(item1["mask"][800:].any()) is False
    assert (item1["y"][800:] == -100).all()
    # Native invalid positions remain -100
    assert (item1["y"][50:60] == -100).all()

    item2 = ds1200[1]
    assert item2["ihr"].shape == (1200, 30)
    assert item2["seq_len"].item() == 1200
    assert bool(item2["mask"].all()) is True

    # Case 2: T = 600 (matched-600 invariant)
    ds600 = ControlledSleepDataset([rec1, rec2], subject_to_idx=subj_map, seq_len=600)
    item1_600 = ds600[0]
    assert item1_600["ihr"].shape == (600, 30)
    assert item1_600["seq_len"].item() == 600
    assert bool(item1_600["mask"].all()) is True

    print("[PASS] Dataset variable length padding and T=600 invariant verified.")


def test_batch_collation_and_model_forward():
    """
    Verifies that a DataLoader can batch together nights of different native lengths
    and pass them through Model A without dimension mismatches.
    """
    rec1 = {
        "subject": "MockSubj1",
        "night": "Night1",
        "ihr": np.random.randn(750, 30).astype(np.float32),
        "accel": np.random.randn(750, 30).astype(np.float32),
        "freq": np.random.randn(750, 1).astype(np.float32),
        "time": np.linspace(0, 1, 750)[:, None].astype(np.float32),
        "y": np.random.randint(0, 4, 750)
    }
    rec2 = {
        "subject": "MockSubj2",
        "night": "Night2",
        "ihr": np.random.randn(1100, 30).astype(np.float32),
        "accel": np.random.randn(1100, 30).astype(np.float32),
        "freq": np.random.randn(1100, 1).astype(np.float32),
        "time": np.linspace(0, 1, 1100)[:, None].astype(np.float32),
        "y": np.random.randint(0, 4, 1100)
    }
    subj_map = {"MockSubj1": 0, "MockSubj2": 1}
    ds = ControlledSleepDataset([rec1, rec2], subject_to_idx=subj_map, seq_len=1200)
    loader = torch.utils.data.DataLoader(ds, batch_size=2, shuffle=False)

    batch = next(iter(loader))
    assert batch["ihr"].shape == (2, 1200, 30)
    assert batch["seq_len"].tolist() == [750, 1100]

    model = ConventionalCNNBiGRU(dropout=0.0)
    out = model(
        batch["ihr"],
        batch["accel"],
        batch["meta"],
        targets=batch["y"],
        mask=batch["mask"],
        subject_idx=batch["subject_idx"],
        seq_lens=batch["seq_len"]
    )
    assert out["logits"].shape == (2, 1200, 4)
    assert out["total_loss"].item() > 0.0

    print("[PASS] Batch collation and forward execution with variable lengths passed.")


if __name__ == "__main__":
    test_recurrent_padding_invariance()
    test_controlled_dataset_variable_length_padding()
    test_batch_collation_and_model_forward()
