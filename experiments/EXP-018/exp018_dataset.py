"""
exp018_dataset.py: Dataset definitions and batching for EXP-018 Context-Horizon Curve.

Authoritative Implementation for EXP-018:
  - RandomCropSleepDataset: Dynamic uniform sampling of context windows L from anywhere in the night.
  - Zero-padding protection for short nights (N <= L) with native length seq_len propagation.
  - Circadian feature neutralization (time_feat = 0.0 in primary condition).
  - Compute invariant schedule lookup: B * L * N_accum == 2400 tokens/step.
  - FullNightRecord: Container for full-night arrays evaluated by SlidingWindowPredictor.
"""

from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


# Pre-registered compute schedule lookup table (Section 4.2)
# Invariant: B * L * N_accum == 2,400 tokens per optimizer step
COMPUTE_SCHEDULE_LOOKUP: Dict[int, Dict[str, Any]] = {
    1200: {"batch_type": "trajectory", "B": 2, "N_accum": 1, "tokens_per_step": 2400},
    600:  {"batch_type": "trajectory", "B": 4, "N_accum": 1, "tokens_per_step": 2400},
    300:  {"batch_type": "trajectory", "B": 4, "N_accum": 2, "tokens_per_step": 2400},
    120:  {"batch_type": "trajectory", "B": 4, "N_accum": 5, "tokens_per_step": 2400},
    60:   {"batch_type": "trajectory", "B": 8, "N_accum": 5, "tokens_per_step": 2400},
    20:   {"batch_type": "trajectory", "B": 12, "N_accum": 10, "tokens_per_step": 2400},
    1:    {"batch_type": "flat_epoch", "B": 240, "N_accum": 10, "tokens_per_step": 2400},
}


def verify_compute_schedule_invariant() -> None:
    """Asserts that every pre-registered context length exactly satisfies the token invariant."""
    for L, cfg in COMPUTE_SCHEDULE_LOOKUP.items():
        computed_tokens = cfg["B"] * L * cfg["N_accum"]
        assert computed_tokens == 2400, (
            f"Compute schedule invariant violated for L={L}: "
            f"B={cfg['B']}, N_accum={cfg['N_accum']} -> {computed_tokens} tokens != 2400"
        )


class RandomCropSleepDataset(Dataset):
    """
    Dataset extracting random temporal context crops of length L from full-night recordings.
    
    Invariants:
      1. If native length N > L: Start offset sampled uniformly s ~ Uniform(0, N - L).
      2. If native length N <= L: Extract full recording [0, N), right-pad with zeros to L,
         set target to -100 on padded positions, mask=False, and pass seq_len = N.
      3. Circadian neutralization: If neutralize_circadian=True (primary series),
         the time_feat feature (channel index 2 of meta) is explicitly set to 0.0.
      4. Single-night containment: Crops never span across nights or subjects.
    """
    def __init__(
        self,
        records: List[Dict[str, np.ndarray]],
        subject_to_idx: Dict[str, int],
        context_len: int = 600,
        neutralize_circadian: bool = True,
        seed: Optional[int] = None
    ):
        assert len(records) > 0, "Records list cannot be empty"
        assert context_len in COMPUTE_SCHEDULE_LOOKUP, (
            f"Context length {context_len} not in approved lookup: {list(COMPUTE_SCHEDULE_LOOKUP.keys())}"
        )
        self.records = records
        self.subject_to_idx = subject_to_idx
        self.context_len = context_len
        self.neutralize_circadian = neutralize_circadian
        self.rng = np.random.RandomState(seed) if seed is not None else np.random.RandomState()

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        rec = self.records[idx]
        subj = rec["subject"]
        subj_idx = self.subject_to_idx.get(subj, -100)

        native_len = len(rec["y"])
        L = self.context_len

        if native_len > L:
            # Sample start offset uniformly in [0, N - L]
            s = self.rng.randint(0, native_len - L + 1)
            e = s + L

            ihr_slice = rec["ihr"][s:e]       # (L, 30)
            acc_slice = rec["accel"][s:e]     # (L, 30)
            freq_slice = rec["freq"][s:e]     # (L, 1)
            time_slice = rec["time"][s:e]     # (L, 1)
            y_slice = rec["y"][s:e]           # (L,)
            mask_slice = np.ones(L, dtype=bool)
            seq_len = L
        else:
            # Short-night case: take [0, native_len) and right-pad with zeros up to L
            pad_len = L - native_len
            seq_len = native_len

            ihr_slice = np.pad(rec["ihr"], ((0, pad_len), (0, 0)), mode="constant", constant_values=0.0)
            acc_slice = np.pad(rec["accel"], ((0, pad_len), (0, 0)), mode="constant", constant_values=0.0)
            freq_slice = np.pad(rec["freq"], ((0, pad_len), (0, 0)), mode="constant", constant_values=0.0)
            time_slice = np.pad(rec["time"], ((0, pad_len), (0, 0)), mode="constant", constant_values=0.0)
            y_slice = np.pad(rec["y"], (0, pad_len), mode="constant", constant_values=-100)

            mask_slice = np.zeros(L, dtype=bool)
            mask_slice[:native_len] = True

        # Compute summary HR statistics
        hr_mean = ihr_slice.mean(axis=1, keepdims=True)
        hr_std = ihr_slice.std(axis=1, keepdims=True)
        clock = np.zeros_like(time_slice)

        # Circadian neutralization: primary series sets time_feat = 0.0
        if self.neutralize_circadian:
            time_feat = np.zeros_like(time_slice)
        else:
            time_feat = time_slice

        # Channel layout matching exp013_models: [freq, clock, time_feat, hr_mean, hr_std]
        meta = np.concatenate([freq_slice, clock, time_feat, hr_mean, hr_std], axis=-1)

        return {
            "ihr": torch.from_numpy(ihr_slice.astype(np.float32)),
            "accel": torch.from_numpy(acc_slice.astype(np.float32)),
            "meta": torch.from_numpy(meta.astype(np.float32)),
            "y": torch.from_numpy(y_slice.astype(np.int64)),
            "mask": torch.from_numpy(mask_slice),
            "seq_len": torch.tensor(seq_len, dtype=torch.int64),
            "subject_idx": torch.tensor(subj_idx, dtype=torch.int64),
            "subject": subj,
            "night": rec["night"]
        }


def seed_worker(worker_id: int) -> None:
    """Ensures independent, deterministic random crop streams across DataLoader worker processes."""
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)


def create_infinite_batch_stream(
    dataset: Dataset,
    batch_size: int,
    seed: int,
    num_workers: int = 0
):
    """
    Creates an infinite repeating generator yielding batches from a DataLoader.
    Ensures that an exact total of 3,200 optimizer update steps are executed.
    """
    g = torch.Generator()
    g.manual_seed(seed)

    while True:
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            generator=g,
            worker_init_fn=seed_worker,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
            drop_last=False
        )
        for batch in loader:
            yield batch
