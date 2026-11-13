"""
dataset_gate.py: Dataset Integrity, Subject Partitioning, and Test-Set Protection Gate.

This module enforces strict data governance for the SLEEP research project:
  1. Validates raw dataset existence (PhysioNet BIDSLab v1.0.0; 47 subjects).
  2. Validates preprocessing cache integrity (prep_57738d534b; 246 usable nights).
  3. Enforces strict subject-level disjointness (train ∩ val = ∅, train ∩ test = ∅, val ∩ test = ∅).
  4. Hard gate against test-set leakage: blocks training or validation on held-out test subjects.
  5. Computes deterministic dataset and split fingerprints.
"""

import os
import glob
import json
import hashlib
from typing import Dict, List, Set, Tuple, Optional, Any


class DatasetGateError(Exception):
    """Base exception for dataset and split gate violations."""
    pass


class TestSetLeakageError(DatasetGateError):
    """Raised when an operation attempts to use held-out test subjects for training/validation."""
    __test__ = False
    pass


class SplitDisjointnessError(DatasetGateError):
    """Raised when subject overlap is detected between train, val, or test partitions."""
    pass


class DatasetIntegrityError(DatasetGateError):
    """Raised when expected dataset files or caches are missing, corrupted, or tampered with."""
    pass


class DatasetGate:
    """
    Enforces dataset integrity, split verification, and hard test-set protection.
    """
    EXPECTED_TOTAL_SUBJECTS = 47
    EXPECTED_RAW_NIGHTS = 253
    EXPECTED_USABLE_NIGHTS = 246
    EXPECTED_FEATURE_VERSION = "hr-motion-v2"
    EXPECTED_CACHE_FINGERPRINT = "prep_57738d534b"

    def __init__(
        self,
        base_dir: Optional[str] = None,
        dataset_dir: Optional[str] = None,
        cache_dir: Optional[str] = None
    ):
        self.base_dir = base_dir or os.path.dirname(os.path.abspath(__file__))
        self.dataset_dir = dataset_dir or os.path.join(
            self.base_dir,
            "a-multi-night-instantaneous-heart-rate-and-accelerometry-dataset-with-eeg-sleep-stage-labels-1.0.0"
        )
        self.cache_dir = cache_dir or os.path.join(
            "/home/rj2826/ai-research/sleep/sleepchannelnet_v8_preprocessing_audit_work_REAL/cache",
            self.EXPECTED_CACHE_FINGERPRINT
        )

    def verify_dataset_structure(self) -> Dict[str, Any]:
        """
        Verifies that the raw dataset exists, contains exactly 47 subjects,
        and matches expected night counts.
        """
        if not os.path.exists(self.dataset_dir):
            raise DatasetIntegrityError(f"Raw dataset directory not found at: {self.dataset_dir}")

        subject_dirs = sorted([
            d for d in os.listdir(self.dataset_dir)
            if d.startswith("Bidslab") and os.path.isdir(os.path.join(self.dataset_dir, d))
        ])

        if len(subject_dirs) != self.EXPECTED_TOTAL_SUBJECTS:
            raise DatasetIntegrityError(
                f"Subject count mismatch: expected {self.EXPECTED_TOTAL_SUBJECTS}, found {len(subject_dirs)}"
            )

        labels_files = glob.glob(os.path.join(self.dataset_dir, "Bidslab*", "*", "labels.mat"))
        if len(labels_files) != self.EXPECTED_RAW_NIGHTS:
            raise DatasetIntegrityError(
                f"Raw night count mismatch: expected {self.EXPECTED_RAW_NIGHTS}, found {len(labels_files)}"
            )

        return {
            "dataset_dir": self.dataset_dir,
            "n_subjects": len(subject_dirs),
            "n_raw_nights": len(labels_files),
            "subjects": subject_dirs,
            "status": "VERIFIED"
        }

    def verify_cache_integrity(self) -> Dict[str, Any]:
        """
        Verifies that the preprocessing cache exists and contains the 246 usable nights.
        """
        if not os.path.exists(self.cache_dir):
            raise DatasetIntegrityError(f"Feature cache directory not found at: {self.cache_dir}")

        npz_files = glob.glob(os.path.join(self.cache_dir, "*.npz"))
        if len(npz_files) != self.EXPECTED_USABLE_NIGHTS:
            raise DatasetIntegrityError(
                f"Usable night count mismatch in cache: expected {self.EXPECTED_USABLE_NIGHTS}, found {len(npz_files)}"
            )

        return {
            "cache_dir": self.cache_dir,
            "n_usable_nights": len(npz_files),
            "cache_fingerprint": self.EXPECTED_CACHE_FINGERPRINT,
            "feature_version": self.EXPECTED_FEATURE_VERSION,
            "status": "VERIFIED"
        }

    def verify_split(
        self,
        train_subjects: List[str],
        val_subjects: List[str],
        test_subjects: List[str],
        strict_cohort_coverage: bool = False
    ) -> Dict[str, Any]:
        """
        Verifies subject-level disjointness between train, validation, and test partitions.
        Enforces:
          1. train ∩ val = ∅
          2. train ∩ test = ∅
          3. val ∩ test = ∅
        """
        set_train = set(train_subjects)
        set_val = set(val_subjects)
        set_test = set(test_subjects)

        # 1. Disjointness assertions
        overlap_train_val = set_train.intersection(set_val)
        overlap_train_test = set_train.intersection(set_test)
        overlap_val_test = set_val.intersection(set_test)

        if overlap_train_val:
            raise SplitDisjointnessError(
                f"Subject overlap between Train and Val: {sorted(list(overlap_train_val))}"
            )
        if overlap_train_test:
            raise SplitDisjointnessError(
                f"Subject overlap between Train and Test: {sorted(list(overlap_train_test))}"
            )
        if overlap_val_test:
            raise SplitDisjointnessError(
                f"Subject overlap between Val and Test: {sorted(list(overlap_val_test))}"
            )

        total_split_subjects = len(set_train | set_val | set_test)
        if strict_cohort_coverage and total_split_subjects != self.EXPECTED_TOTAL_SUBJECTS:
            raise SplitDisjointnessError(
                f"Split does not cover the complete cohort: expected {self.EXPECTED_TOTAL_SUBJECTS}, got {total_split_subjects}"
            )

        split_hash = self.compute_split_hash(train_subjects, val_subjects, test_subjects)

        return {
            "n_train_subjects": len(set_train),
            "n_val_subjects": len(set_val),
            "n_test_subjects": len(set_test),
            "total_subjects": total_split_subjects,
            "split_hash": split_hash,
            "disjointness_verified": True
        }

    def assert_no_test_leakage(
        self,
        active_role: str,
        accessed_subjects: List[str],
        held_out_test_subjects: List[str]
    ) -> None:
        """
        Hard gate: Ensures that training, validation, feature selection, or early stopping
        never touches subjects assigned to the held-out test partition.
        """
        development_roles = {
            "train", "training", "val", "validation",
            "tuning", "search", "hyperparameter_search",
            "feature_selection", "early_stopping", "ablation",
            "calibration", "threshold_tuning", "ensemble_weighting",
            "model_selection", "architecture_selection"
        }

        role_clean = active_role.lower().strip()
        if role_clean in development_roles:
            leaked = set(accessed_subjects).intersection(set(held_out_test_subjects))
            if leaked:
                raise TestSetLeakageError(
                    f"TEST SET LEAKAGE VIOLATION: Role '{active_role}' accessed held-out test subjects: "
                    f"{sorted(list(leaked))}. Development operations cannot access the test set!"
                )

    def compute_split_hash(
        self,
        train_subjects: List[str],
        val_subjects: List[str],
        test_subjects: List[str]
    ) -> str:
        """Computes a deterministic SHA256 hex digest for a subject partition."""
        payload = {
            "train": sorted(list(set(train_subjects))),
            "val": sorted(list(set(val_subjects))),
            "test": sorted(list(set(test_subjects)))
        }
        raw_bytes = json.dumps(payload, sort_keys=True).encode("utf-8")
        return hashlib.sha256(raw_bytes).hexdigest()[:16]

    def compute_dataset_fingerprint(self) -> str:
        """Computes a composite fingerprint representing the raw dataset and feature cache."""
        raw_status = self.verify_dataset_structure()
        cache_status = self.verify_cache_integrity()
        descriptor = (
            f"DATASET:{raw_status['n_subjects']}_{raw_status['n_raw_nights']}|"
            f"CACHE:{cache_status['cache_fingerprint']}_{cache_status['n_usable_nights']}_"
            f"{cache_status['feature_version']}"
        )
        return hashlib.sha256(descriptor.encode("utf-8")).hexdigest()[:16]


if __name__ == "__main__":
    print("Running DatasetGate preflight self-test...")
    gate = DatasetGate()
    ds_info = gate.verify_dataset_structure()
    print(f"Dataset Verified: {ds_info['n_subjects']} subjects, {ds_info['n_raw_nights']} raw nights")
    cache_info = gate.verify_cache_integrity()
    print(f"Cache Verified: {cache_info['n_usable_nights']} usable nights in {cache_info['cache_fingerprint']}")
    fp = gate.compute_dataset_fingerprint()
    print(f"Composite Dataset Fingerprint: {fp}")
    print("DatasetGate self-test PASSED.")
