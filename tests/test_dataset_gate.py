"""
tests/test_dataset_gate.py: Unit tests for DatasetGate, Split Protection, and Leakage Prevention.
"""

import pytest
from dataset_gate import (
    DatasetGate,
    DatasetGateError,
    SplitDisjointnessError,
    TestSetLeakageError,
    DatasetIntegrityError
)


def test_dataset_and_cache_integrity():
    gate = DatasetGate()
    ds = gate.verify_dataset_structure()
    assert ds["n_subjects"] == 47
    assert ds["n_raw_nights"] == 253

    cache = gate.verify_cache_integrity()
    assert cache["n_usable_nights"] == 246
    assert cache["cache_fingerprint"] == "prep_57738d534b"

    fp = gate.compute_dataset_fingerprint()
    assert isinstance(fp, str) and len(fp) == 16


def test_split_disjointness_pass():
    gate = DatasetGate()
    train = ["Bidslab00", "Bidslab01", "Bidslab02"]
    val = ["Bidslab06", "Bidslab07"]
    test = ["Bidslab08", "Bidslab09"]

    res = gate.verify_split(train, val, test)
    assert res["disjointness_verified"] is True
    assert res["n_train_subjects"] == 3
    assert res["n_val_subjects"] == 2
    assert res["n_test_subjects"] == 2


def test_split_disjointness_overlap_train_val():
    gate = DatasetGate()
    train = ["Bidslab00", "Bidslab01", "Bidslab02"]
    val = ["Bidslab02", "Bidslab06"]  # overlap on Bidslab02
    test = ["Bidslab08"]

    with pytest.raises(SplitDisjointnessError) as exc_info:
        gate.verify_split(train, val, test)
    assert "Bidslab02" in str(exc_info.value)


def test_split_disjointness_overlap_train_test():
    gate = DatasetGate()
    train = ["Bidslab00", "Bidslab01"]
    val = ["Bidslab02"]
    test = ["Bidslab01", "Bidslab08"]  # overlap on Bidslab01

    with pytest.raises(SplitDisjointnessError) as exc_info:
        gate.verify_split(train, val, test)
    assert "Bidslab01" in str(exc_info.value)


def test_assert_no_test_leakage_blocked():
    gate = DatasetGate()
    held_out_test = ["Bidslab10", "Bidslab11", "Bidslab12"]
    accessed_in_training = ["Bidslab00", "Bidslab01", "Bidslab10"]  # contains Bidslab10

    with pytest.raises(TestSetLeakageError) as exc_info:
        gate.assert_no_test_leakage("training", accessed_in_training, held_out_test)
    assert "Bidslab10" in str(exc_info.value)


def test_assert_no_test_leakage_clean():
    gate = DatasetGate()
    held_out_test = ["Bidslab10", "Bidslab11", "Bidslab12"]
    accessed_in_val = ["Bidslab00", "Bidslab01", "Bidslab02"]

    # Should pass without raising
    gate.assert_no_test_leakage("validation", accessed_in_val, held_out_test)
