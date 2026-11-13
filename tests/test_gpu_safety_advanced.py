"""
tests/test_gpu_safety_advanced.py: Advanced unit tests for dual RTX 3060 GPU resource safety.

Tests:
  1. Normal allocation
  2. Normal release
  3. Double allocation (resource exhaustion)
  4. Invalid GPU specifications & out-of-bounds indices
  5. VRAM overflow beyond safe ceiling (10,240 MB)
  6. Concurrent allocation exclusivity
  7. Simulated failure release via context manager
  8. Simulated exception release via context manager
  9. Stale allocation recovery
"""

import time
import pytest
from research_engine.resource_manager import GPUResourceManager, ResourceAllocationError


@pytest.fixture
def resource_manager():
    # Instantiate with simulated 2 GPUs to strictly test dual-GPU logic irrespective of host CUDA state
    return GPUResourceManager(simulated_device_count=2)


# 1. Normal allocation
def test_normal_allocation(resource_manager):
    dev = resource_manager.allocate_gpu("EXP-001", "SHORT")
    assert dev == "cuda:0"
    stat = resource_manager.get_device_status()
    assert stat[0]["active_lease"] == "EXP-001"
    assert stat[0]["is_available"] is False
    assert stat[1]["is_available"] is True


# 2. Normal release
def test_normal_release(resource_manager):
    dev = resource_manager.allocate_gpu("EXP-001", "SHORT")
    resource_manager.release_gpu(dev)
    stat = resource_manager.get_device_status()
    assert stat[0]["active_lease"] is None
    assert stat[0]["is_available"] is True


# 3. Double allocation (saturation)
def test_double_allocation_saturation(resource_manager):
    dev1 = resource_manager.allocate_gpu("EXP-001", "SHORT")
    dev2 = resource_manager.allocate_gpu("EXP-002", "STANDARD")
    assert dev1 == "cuda:0"
    assert dev2 == "cuda:1"

    # Third allocation must fail
    with pytest.raises(ResourceAllocationError) as exc_info:
        resource_manager.allocate_gpu("EXP-003", "SMOKE")
    assert "All 2 GPUs are currently busy" in str(exc_info.value)


# 4. Invalid GPU specifications
def test_invalid_gpu_specifications(resource_manager):
    with pytest.raises(ResourceAllocationError) as exc_info1:
        resource_manager.validate_device("cuda:99")
    assert "out of range" in str(exc_info1.value)

    with pytest.raises(ResourceAllocationError) as exc_info2:
        resource_manager.validate_device("cuda:-1")
    assert "out of range" in str(exc_info2.value)

    with pytest.raises(ResourceAllocationError) as exc_info3:
        resource_manager.validate_device("invalid_device_name")
    assert "Invalid device format" in str(exc_info3.value)


# 5. VRAM overflow beyond safe ceiling
def test_vram_overflow(resource_manager):
    # Safe ceiling is 10,240 MB
    with pytest.raises(ResourceAllocationError) as exc_info:
        resource_manager.allocate_gpu("EXP-OVERFLOW", "FULL", requested_vram_mb=11000)
    assert "exceeds safe ceiling" in str(exc_info.value)


# 6. Concurrent allocation exclusivity
def test_concurrent_allocation_exclusivity(resource_manager):
    dev1 = resource_manager.allocate_gpu("JOB_A", "SHORT")
    stat = resource_manager.get_device_status()
    assert stat[0]["active_lease"] == "JOB_A"
    
    # Second job gets the other GPU, never the same GPU
    dev2 = resource_manager.allocate_gpu("JOB_B", "SHORT")
    assert dev2 != dev1
    assert dev2 == "cuda:1"


# 7. Simulated failure release via context manager
def test_simulated_failure_release(resource_manager):
    class SimulatedTrainingFailure(Exception):
        pass

    try:
        with resource_manager.lease("EXP-FAIL", "SHORT") as dev:
            assert dev == "cuda:0"
            assert resource_manager.active_leases[0] == "EXP-FAIL"
            raise SimulatedTrainingFailure("OOM or loss divergence")
    except SimulatedTrainingFailure:
        pass

    # Ensure device was cleaned up and released despite exception
    assert resource_manager.active_leases[0] is None
    assert resource_manager.allocated_vram_mb[0] == 0


# 8. Simulated exception release via context manager
def test_simulated_exception_release(resource_manager):
    try:
        with resource_manager.lease("EXP-EXCEPTION", "STANDARD") as dev:
            assert dev == "cuda:0"
            raise KeyboardInterrupt("Simulated user interrupt")
    except KeyboardInterrupt:
        pass

    # Ensure device was cleaned up and released
    assert resource_manager.active_leases[0] is None


# 9. Stale allocation recovery
def test_stale_allocation_recovery(resource_manager):
    dev = resource_manager.allocate_gpu("EXP-STALE", "SHORT")
    assert resource_manager.active_leases[0] == "EXP-STALE"

    # Simulate lease age of 4000 seconds (older than 3600 max age)
    resource_manager.lease_timestamps[0] = time.time() - 4000

    recovered = resource_manager.recover_stale_leases(max_age_seconds=3600)
    assert len(recovered) == 1
    assert "cuda:0" in recovered[0]
    assert resource_manager.active_leases[0] is None
