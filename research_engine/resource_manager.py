"""
research_engine/resource_manager.py: Dual-GPU Resource Manager for Local RTX 3060 Devices.

Hardware Configuration:
  - GPU 0: NVIDIA GeForce RTX 3060 12 GB
  - GPU 1: NVIDIA GeForce RTX 3060 12 GB
  - RAM: 32 GB
  - CPU: AMD Ryzen 7 7700 (8-core / 16-thread)

Guarantees:
  1. No VRAM oversubscription (hard safe ceiling: 10,240 MB).
  2. Single-job exclusivity per device for training runs.
  3. Context manager pattern guaranteeing release on success, failure, or exception.
  4. Stale lock recovery mechanism to prevent permanent GPU lockouts.
  5. Strict validation of device strings and bounds.
"""

import time
from contextlib import contextmanager
from typing import Dict, List, Optional, Any
import torch


class ResourceAllocationError(Exception):
    """Raised when GPU or compute resources cannot be safely allocated."""
    pass


class GPUResourceManager:
    """
    Manages allocation, VRAM safety limits, execution locks, and cleanup across GPUs.
    """
    TOTAL_VRAM_MB = 12288
    SAFE_ALLOCATABLE_VRAM_MB = 10240  # 10 GB limit leaving 2 GB system/CUDA overhead

    BUDGET_VRAM_ESTIMATES = {
        "SMOKE": 1500,     # ~1.5 GB
        "SHORT": 3000,     # ~3.0 GB
        "STANDARD": 4500,  # ~4.5 GB
        "FULL": 6000       # ~6.0 GB
    }

    def __init__(self, simulated_device_count: Optional[int] = None):
        if simulated_device_count is not None:
            self.device_count = simulated_device_count
        else:
            self.device_count = torch.cuda.device_count() if torch.cuda.is_available() else 0
        
        n_slots = max(1, self.device_count)
        self.active_leases: Dict[int, Optional[str]] = {i: None for i in range(n_slots)}
        self.allocated_vram_mb: Dict[int, int] = {i: 0 for i in range(n_slots)}
        self.lease_timestamps: Dict[int, Optional[float]] = {i: None for i in range(n_slots)}

    def get_device_status(self) -> List[Dict[str, Any]]:
        """Returns the real-time operational status of all managed devices."""
        status = []
        if self.device_count == 0:
            return [{
                "device_id": -1,
                "device_str": "cpu",
                "name": "CPU",
                "total_vram_mb": 0,
                "allocated_vram_mb": 0,
                "active_lease": self.active_leases.get(0),
                "is_available": self.active_leases.get(0) is None,
                "lease_age_seconds": (
                    time.time() - self.lease_timestamps[0]
                    if self.lease_timestamps.get(0) else None
                )
            }]

        for i in range(self.device_count):
            name = (
                torch.cuda.get_device_name(i)
                if torch.cuda.is_available() and i < torch.cuda.device_count()
                else f"Simulated GPU {i}"
            )
            allocated = self.allocated_vram_mb.get(i, 0)
            lease = self.active_leases.get(i)
            status.append({
                "device_id": i,
                "device_str": f"cuda:{i}",
                "name": name,
                "total_vram_mb": self.TOTAL_VRAM_MB,
                "allocated_vram_mb": allocated,
                "active_lease": lease,
                "is_available": (lease is None),
                "lease_age_seconds": (
                    time.time() - self.lease_timestamps[i]
                    if self.lease_timestamps.get(i) else None
                )
            })
        return status

    def validate_device(self, device_str: str) -> int:
        """Validates device string and returns the internal slot index."""
        if not device_str or not isinstance(device_str, str):
            raise ResourceAllocationError(f"Invalid device specification: '{device_str}'")

        if device_str == "cpu":
            if self.device_count > 0:
                pass
            return 0

        if not device_str.startswith("cuda:"):
            raise ResourceAllocationError(
                f"Invalid device format: '{device_str}'. Must be 'cpu' or 'cuda:<index>'."
            )

        try:
            idx = int(device_str.split(":")[-1])
        except ValueError:
            raise ResourceAllocationError(f"Malformed CUDA device index in '{device_str}'.")

        if idx < 0 or idx >= max(1, self.device_count):
            raise ResourceAllocationError(
                f"Device index {idx} out of range! Managed devices: 0 to {max(1, self.device_count) - 1}."
            )

        return idx

    def allocate_gpu(
        self,
        experiment_id: str,
        budget_name: str,
        requested_vram_mb: Optional[int] = None
    ) -> str:
        """
        Allocates a dedicated GPU for an experiment.
        Ensures strict single-job exclusivity per device for training.
        """
        if not experiment_id:
            raise ResourceAllocationError("Cannot allocate GPU without a valid experiment ID.")

        budget = budget_name.upper()
        vram_needed = requested_vram_mb or self.BUDGET_VRAM_ESTIMATES.get(budget, 4500)

        if vram_needed <= 0:
            raise ResourceAllocationError(f"Invalid VRAM allocation requested: {vram_needed} MB.")

        if vram_needed > self.SAFE_ALLOCATABLE_VRAM_MB:
            raise ResourceAllocationError(
                f"Requested VRAM ({vram_needed} MB) exceeds safe ceiling ({self.SAFE_ALLOCATABLE_VRAM_MB} MB)!"
            )

        if self.device_count == 0:
            if self.active_leases.get(0) is not None:
                raise ResourceAllocationError(
                    f"CPU compute slot is currently locked by experiment '{self.active_leases[0]}'."
                )
            self.active_leases[0] = experiment_id
            self.lease_timestamps[0] = time.time()
            return "cpu"

        # Search for an idle GPU
        for i in range(self.device_count):
            if self.active_leases[i] is None:
                self.active_leases[i] = experiment_id
                self.allocated_vram_mb[i] = vram_needed
                self.lease_timestamps[i] = time.time()
                return f"cuda:{i}"

        active_str = ", ".join(f"GPU {i}: '{self.active_leases[i]}'" for i in range(self.device_count))
        raise ResourceAllocationError(
            f"All {self.device_count} GPUs are currently busy ({active_str}). "
            "Concurrent multi-job execution on a single GPU is forbidden to prevent VRAM oversubscription."
        )

    def release_gpu(self, device_str: str) -> None:
        """Releases the lease and allocated memory on the specified device."""
        idx = self.validate_device(device_str)
        self.active_leases[idx] = None
        self.allocated_vram_mb[idx] = 0
        self.lease_timestamps[idx] = None
        if torch.cuda.is_available():
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass

    def recover_stale_leases(self, max_age_seconds: float = 3600.0) -> List[str]:
        """
        Detects and releases stale locks older than max_age_seconds.
        Prevents failed or hung jobs from permanently blocking GPUs.
        """
        recovered: List[str] = []
        now = time.time()
        for idx, lease in list(self.active_leases.items()):
            if lease is not None:
                ts = self.lease_timestamps.get(idx)
                if ts is not None and (now - ts) > max_age_seconds:
                    device_str = "cpu" if self.device_count == 0 else f"cuda:{idx}"
                    self.release_gpu(device_str)
                    recovered.append(f"{device_str} (leased to '{lease}')")
        return recovered

    @contextmanager
    def lease(
        self,
        experiment_id: str,
        budget_name: str,
        requested_vram_mb: Optional[int] = None
    ):
        """
        Context manager guaranteeing release of GPU slot on normal exit or exception.
        """
        device = self.allocate_gpu(experiment_id, budget_name, requested_vram_mb)
        try:
            yield device
        finally:
            self.release_gpu(device)
