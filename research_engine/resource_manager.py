"""
research_engine/resource_manager.py: Dual-GPU Resource Manager for Local RTX 3060 Devices.

Hardware Configuration:
  - GPU 0: NVIDIA GeForce RTX 3060 12 GB
  - GPU 1: NVIDIA GeForce RTX 3060 12 GB
  - RAM: 32 GB
  - CPU: AMD Ryzen 7 7700 (8-core / 16-thread)

Guarantees:
  1. No VRAM oversubscription.
  2. No concurrent execution of two memory-heavy training runs on the same device.
  3. Dynamic tracking of active leases and device availability.
"""

from typing import Dict, List, Optional, Any
import torch


class ResourceAllocationError(Exception):
    """Raised when GPU or compute resources cannot be safely allocated."""
    pass


class GPUResourceManager:
    """
    Manages allocation, VRAM safety limits, and execution locks across available GPUs.
    """
    TOTAL_VRAM_MB = 12288
    SAFE_ALLOCATABLE_VRAM_MB = 10240  # 10 GB limit leaving 2 GB system/CUDA overhead

    BUDGET_VRAM_ESTIMATES = {
        "SMOKE": 1500,     # ~1.5 GB
        "SHORT": 3000,     # ~3.0 GB
        "STANDARD": 4500,  # ~4.5 GB
        "FULL": 6000       # ~6.0 GB
    }

    def __init__(self):
        self.device_count = torch.cuda.device_count() if torch.cuda.is_available() else 0
        self.active_leases: Dict[int, Optional[str]] = {i: None for i in range(max(1, self.device_count))}
        self.allocated_vram_mb: Dict[int, int] = {i: 0 for i in range(max(1, self.device_count))}

    def get_device_status(self) -> List[Dict[str, Any]]:
        """Returns the real-time operational status of all managed devices."""
        status = []
        if self.device_count == 0:
            return [{
                "device_id": -1,
                "name": "CPU",
                "total_vram_mb": 0,
                "allocated_vram_mb": 0,
                "active_lease": self.active_leases.get(0),
                "is_available": self.active_leases.get(0) is None
            }]

        for i in range(self.device_count):
            name = torch.cuda.get_device_name(i)
            allocated = self.allocated_vram_mb.get(i, 0)
            lease = self.active_leases.get(i)
            status.append({
                "device_id": i,
                "device_str": f"cuda:{i}",
                "name": name,
                "total_vram_mb": self.TOTAL_VRAM_MB,
                "allocated_vram_mb": allocated,
                "active_lease": lease,
                "is_available": (lease is None)
            })
        return status

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
        budget = budget_name.upper()
        vram_needed = requested_vram_mb or self.BUDGET_VRAM_ESTIMATES.get(budget, 4500)

        if vram_needed > self.SAFE_ALLOCATABLE_VRAM_MB:
            raise ResourceAllocationError(
                f"Requested VRAM ({vram_needed} MB) exceeds safe ceiling ({self.SAFE_ALLOCATABLE_VRAM_MB} MB)!"
            )

        if self.device_count == 0:
            if self.active_leases.get(0) is not None:
                raise ResourceAllocationError("CPU compute slot is currently locked by another task.")
            self.active_leases[0] = experiment_id
            return "cpu"

        # Search for an idle GPU
        for i in range(self.device_count):
            if self.active_leases[i] is None:
                self.active_leases[i] = experiment_id
                self.allocated_vram_mb[i] = vram_needed
                return f"cuda:{i}"

        raise ResourceAllocationError(
            f"All {self.device_count} GPUs are currently busy. Active leases: {self.active_leases}. "
            "Concurrent multi-job execution on a single GPU is forbidden to prevent VRAM oversubscription."
        )

    def release_gpu(self, device_str: str) -> None:
        """Releases the lease and allocated memory on the specified device."""
        if device_str == "cpu":
            self.active_leases[0] = None
            self.allocated_vram_mb[0] = 0
            return

        if device_str.startswith("cuda:"):
            try:
                idx = int(device_str.split(":")[-1])
                self.active_leases[idx] = None
                self.allocated_vram_mb[idx] = 0
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except (ValueError, IndexError):
                pass
