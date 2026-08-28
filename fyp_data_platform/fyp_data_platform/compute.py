from __future__ import annotations

import os
import platform
from typing import Any


def compute_info() -> dict[str, Any]:
    """Return the compute runtime that PyTorch can actually use."""
    result: dict[str, Any] = {
        "profile": os.getenv("FYP_COMPUTE_PROFILE", "cpu"),
        "mode": "cpu",
        "cuda_available": False,
        "pytorch_version": None,
        "cuda_runtime_version": None,
        "cudnn_version": None,
        "device_count": 0,
        "devices": [],
        "cpu": platform.processor() or platform.machine() or "Unknown CPU",
        "logical_cpu_count": os.cpu_count(),
        "architecture": platform.machine(),
        "error": None,
    }
    try:
        import torch

        result["pytorch_version"] = torch.__version__
        result["cuda_runtime_version"] = torch.version.cuda
        result["cudnn_version"] = torch.backends.cudnn.version()
        result["cuda_available"] = bool(torch.cuda.is_available())
        if result["cuda_available"]:
            result["mode"] = "cuda"
            result["device_count"] = torch.cuda.device_count()
            result["devices"] = [
                {
                    "index": index,
                    "name": torch.cuda.get_device_name(index),
                    "memory_bytes": torch.cuda.get_device_properties(index).total_memory,
                    "compute_capability": ".".join(
                        str(value) for value in torch.cuda.get_device_capability(index)
                    ),
                }
                for index in range(result["device_count"])
            ]
    except Exception as error:  # Runtime reporting must not take down the API.
        result["error"] = str(error)
    return result
