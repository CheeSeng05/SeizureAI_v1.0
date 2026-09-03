from __future__ import annotations

from fyp_data_platform.compute import compute_info


def test_compute_info_has_stable_api() -> None:
    info = compute_info()

    assert info["mode"] in {"cpu", "cuda"}
    assert isinstance(info["cuda_available"], bool)
    assert isinstance(info["devices"], list)
    assert "pytorch_version" in info
    assert "logical_cpu_count" in info
