# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from unittest.mock import Mock

import pytest
import torch

from torchtitan.tools.utils import (
    GarbageCollection,
    get_cuda_flash_attention_impl,
    get_local_device,
    get_peak_flops,
)


class _FakeDeviceModule:
    def __init__(self, num_devices: int):
        self.num_devices = num_devices

    def device_count(self) -> int:
        return self.num_devices


def test_gc_debug_collects_once_per_step(monkeypatch: pytest.MonkeyPatch) -> None:
    gc_collect = Mock()
    monkeypatch.setattr("torchtitan.tools.utils.gc.collect", gc_collect)
    garbage_collection = GarbageCollection.__new__(GarbageCollection)
    garbage_collection.debug = True

    for step in (1, 2):
        assert garbage_collection.run(step)
        gc_collect.assert_called_once_with(2)
        gc_collect.reset_mock()


def test_get_local_device_uses_local_rank_when_multiple_devices_visible(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("LOCAL_RANK", "3")
    monkeypatch.setattr("torchtitan.tools.utils.device_type", "cuda")
    monkeypatch.setattr("torchtitan.tools.utils.device_module", _FakeDeviceModule(8))

    assert get_local_device() == torch.device("cuda:3")


def test_get_local_device_uses_zero_when_one_device_visible(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("LOCAL_RANK", "3")
    monkeypatch.setattr("torchtitan.tools.utils.device_type", "cuda")
    monkeypatch.setattr("torchtitan.tools.utils.device_module", _FakeDeviceModule(1))

    assert get_local_device() == torch.device("cuda:0")


def test_get_local_device_rejects_out_of_range_rank(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("LOCAL_RANK", "8")
    monkeypatch.setattr("torchtitan.tools.utils.device_type", "cuda")
    monkeypatch.setattr("torchtitan.tools.utils.device_module", _FakeDeviceModule(8))

    with pytest.raises(ValueError, match="outside the visible cuda device count"):
        get_local_device()


@pytest.mark.parametrize(
    ("capability", "expected_impl"),
    [
        ((8, 0), None),
        ((9, 0), "FA3"),
        ((9, 1), "FA3"),
        ((10, 0), "FA4"),
        ((10, 3), "FA4"),
        # SM 11.0+ falls through to the newest known impl (FA4).
        ((11, 0), "FA4"),
    ],
)
def test_get_cuda_flash_attention_impl(monkeypatch, capability, expected_impl):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda: capability)
    monkeypatch.setattr(torch.version, "hip", None)

    assert get_cuda_flash_attention_impl() == expected_impl


def test_get_cuda_flash_attention_impl_without_cuda(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    assert get_cuda_flash_attention_impl() is None


def test_get_cuda_flash_attention_impl_on_rocm(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.version, "hip", "7.0")

    assert get_cuda_flash_attention_impl() is None


@pytest.mark.parametrize(
    ("device_name", "expected_flops"),
    [
        ("NVIDIA H200", 989e12),
        ("AMD Instinct MI300X", 1300e12),
    ],
)
def test_peak_flops_uses_non_h100_device_without_pci_probe(
    monkeypatch, device_name, expected_flops
):
    # A different H100 in the PCI listing must not replace the selected device.
    run = Mock(return_value=Mock(stdout="NVIDIA H100 PCIe"))
    monkeypatch.setattr("torchtitan.tools.utils.subprocess.run", run)

    assert get_peak_flops(device_name) == expected_flops
    run.assert_not_called()


@pytest.mark.parametrize(
    ("pci_name", "expected_flops"),
    [("NVIDIA H100 NVL", 835e12), ("NVIDIA H100 PCIe", 756e12)],
)
def test_peak_flops_keeps_h100_variant_probe(monkeypatch, pci_name, expected_flops):
    run = Mock(return_value=Mock(stdout=pci_name))
    monkeypatch.setattr("torchtitan.tools.utils.subprocess.run", run)

    assert get_peak_flops("NVIDIA H100") == expected_flops
    run.assert_called_once()


def test_peak_flops_h100_without_lspci_uses_device_name(monkeypatch):
    monkeypatch.setattr(
        "torchtitan.tools.utils.subprocess.run", Mock(side_effect=FileNotFoundError)
    )

    assert get_peak_flops("NVIDIA H100") == 989e12


def test_peak_flops_override_does_not_probe_device(monkeypatch):
    monkeypatch.setenv("GPU_PEAK_FLOPS", "2.25e15")
    probe = Mock(side_effect=AssertionError("Unexpected device probe"))
    monkeypatch.setattr("torchtitan.tools.utils.subprocess.run", probe)
    assert get_peak_flops("unspecified accelerator") == 2.25e15
    probe.assert_not_called()


@pytest.mark.parametrize("value", ["0", "-1", "nan", "inf"])
def test_peak_flops_rejects_invalid_override(monkeypatch, value):
    monkeypatch.setenv("GPU_PEAK_FLOPS", value)
    with pytest.raises(ValueError, match="positive finite"):
        get_peak_flops("unspecified accelerator")


def test_unknown_peak_omits_mfu_without_changing_throughput(monkeypatch):
    from torchtitan.observability.metrics import compute_training_performance_metrics

    monkeypatch.delenv("GPU_PEAK_FLOPS", raising=False)
    peak = get_peak_flops("unspecified accelerator")
    assert peak == 0.0
    metrics = compute_training_performance_metrics(
        num_tokens=4096,
        elapsed_time=2.0,
        non_data_parallel_size=4,
        num_flops_per_token=1e9,
        gpu_peak_flops=peak,
        has_quantization=False,
    )
    assert metrics == {"tokens_per_second": 512.0, "tflops": 0.512}
