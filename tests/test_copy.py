"""Tests for the copy=True/copy=False API on safe_open (issues #8 and #9)."""

import warnings

import pytest
import torch

cuda_available = torch.cuda.is_available()
pytestmark = pytest.mark.skipif(not cuda_available, reason="CUDA required")

if cuda_available:
    from safetensors.torch import save_file

    from instanttensor import safe_open


@pytest.fixture
def small_safetensors(tmp_path):
    tensors = {f"w{i}": torch.arange(64, dtype=torch.float32) + i for i in range(8)}
    path = tmp_path / "tiny.safetensors"
    save_file(tensors, str(path))
    return str(path), tensors


@pytest.fixture
def ring_buffer_safetensors(tmp_path):
    """16 x 4 MiB tensors so an 8 MiB buffer forces ring-buffer reuse."""
    tensors = {
        f"w{i}": torch.full((1024 * 1024,), float(i), dtype=torch.float32)
        for i in range(16)
    }
    path = tmp_path / "ring.safetensors"
    save_file(tensors, str(path))
    return str(path), tensors


def test_copy_true_default_yields_owning_tensors(small_safetensors):
    path, expected = small_safetensors
    collected = {}
    with safe_open(path, framework="pt", device=0) as f:
        for name, tensor in f.tensors():
            collected[name] = tensor
    for name, exp in expected.items():
        torch.testing.assert_close(collected[name].cpu(), exp)


def test_copy_true_tensors_have_independent_storage(small_safetensors):
    path, _ = small_safetensors
    with safe_open(path, framework="pt", device=0, copy=True) as f:
        collected = list(f.tensors())
    intervals = sorted(
        (
            t.data_ptr(),
            t.data_ptr() + t.numel() * t.element_size(),
            name,
        )
        for name, t in collected
    )
    overlaps = [
        ((left_start, left_end, left_name), (right_start, right_end, right_name))
        for (left_start, left_end, left_name), (right_start, right_end, right_name)
        in zip(intervals, intervals[1:])
        if left_end > right_start
    ]
    assert not overlaps, overlaps


def test_copy_true_survives_list_materialization(small_safetensors):
    """Issue #8: list(f.tensors()) then exit, then read."""
    path, expected = small_safetensors
    with safe_open(path, framework="pt", device=0) as f:
        weights_list = list(f.tensors())
    for name, tensor in weights_list:
        torch.testing.assert_close(tensor.cpu(), expected[name])


def test_copy_false_warns_when_buffer_smaller_than_total(ring_buffer_safetensors):
    """Issue #9: small ring buffer + copy=False should warn."""
    path, _ = ring_buffer_safetensors
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        with safe_open(
            path, framework="pt", device=0,
            copy=False, buffer_size=8 * 1024 * 1024,
        ) as f:
            assert f.buffer_size < f.total_tensor_size
            for _ in f.tensors():
                pass
    msgs = [str(x.message) for x in w if issubclass(x.category, UserWarning)]
    assert any("copy=False" in m for m in msgs), msgs


def test_copy_true_does_not_warn_with_small_buffer(ring_buffer_safetensors):
    path, _ = ring_buffer_safetensors
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        with safe_open(
            path, framework="pt", device=0,
            copy=True, buffer_size=8 * 1024 * 1024,
        ) as f:
            for _ in f.tensors():
                pass
    msgs = [str(x.message) for x in w if issubclass(x.category, UserWarning)]
    assert not any("copy=False" in m for m in msgs), msgs


def test_buffer_size_and_total_tensor_size_are_public(small_safetensors):
    path, _ = small_safetensors
    with safe_open(path, framework="pt", device=0) as f:
        assert isinstance(f.buffer_size, int) and f.buffer_size > 0
        assert isinstance(f.total_tensor_size, int) and f.total_tensor_size > 0


def test_tensors_after_exit_raises(small_safetensors):
    path, _ = small_safetensors
    with safe_open(path, framework="pt", device=0) as f:
        pass
    with pytest.raises(RuntimeError, match="context exited"):
        next(iter(f.tensors()))


def test_copy_true_correct_with_small_ring_buffer(ring_buffer_safetensors):
    """Issue #9: copy=True must produce correct data even with small buffer."""
    path, expected = ring_buffer_safetensors
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with safe_open(
            path, framework="pt", device=0,
            copy=True, buffer_size=8 * 1024 * 1024,
        ) as f:
            collected = list(f.tensors())
    for name, tensor in collected:
        torch.testing.assert_close(tensor.cpu(), expected[name])
