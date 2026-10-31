"""Functional tests for the FGDS backend.

FGDS opens files with O_DIRECT (Direct I/O), so every read must be page-aligned
in memory address, file offset, and transfer length. InstantTensor satisfies
this with a page-aligned device staging buffer plus a D2D copy. These tests
exercise the alignment edge cases: tensors whose sizes are not page-aligned,
sizes straddling page boundaries, ring-buffer reuse across IO windows, and
small chunk sizes that cut chunks at unaligned file offsets.

Note: O_DIRECT is not supported by tmpfs/ramfs, so the tests are skipped when
pytest's temporary directory is memory-backed (probe in the ``odirect_dir``
fixture). To run on a disk-backed path instead, pass a basetemp outside /tmp::

    pytest tests/test_fgds.py --basetemp=/path/on/disk/pytest-tmp

The benchmark counterpart lives in ``tests/benchmark_backends.py``.
"""

import os

import pytest

try:
    import torch

    cuda_available = torch.cuda.is_available()
except ImportError:
    torch = None
    cuda_available = False

pytestmark = pytest.mark.skipif(not cuda_available, reason="CUDA required")

if cuda_available:
    from safetensors.torch import save_file

    from instanttensor import Backend, safe_open
    from instanttensor._impl import _C


def _fgds_available() -> bool:
    available, _, _ = _C.backend_status(Backend.FGDS.value)
    return available


@pytest.fixture(scope="module")
def odirect_dir(tmp_path_factory):
    """Directory on a filesystem that supports O_DIRECT.

    Skips when the FGDS backend is unavailable (libfgds.so and the fgds
    kernel module are required), and when the pytest temporary directory is
    memory-backed: tmpfs/ramfs do not support O_DIRECT, which FGDS requires.
    """
    if not _fgds_available():
        pytest.skip(
            "FGDS backend is not available; install libfgds.so and load the "
            "fgds kernel module (https://github.com/Storage-and-OS-for-AI/fgds)"
        )
    directory = tmp_path_factory.mktemp("fgds_odirect")
    probe = directory / "probe"
    probe.write_bytes(b"\0")
    try:
        fd = os.open(str(probe), os.O_RDONLY | os.O_DIRECT)
    except OSError:
        pytest.skip(
            f"O_DIRECT is not supported on {directory} (memory-backed filesystem); "
            "FGDS requires a Direct-I/O capable filesystem"
        )
    os.close(fd)
    probe.unlink()
    return directory


def _pattern_bytes(size_bytes: int, seed: int):
    """Deterministic int8 pattern tensor of exactly ``size_bytes`` bytes."""
    values = (torch.arange(size_bytes, dtype=torch.int64) * 31 + seed) % 255 - 127
    return values.to(torch.int8)


def _pattern_float_numel(numel: int, seed: int):
    """Deterministic float32 pattern tensor with ``numel`` elements."""
    return (
        (torch.arange(numel, dtype=torch.float64) * 0.25 + seed) % 1024.0
    ).to(torch.float32)


@pytest.fixture(scope="module")
def unaligned_safetensors(odirect_dir):
    """Tensors with byte sizes that are neither page- nor 16-byte aligned.

    Sizes: 6 B, 4095 B (one page minus one), 4097 B (one page plus one),
    and 12345 B (odd middle size). The safetensors JSON header also shifts
    the first tensor to an unaligned file offset.
    """
    tensors = {
        "tiny": _pattern_bytes(6, 1),
        "page_minus_one": _pattern_bytes(4095, 2),
        "page_plus_one": _pattern_bytes(4097, 3),
        "odd_middle": _pattern_bytes(12345, 4),
    }
    path = odirect_dir / "unaligned.safetensors"
    save_file(tensors, str(path))
    return str(path), tensors


@pytest.fixture(scope="module")
def ring_buffer_safetensors(odirect_dir):
    """16 x 4 MiB tensors so an 8 MiB buffer forces ring-buffer reuse."""
    tensors = {
        f"w{i}": _pattern_float_numel(1024 * 1024, i) for i in range(16)
    }
    path = odirect_dir / "ring.safetensors"
    save_file(tensors, str(path))
    return str(path), tensors


@pytest.fixture(scope="module")
def multi_file_safetensors(odirect_dir):
    """Three safetensors files to exercise cross-file chunk boundaries."""
    tensors_per_file = []
    filenames = []
    for file_idx in range(3):
        # NOTE: tensors are listed in decreasing dtype size order (FP32 -> INT8),
        # matching the layout assumption of safetensors model files.
        tensors = {
            f"f{file_idx}_a": _pattern_float_numel(700 + file_idx * 3, file_idx * 10 + 2),
            f"f{file_idx}_b": _pattern_bytes(1000 + file_idx, file_idx * 10 + 1),
        }
        path = odirect_dir / f"multi_{file_idx}.safetensors"
        save_file(tensors, str(path))
        filenames.append(str(path))
        tensors_per_file.append(tensors)
    return filenames, tensors_per_file


def _load_all(path, **kwargs):
    # Keep chunk_size/io_depth small: the FGDS defaults (8 MiB chunk, io_depth 128)
    # make the C-side allocate ~1 GiB device buffer plus an equally large staging
    # buffer, which is wasteful for tiny test files and can OOM small GPUs.
    kwargs.setdefault("chunk_size", 64 * 1024)
    kwargs.setdefault("io_depth", 4)
    collected = {}
    with safe_open(path, framework="pt", device=0, backend=Backend.FGDS, **kwargs) as f:
        for name, tensor in f.tensors():
            collected[name] = tensor
    return collected


def test_fgds_loads_unaligned_tensors_correctly(unaligned_safetensors):
    """Sizes below/above one page and non-16-byte-aligned sizes must round-trip."""
    path, expected = unaligned_safetensors
    collected = _load_all(path)
    assert set(collected) == set(expected)
    for name, exp in expected.items():
        tensor = collected[name]
        assert tensor.shape == exp.shape
        assert tensor.dtype == exp.dtype
        torch.testing.assert_close(tensor.cpu(), exp)


def test_fgds_small_chunk_size_crosses_pages(unaligned_safetensors):
    """A 64 KiB chunk size cuts chunks at unaligned file offsets.

    This forces reads of the first/last partial page of every chunk, which
    stresses the page-aligned staging path of the FGDS backend.
    """
    path, expected = unaligned_safetensors
    collected = _load_all(path, chunk_size=64 * 1024)
    for name, exp in expected.items():
        torch.testing.assert_close(collected[name].cpu(), exp)


def test_fgds_ring_buffer_reuse(ring_buffer_safetensors):
    """copy=True must stay correct when the buffer cycles many chunks.

    16 x 4 MiB tensors with an 8 MiB buffer (chunk_size 2 MiB, io_depth 4)
    force the ring buffer to wrap several times during iteration.
    """
    path, expected = ring_buffer_safetensors
    collected = _load_all(
        path, buffer_size=8 * 1024 * 1024, chunk_size=2 * 1024 * 1024,
        io_depth=4, copy=True,
    )
    for name, exp in expected.items():
        torch.testing.assert_close(collected[name].cpu(), exp)


def test_fgds_copy_false_inline_consumption(ring_buffer_safetensors):
    """copy=False tensors are valid when consumed inline during iteration."""
    path, expected = ring_buffer_safetensors
    seen = {}
    with safe_open(
        path, framework="pt", device=0, backend=Backend.FGDS,
        buffer_size=8 * 1024 * 1024, chunk_size=2 * 1024 * 1024,
        io_depth=4, copy=False,
    ) as f:
        for name, tensor in f.tensors():
            seen[name] = tensor.clone().cpu()
    for name, exp in expected.items():
        torch.testing.assert_close(seen[name], exp)


def test_fgds_multi_file(multi_file_safetensors):
    """Loading several files in one safe_open must keep per-file data intact."""
    filenames, expected_per_file = multi_file_safetensors
    collected = _load_all(filenames)
    expected = {name: t for tensors in expected_per_file for name, t in tensors.items()}
    assert set(collected) == set(expected)
    for name, exp in expected.items():
        torch.testing.assert_close(collected[name].cpu(), exp)


def test_fgds_multi_file_small_chunk_size(multi_file_safetensors):
    """Cross-file boundaries combined with unaligned chunk cuts."""
    filenames, expected_per_file = multi_file_safetensors
    collected = _load_all(filenames, chunk_size=64 * 1024)
    expected = {name: t for tensors in expected_per_file for name, t in tensors.items()}
    for name, exp in expected.items():
        torch.testing.assert_close(collected[name].cpu(), exp)
