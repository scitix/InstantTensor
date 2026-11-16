"""Tests for loading to Intel XPU through the host runtime.

On XPU the native pipeline runs on the host runtime (buffers in host memory,
synchronous copies) and each yielded tensor is copied to the device by PyTorch.
With an XPU present the tensors land on it; otherwise the final device copy is
redirected to CPU, which still runs every native stage. CUDA hosts skip this
file: selecting the host runtime is process-wide.
"""

import pytest
import torch

cuda_available = torch.cuda.is_available()
pytestmark = pytest.mark.skipif(
    cuda_available, reason="the host runtime is process-wide; run on a non-CUDA host"
)

if not cuda_available:
    from safetensors import safe_open as reference_open
    from safetensors.torch import save_file

    import instanttensor
    from instanttensor import Backend, safe_open
    from instanttensor._internal import resources

xpu_available = hasattr(torch, "xpu") and torch.xpu.is_available()
DEVICE = "xpu:0"


def _open(paths, **kwargs):
    f = safe_open(paths, framework="pt", device=DEVICE, **kwargs)
    if not xpu_available:
        # Stand in for the XPU copy; everything before it runs for real.
        f.device = torch.device("cpu")
    return f


@pytest.fixture
def checkpoint(tmp_path):
    """Two files, several dtypes, odd sizes, and tensors larger than the buffer."""
    gen = torch.Generator().manual_seed(0)
    shards = [
        {
            "a.bf16": torch.randn(1031, 7, generator=gen).to(torch.bfloat16),
            "b.fp16": torch.randn(4099, generator=gen).to(torch.float16),
            "c.i8": torch.randint(-128, 127, (777,), dtype=torch.int8, generator=gen),
        },
        {
            "d.fp32": torch.randn(1024 * 1024 + 3, generator=gen),
            "e.u8": torch.randint(0, 255, (5,), dtype=torch.uint8, generator=gen),
            "f.big": torch.randn(3 * 1024 * 1024, generator=gen),
        },
    ]
    paths = []
    for index, shard in enumerate(shards):
        path = tmp_path / f"model-{index}.safetensors"
        save_file(shard, str(path))
        paths.append(str(path))
    return paths


def _reference(paths):
    tensors = {}
    for path in paths:
        with reference_open(path, framework="pt") as f:
            tensors.update({name: f.get_tensor(name) for name in f.keys()})
    return tensors


def test_capabilities():
    assert instanttensor.supports_device("xpu")
    assert instanttensor.supports_device("cuda")
    assert not instanttensor.supports_device("npu")
    assert instanttensor.supports_process_group("cuda")
    assert not instanttensor.supports_process_group("xpu")


@pytest.mark.parametrize("buffer_size", [None, 8 * 1024 * 1024])
def test_loads_every_tensor_bit_exact(checkpoint, buffer_size):
    """8 MiB forces ring-buffer reuse across the 12 MiB tensor."""
    expected = _reference(checkpoint)
    with _open(checkpoint, buffer_size=buffer_size) as f:
        got = {name: tensor for name, tensor in f.tensors()}
    assert got.keys() == expected.keys()
    for name, tensor in got.items():
        if xpu_available:
            assert tensor.device.type == "xpu"
        tensor = tensor.cpu()
        assert tensor.dtype == expected[name].dtype
        assert tensor.shape == expected[name].shape
        assert torch.equal(tensor.view(torch.uint8), expected[name].view(torch.uint8)), name


def test_yielded_tensors_own_their_memory(checkpoint):
    with _open(checkpoint, buffer_size=8 * 1024 * 1024) as f:
        tensors = dict(f.tensors())
    expected = _reference(checkpoint)
    # Read after the context closed and the ring buffer was released.
    for name, tensor in tensors.items():
        assert torch.equal(tensor.cpu().view(torch.uint8), expected[name].view(torch.uint8)), name


def test_unsupported_xpu_options_are_refused(checkpoint):
    with pytest.raises(ValueError, match="copy=False is not supported on XPU"):
        safe_open(checkpoint, framework="pt", device=DEVICE, copy=False)
    with pytest.raises(ValueError, match="CUFILE and FGDS are not available on XPU"):
        safe_open(checkpoint, framework="pt", device=DEVICE, backend=Backend.CUFILE)


def test_cuda_is_refused_after_the_host_runtime_is_selected(checkpoint):
    with _open(checkpoint) as f:
        dict(f.tensors())
    with pytest.raises(RuntimeError, match="selected the host runtime for XPU"):
        safe_open(checkpoint, framework="pt", device="cuda:0")


def test_xpu_budget_counts_host_memory_as_shared():
    rank = resources.query_rank_resources(
        torch.device(DEVICE), max_free_mem_usage=0.5
    )
    assert rank.is_integrated
    assert rank.device_free_mem == rank.avail_mem > 0
