# InstantTensor

[![PyPI](https://img.shields.io/pypi/v/instanttensor?logo=pypi&logoColor=white&label=PyPI&color=3775A9)](https://pypi.org/project/instanttensor/) 
[![Downloads](https://img.shields.io/pypi/dm/instanttensor?logo=pypi&logoColor=white&label=PyPI&color=3775A9)](https://pypi.org/project/instanttensor/) 
[![License](https://img.shields.io/badge/License-Apache_2.0-green.svg)](LICENSE)

InstantTensor is an **ultra-fast, distributed Safetensors loader** designed to maximize I/O throughput when moving model weights from Safetensors files to GPU memory.

### Model loading benchmark on inference engines

| Model | GPU | Backend | Load Time (s) | Throughput (GB/s) | Speedup |
|---|---|---|---|---|---|
| Qwen3-30B-A3B | 1*H200 | Safetensors   | 57.4  | 1.1 | 1x |
| Qwen3-30B-A3B | 1*H200 | InstantTensor | 1.77 | 35  | <span style="color: green">**32.4x**</span> |
| DeepSeek-R1   | 8*H200 | Safetensors   | 160  | 4.3 | 1x |
| DeepSeek-R1   | 8*H200 | InstantTensor | 15.3 | 45  | <span style="color: green">**10.5x**</span> |

See [Benchmark](./docs/benchmark.md) for full benchmarks.

### Quickstart

```python
from instanttensor import safe_open

tensors = {}
with safe_open("model.safetensors", framework="pt", device=0) as f:
    for name, tensor in f.tensors():
        tensors[name] = tensor
```

Yielded tensors own their memory by default (`copy=True`). For zero-copy
streaming into preallocated storage, see [Zero-copy mode](#zero-copy-mode).

See [Usage](#usage) for multi-file and distributed usage.

### Used by

- **[vLLM](https://github.com/vllm-project/vllm)**: [Loading Model Weights with InstantTensor](https://docs.vllm.ai/en/latest/models/extensions/instanttensor/)
- **[SGLang](https://github.com/sgl-project/sglang)**: [Model Loading](https://lmsysorg.mintlify.app/docs/advanced_features/model_loading)

## Why InstantTensor?

- **Fast:** Pipeline I/O, GPU transfers, and prefetching.
- **Distributed:** Coordinate reads across GPUs with NCCL.
- **Selective:** Load only the tensors you need.
- **Flexible:** Support Direct I/O, Buffered I/O, MMAP, and GPUDirect Storage.
- **Memory-aware:** Reuse configurable staging buffers, with optional zero-copy views.

## When to Use InstantTensor

- Loading large checkpoints from fast storage.
- Cold starts or model switching with limited host-memory cache.
- Multi-GPU loading from shared checkpoints.
- Loading only part of a checkpoint, such as MTP weights.
- Repeated loading from the page cache or tmpfs.

Performance depends on your hardware and workload; benchmark before choosing.

## Installation

### Requirements

- GPU platforms: CUDA, ROCm
- Framework: PyTorch
- `URING`/`URING_BUFFERED`: Linux kernel 5.6 or newer; 5.15 or newer is recommended
- `FGDS`: the [fgds](https://github.com/Storage-and-OS-for-AI/fgds) kernel module and `libfgds.so` must be installed (available in the process via `LD_PRELOAD` or on the `LD_LIBRARY_PATH`)


### Method 1: Install from pip
  ```bash
  pip install instanttensor
  ```

### Method 2: Build from source
  ```bash
  git clone https://github.com/scitix/InstantTensor.git
  cd InstantTensor
  ./checkout_submodules.sh
  pip install .
  # For a debug build, set "DEBUG=1" before "pip"
  ```

## Usage

### Distributed loading

InstantTensor can use a `torch.distributed` NCCL process group to coordinate loading and achieve higher throughput compared to running `safe_open` independently on each GPU.

Passing all files in a single `safe_open` call lets the backend plan reads across files and typically provides higher throughput than loading each file separately. This applies to both single-GPU and distributed loading.

```python
import torch
import torch.distributed as dist
from instanttensor import safe_open

dist.init_process_group(backend="nccl")
process_group = dist.GroupMember.WORLD

files = ["model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors"]
tensors = {}
with safe_open(files, framework="pt", device=torch.cuda.current_device(), process_group=process_group) as f:
    for name, tensor in f.tensors():
        tensors[name] = tensor
```

> **NOTE:** You can also load weights using a subgroup created via `dist.new_group`, which allows multiple subgroups to load weights independently. For example, if you have TP=8 and PP=2 (i.e., two TP groups), you can create two subgroups and load weights independently on each TP group. In cross-node (multi-machine) scenarios, loading using per-node subgroups can sometimes be faster than loading on the world group. However, for most cases, the world group is a good default choice.

### Selective loading

Use `tensor_filter` to skip tensor payloads before I/O. Buffer sizing,
`keys()`, and `total_tensor_size` include only selected tensors. Reads may
include neighboring bytes required for page alignment.

```python
with safe_open(files, framework="pt", device=0,
               tensor_filter=["model.embed_tokens.weight", "model.layers.0.*"]) as f:
    for name, tensor in f.tensors():
        tensors[name] = tensor
```

Filtering requires `process_group=None`: each GPU loads its own selection
independently. Unfiltered distributed loading is unchanged.
Use `tensor_name_match_mode` to choose how all entries in the list are interpreted:

| Mode | Example | Matching |
|---|---|---|
| `"glob"` (default) | `"model.layers.*.weight"` | Shell-style wildcards (`*`, `?`, `[abc]`); dots are literal. |
| `"exact"` | `"model.layers.0.weight"` | Literal full names, including any wildcard or regex characters. |
| `"regex"` | `r"model\.layers\.\d+\.weight"` | Full-match regular expressions (`re.fullmatch`). |

Matching is case-sensitive by default and selects a tensor if any pattern matches its
whole name; the mode is never inferred from checkpoint contents. In glob mode,
`*` can span dots, and names containing wildcard characters need escaping or
`tensor_name_match_mode="exact"`. `None` selects all tensors and `[]` selects none.
With `copy=False`, consume each tensor inline
because alignment padding can cause buffer reuse even when tensor bytes fit.

### Buffered I/O

For regular disk files, InstantTensor defaults to Direct I/O to prioritize cold
load performance. This is usually the right choice when a model is loaded once
for a long-running workload.

If the same model is loaded repeatedly within a short period, Buffered I/O can
be faster because later reads may benefit from the page cache. Enable it with
the `backend=BackendPolicy.BUFFERED` argument to `safe_open`, or, when
`backend=None`, by setting the `INSTANTTENSOR_BACKEND=BUFFERED` environment
variable.

### Zero-copy mode

Pass `copy=False` to skip the per-tensor clone and yield views into the
internal ring buffer:

```python
with safe_open(files, framework="pt", device=0, copy=False) as f:
    for name, tensor in f.tensors():
        model_param[name].copy_(tensor)
```

Two rules:

1. Consume each tensor before the next is yielded — `list(f.tensors())` and
   similar patterns silently corrupt data when `buffer_size < total_tensor_size`.
2. Do not keep references past the `with` block — the buffer is freed on exit.

A `UserWarning` fires when `copy=False` and `buffer_size < total_tensor_size`.
Both attributes are public on the `safe_open` object.

See `tests/test.py` for a full benchmark harness (TP/PP grouping, checksums, etc.).

### Backend selection

InstantTensor selects an I/O backend automatically by default. You can provide
one or more backend candidates with the `backend` argument to `safe_open`, or
with the `INSTANTTENSOR_BACKEND` environment variable when `backend=None`.
InstantTensor tries the candidates in order and uses the first backend that is
supported by the file system and available on the current system.

Supported backend values are `Backend.AIO`, `Backend.AIO_BUFFERED`,
`Backend.URING`, `Backend.URING_BUFFERED`, `Backend.CUFILE`, `Backend.MMAP`,
and `Backend.FGDS`. The `backend` argument accepts a single `Backend` or a
list of `Backend`/`BackendPolicy` values:

```python
from instanttensor import Backend, BackendPolicy, safe_open

safe_open("model.safetensors", framework="pt", device=0, backend=Backend.URING)
safe_open("model.safetensors", framework="pt", device=0, backend=[Backend.URING, Backend.AIO])
safe_open("model.safetensors", framework="pt", device=0, backend=BackendPolicy.BUFFERED)
```

`BackendPolicy.BUFFERED` expands to `[Backend.URING_BUFFERED, Backend.MMAP]`. This is a
good choice when you want Buffered I/O. `AIO_BUFFERED` remains available when selected
explicitly.

`INSTANTTENSOR_BACKEND` accepts comma-separated backend or policy names:

```bash
INSTANTTENSOR_BACKEND=URING,AIO
INSTANTTENSOR_BACKEND=BUFFERED
```

Backends are used in different file-system and I/O scenarios:

- **In-memory file systems** (available backends: `MMAP`, `URING_BUFFERED`,
  `AIO_BUFFERED`): when model files are
  stored on tmpfs or ramfs, `MMAP` provides the best compatibility and
  performance for this case. The other backends are usually slower for
  in-memory files.
- **Regular file systems**: InstantTensor can use either Direct I/O or Buffered
  I/O.
  - **Direct I/O** (available backends: `AIO`, `URING`, `CUFILE`, `FGDS`) is
    best when a model is loaded once for a long-running
    workload. It avoids page-cache cold-start effects and reduces page-cache
    pollution. When choosing manually, `URING` may be faster on newer platforms.
    `AIO` has the broadest platform compatibility. `CUFILE` requires GPUDirect
    Storage support, and its higher throughput can be offset by cuFile
    initialization overhead. `FGDS` is an alternative GPUDirect Storage
    implementation (see [fgds](https://github.com/Storage-and-OS-for-AI/fgds))
    that requires the `libfgds.so` library and the `fgds` kernel module, and
    typically achieves higher throughput than `CUFILE` without extra
    initialization cost.
  - **Buffered I/O** (available backends: `AIO_BUFFERED`, `URING_BUFFERED`,
    `MMAP`) is best when the same model is
    loaded repeatedly within a short period. Later reads can benefit from the
    page cache, though the first read is usually slower than Direct I/O.
    `URING_BUFFERED` is preferred on platforms with io_uring support;
    `AIO_BUFFERED` provides a more compatible option, while `MMAP` is available
    but usually not preferred.

If no backend is specified, InstantTensor tries `[URING, AIO]` for regular disk
files. For tmpfs/ramfs files, it uses `MMAP`. If none of the requested
candidates can be used, InstantTensor raises an error listing why each candidate
was rejected.

### Environment variables

Set these variables before the first `safe_open` call. An explicit `safe_open`
argument takes precedence over its corresponding environment variable.

| Variable | Description | Default |
| --- | --- | --- |
| `INSTANTTENSOR_BACKEND` | Comma-separated backend or policy candidates, for example `URING,AIO` or `BUFFERED`. | Automatically select by filesystem type. |
| `INSTANTTENSOR_BUFFER_SIZE` | Requested logical GPU tensor ring-buffer size in bytes. It constrains `io_depth` but may be enlarged to fit the largest tensor. | Automatically determined from tensor sizes and I/O settings. |
| `INSTANTTENSOR_CHUNK_SIZE` | File I/O chunk size in bytes. | Automatically determined for the selected backend. |
| `INSTANTTENSOR_CONCURRENCY` | Number of worker threads for `MMAP` and `CUFILE`; other backends ignore it. | Automatically determined for the selected backend. |
| `INSTANTTENSOR_IO_DEPTH` | Maximum number of rank-local I/O operations in flight. Higher values can increase throughput and staging-memory usage; the maximum is 1024. | Automatically determined for the selected backend. |
| `INSTANTTENSOR_MAX_FREE_MEM_USAGE` | Maximum fraction of currently free GPU memory available to the logical device buffer. | `0.5` |
| `INSTANTTENSOR_CACHE_BUFFER` | Set to `1` to cache pinned host staging buffers across loader opens. Cached memory remains pinned until process cleanup. | `0` |
| `INSTANTTENSOR_DEBUG` | Set to `1` to print backend selection, buffer sizes, timing, and throughput diagnostics. | `0` |

The I/O capacity required by a configuration is
`round_up(chunk_size, page_size) * io_depth * world_size`. When `buffer_size`
is omitted, InstantTensor chooses the larger of this value and the
tensor-layout recommendation. When `buffer_size` and `io_depth` are both set,
they must be compatible. When only `buffer_size` is set, InstantTensor reduces
the default `io_depth` as needed. The internal device allocation includes a
small additional alignment guard beyond the logical `buffer_size`. If the
final logical buffer exceeds the resulting device-memory budget, opening fails
before the native allocation is attempted.

For example:

```bash
INSTANTTENSOR_BACKEND=BUFFERED \
INSTANTTENSOR_IO_DEPTH=32 \
INSTANTTENSOR_DEBUG=1 python load_model.py
```

## API reference

See [Build API reference](./docs/build_doc.md)

<!-- ## Benchmark -->

<!-- ## Roadmap
- **Host-device buffer decoupling**: Support larger host buffers and smaller device buffers.
- **Asymmetric loading**: Each GPU loads different tensors.
- **CPU loading**: E.g., for CPU inference.
- **Improved scalability**: E.g., collective loading across 32+ GPUs.
-->

## Thanks

Thanks to the AI Systems and Optimization team at ScitiX AI and the Wenfei Wu Lab at Peking University.
