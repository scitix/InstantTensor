# InstantTensor

InstantTensor is an **ultra-fast, distributed Safetensors loader** designed to maximize I/O throughput when moving model weights from Safetensors files to GPU memory.

**Model loading benchmark on inference engines:**

| Model | GPU | Backend | Load Time (s) | Throughput (GB/s) | Speedup |
|---|---:|---|---:|---:|---|
| Qwen3-30B-A3B | 1*H20 | Safetensors   | 57.4  | 1.2 | 1x |
| Qwen3-30B-A3B | 1*H20 | InstantTensor | 1.77 | 39  | <span style="color: green">**32.5x**</span> |
| DeepSeek-R1   | 8*H20 | Safetensors   | 160  | 4.3 | 1x |
| DeepSeek-R1   | 8*H20 | InstantTensor | 15.3 | 45  | <span style="color: green">**10.5x**</span> |

### Quickstart

```python
from instanttensor import safe_open

tensors = {}
with safe_open("model.safetensors", framework="pt", device=0) as f:
    for name, tensor in f.tensors():
        tensors[name] = tensor.clone()
```

> **NOTE:** `tensor` points to the internal buffer of InstantTensor and should be copied immediately (e.g. by clone() or copy_()) to avoid data being overwritten during buffer reuse.

See [Usage](#usage) for more details (multi-file and distributed usage).

## Why InstantTensor?

- **Fast weight loading**: 
  - Direct I/O: Avoid the slow page cache allocation on cold start. Friendly for large models and tight memory budgets.
  - Tuned I/O size and concurrency: Maximize hardware throughput.
  - Pipelining and prefetching: Parallelize and overlap the various stages of transmission.
- **Distributed loading**: Use `torch.distributed` (NCCL) to speed up loading under any parallelism policy (TP/PP/EP/CP/DP).
- **Multiple I/O backends**:
  - GPUDirect Storage
  - Legacy Storage
  - Memory-based Storage

## Installation

### Build from source

**Prerequisites**
- **Linux**
- **CUDA**
- **PyTorch >= 2.8.0**
- System libraries/headers:
  - **libaio**
  - **Boost**
  - **NCCL**
  - **cuFile** 

For Ubuntu/Debian with CUDA pre-installed, the typical installation steps are as follows:

1. **Install libaio and Boost:**
    ```bash
    apt install libaio-dev libboost-dev
    ```

2. **Install PyTorch** (skip if already installed):
    ```bash
    pip install torch>=2.8.0
    ```

3. **Install InstantTensor:**
    ```bash
    cd ./instanttensor
    pip install --no-build-isolation .
    # For a debug build, set "DEBUG=1" before "pip"
    ```

## Usage

### Multi-file mode (recommended)

Passing a list of files allows the backend to plan reads and provides higher throughput than making multiple calls to load single files:

```python
from instanttensor import safe_open

files = ["model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors"]
tensors = {}
with safe_open(files, framework="pt", device=0) as f:
    for name, tensor in f.tensors():
        tensors[name] = tensor.clone()
```

### Distributed loading

InstantTensor can use a `torch.distributed` NCCL process group to coordinate loading and achieve higher throughput compared to running `safe_open` independently on each GPU.

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
        tensors[name] = tensor.clone()
```

> **NOTE:** You can also load weights using a subgroup created via `dist.new_group`, which allows multiple subgroups to load weights independently. For example, if you have TP=8 and PP=2 (i.e., two TP groups), you can create two subgroups and load weights independently on each TP group. In cross-node (multi-machine) scenarios, loading using per-node subgroups can sometimes be faster than loading on the world group. However, for most cases, the world group is a good default choice.

See `tests/test.py` for a full benchmark harness (TP/PP grouping, checksums, etc.).

<!-- ## Performance tuning

Users can specify three key parameters in `safe_open` for performance tuning:

- **`buffer_size`**: The size of the GPU buffer used for tensors in bytes.

- **`chunk_size`**: The size of each file I/O operation in bytes.

- **`concurrency`**: The number of concurrent I/O operations. 

When set to None (the default), InstantTensor will automatically select a value based on the storage type for high performance. Otherwise, the user-supplied value is used.  -->

<!-- ### Environment variables

InstantTensor uses a few environment variables to select I/O strategies:

- **`INSTANTTENSOR_USE_CUFILE`**:
  - `1`: Enable cuFile (GPUDirect Storage) path for disk files
  - `0` (default): Use libaio for disk files
- **`INSTANTTENSOR_USE_INTERNAL_MEMORY_REGISTER`**:
  - `1`: For tmpfs/ramfs files, register the mmapped memory and copy directly to GPU
  - `0` (default): Use an external pinned host buffer + CPU memcpy + async H2D copy -->

## API reference

See [Build API reference](./docs/build_doc.md)

<!-- ## Benchmark -->

<!-- ## Roadmap

- **Supporting loading to CPU**: E.g., CPU inference.
- **Improving scalability**: E.g., Collective loading on 32+ GPUs. -->

## Thanks

Thanks to the AI Systems and Optimization team at ScitiX AI and the Wenfei Wu Lab at Peking University.