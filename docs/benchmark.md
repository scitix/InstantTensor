# Full benchmarks

## Test setting

Versions:
+ safetensors 0.7.0
+ fastsafetensors 0.2.2
+ runai_model_streamer 0.15.6 
+ instanttensor 0.1.4
<!-- vllm 0.13.0 -->

<!-- enable distributed mode of runai_model_streamer in vllm through --model-loader-extra-config '{ "distributed": true }' -->

## Test on 50GB/s storage

| Model         | GPU    | Backend                | Load Time (s) | Throughput (GB/s) | Speedup   |
|---            |---     | ---                    |---            |---                |---        |
| Qwen3-30B-A3B | 1*H200 | Safetensors            | 57.4          | 1.07              | 1x        |
|               |        | fastsafetensors        | 22.2          | 2.75              | 2.59x     |
|               |        | RMS                    | 9.29          | 6.57              | 6.18x     |
|               |        | InstantTensor          | 1.77          | 34.5              | **32.4x** |
| DeepSeek-R1   | 8*H200 | Safetensors            | 160           | 4.30              | 1x        |
|               |        | fastsafetensors        | 75.8          | 9.08              | 2.11x     |
|               |        | RMS                    | 101           | 6.82              | 1.58x     |
|               |        | RMS (distributed mode) | 192           | 3.59              | 0.83x    |
|               |        | InstantTensor          | 15.3          | 45.0              | **10.5x** |

<!-- Qwen3-30B-A3B: 61.07GB -->
<!-- DeepSeek-R1: 688.6GB -->

*RMS: Run:ai Model Streamer

*Run DeepSeek-R1 with TP=8 and EP=8
<!-- -tp 8 -ep -->

*In multi-GPU loading, the load time is defined as the time until the last (slowest) GPU finishes loading.

## Test on in-memory file system (tmpfs)

Testbed: 2 numa nodes * 8 channels DDR5 per numa node

| Model         | GPU    | Backend                | Load Time (s) | Throughput (GB/s) | Speedup   |
|---            |---     | ---                    |---            |---                |---        |
| Qwen3-30B-A3B | 1*H200 | Safetensors            | 12.7          | 4.81              | 1x        |
|               |        | fastsafetensors        | 16.7          | 3.66              | 0.76x     |
|               |        | RMS                    | 8.21          | 7.44              | 1.55x     |
|               |        | InstantTensor          | 1.26          | 48.5              | **10.1x** |
| DeepSeek-R1   | 8*H200 | Safetensors            | 22.6          | 30.5              | 1x        |
|               |        | fastsafetensors        | 63.4          | 10.9              | 0.36x     |
|               |        | RMS                    | 79.8          | 8.63              | 0.28x     |
|               |        | RMS (distributed mode) | 156           | 4.41              | 0.14x     |
|               |        | InstantTensor          | 12.1          | 56.9              | **1.87x** |

## Intel XPU

These are experimental results from the [host runtime](./loader-internals.md#101-host-runtime-intel-xpu). They use `tests/test.py` with
`cuda` → `xpu` and `nccl` → `xccl`, `copy=True` (`copy=False` is not supported on
XPU), and no load process group (`supports_process_group("xpu")` is `False`), so
each rank reads every file.

Versions:
+ safetensors 0.8.0
+ torch 2.14.0+xpu

Testbed: Intel Xeon 6980P (2 NUMA nodes), Intel Arc Pro B60 24 GB.

The model is Qwen3-30B-A3B, but not all of it, because a B60 has 24 GB. 1 GPU loads the first 4 shards (16.0 GB); TP=2 reads the first 8 shards (32.0 GB). Each number is the median of 3 runs. fastsafetensors and Run:ai Model Streamer don't support XPU.

### NVMe storage (consumer QLC, cold page cache)

| Model                    | GPU   | Backend                   | Load Time (s) | Throughput (GB/s) | Speedup  |
|---                       |---    | ---                       |---            |---                |---       |
| Qwen3-30B-A3B (4 shards) | 1*B60 | Safetensors               | 57.7          | 0.28              | 1x       |
|                          |       | InstantTensor             | 9.39          | 1.70              | **6.1x** |
|                          |       | InstantTensor (AIO)*      | 6.68          | 2.40              | 8.6x     |
|                          |       | InstantTensor (URING)*    | 8.57          | 1.87              | 6.7x     |
|                          |       | InstantTensor (BUFFERED)* | 30.3          | 0.53              | 1.9x     |
| Qwen3-30B-A3B (8 shards) | 2*B60 | Safetensors               | 75.3          | 0.43              | 1x       |
|                          |       | InstantTensor             | 32.5          | 0.98              | **2.3x** |

*Single run with `INSTANTTENSOR_BACKEND` set.

### In-memory file system (tmpfs)

| Model                    | GPU   | Backend       | Load Time (s) | Throughput (GB/s) | Speedup  |
|---                       |---    | ---           |---            |---                |---       |
| Qwen3-30B-A3B (4 shards) | 1*B60 | Safetensors   | 18.7          | 0.86              | 1x       |
|                          |       | InstantTensor | 3.35          | 4.78              | **5.6x** |
| Qwen3-30B-A3B (8 shards) | 2*B60 | Safetensors   | 22.8          | 1.40              | 1x       |
|                          |       | InstantTensor | 6.60          | 4.85              | **3.5x** |

*Throughput is the checkpoint size divided by the load time of the slowest GPU.

TP=2 gains less because, without a process group, each rank reads the whole
checkpoint and copies every full tensor to its device.
