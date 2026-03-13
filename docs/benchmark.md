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
