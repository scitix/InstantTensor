# InstantTensor Chunk Layout and Loading Pipeline

> **语言：** [English](./chunk-layout.md) | 中文（当前）
>
> **维护说明：** 本文与英文版必须同步更新。

> **版本范围：** 本文描述的是截至 **InstantTensor 0.1.9** 的
> **Chunk Layout and Loading Pipeline**。后续版本若修改 `compute_layout()`、
> ring buffer 复用规则或 I/O backend，需要同步更新本文。

本文基于当前仓库源码，说明 InstantTensor 如何把 safetensors 文件中的连续字节区间切成 chunk，经过 host staging buffer 搬到 device ring buffer，并最终暴露为 PyTorch tensor。I/O 路径重点描述 `io_uring`。

本文的“可复现”是指可以用不同的类、线程库或异步框架实现行为等价的加载器。实现需要保持以下行为，而不要求复制当前 C++ 代码结构：

- 从相同 tensor file offsets 得到等价的 disk/device chunk 边界；
- 保持 file、host、device 三层映射及分布式 rank/thread 切分一致；
- 保证跨 chunk tensor 在 device 上连续；
- 在 host window、CUDA event 和 device ring 复用前等待正确的 completion；
- 使用与 `prefetch_chunk_id` 等价的 watermark，不能覆盖当前仍在使用的 tensor；
- 对外只在 tensor 的最后一个 chunk 完成后返回其 device view。

关键代码：

- [`Loader::compute_layout`](../csrc/loader_common.cpp#L217)
- [`Loader::post_read_chunk`](../csrc/loader_common.cpp#L434)
- [`Loader::post_read_chunk_uring`](../csrc/loader_io_uring.cpp#L217)
- [`safe_open.tensors`](../instanttensor/_impl.py#L708)

## 1. 总体模型

InstantTensor 中没有一份同时常驻于 disk、host 和 device 的统一 `Chunk` 对象。`Chunk` 主要描述一段文件字节及其在 device buffer 中的目标位置：

```cpp
struct Chunk {
    size_t size;
    size_t file_index;
    size_t file_offset;
    size_t device_buffer_offset;
    ...
};
```

同一个逻辑 chunk 在运行时有三种表现：

| 层次 | 内容 | 生命周期 |
| --- | --- | --- |
| disk | `[file_offset, file_offset + size)`，可能带页对齐前缀，并可能和相邻 chunk 重读同一文件页 | 静态布局 |
| host | 当前 rank 对应的 chunk slice，放在 `chunk_id % io_depth` 对应的 pinned staging window 中 | 只保留到该 window 被复用 |
| device | 整个 world chunk 的目标区间；每个 rank 先写自己的 slice，再通过 NCCL all-gather 补齐全部 slice | 在 device ring buffer 中保留到允许被后续 prefetch 覆盖 |

因此三者的关系不是三个等大的副本：

- disk 和 device 描述的是 world-wide chunk；
- 每个进程的 host buffer 只暂存本 rank 负责读取的部分；
- 分布式加载完成后，每个 rank 的 device buffer 都得到完整 world chunk。

### 1.1 Rank 与 `torch.distributed`

在分布式模式中，rank 是 `torch.distributed` process group 中的进程编号，
不是 I/O thread。典型部署方式是每张 GPU 对应一个进程：`world_size` 是
参与进程数，`rank` 的范围为 `[0, world_size)`。调用方负责初始化 process
group 并传给 InstantTensor；InstantTensor 本身不负责拉起这些进程。未传
process group 时使用 `world_size=1`、`rank=0`，并跳过 NCCL all-gather。

每个进程都会打开同一组模型文件，并在自己的 GPU 上拥有独立的 host staging
buffer 和 device buffer。加载一个 chunk 时，各 rank 从文件读取互不重叠的
slice 到本地 buffer；随后使用从 process group 取得的 NCCL communicator 执行
all-gather，使每个进程都得到完整的 world chunk。

## 2. 基本尺寸和对齐

以下符号对应 `Loader::init_buffer()`：

```text
A = thread_alignment = system page size，通常为 4096
T = thread_chunk_size = round_up(user chunk_size, A)
N = num_threads，即 Python concurrency
R = rank_chunk_size = T * N
P = world_size = torch.distributed process group 中的进程数
rank = 当前进程编号，范围为 [0, P)
W = world_chunk_size = R * P
Aw = world_chunk_alignment = A * N * P
D = io_depth
F = first_tensor_alignment = 16
```

这里容易混淆的一点是：Python 参数 `chunk_size` 最终是单个 I/O segment 的上限 `T`，而 C++ `Chunk::size` 的上限通常是 `W = T * N * P`。一个 C++ chunk 会进一步按 rank 和 thread 切分。

Buffer 分配为：

```text
device buffer >= D * W + layout padding
host buffer    = D * R                     # 每个进程只存本 rank 的数据
```

device buffer 额外增加了一小段保守 padding，用于容纳文件页前缀、tensor 首地址对齐和 world chunk 尾部对齐。

### 2.1 有效 buffer size 和初始化顺序

Python 传入的 buffer size 不是 compute_layout() 最终使用值的完整定义。C++ 在布局前执行：

```text
effective_device_buffer_size =
    max(frontend_buffer_size, io_depth * world_chunk_size)

allocated_device_buffer_size =
    effective_device_buffer_size
    + 3 * (first_tensor_alignment + thread_alignment
           + world_chunk_alignment)
```

当前实现随后把 buffer_size 本身更新为包含这段 padding 的 allocation size，compute_layout() 的容量判断使用这个更新后的值。因此行为等价实现若把“用户期望容量”和“实际 allocation 容量”分成两个变量，必须明确布局判断使用哪一个；要复现 0.1.9 的边界，应使用实际 allocation 容量。

资源建立顺序为：

```text
Python:
    read and validate metadata
    choose chunk_size / concurrency / io_depth
    choose frontend buffer_size
    call C++ open

C++:
    open files and initialize io_uring
    derive A, T, R, W, Aw
    enlarge and allocate device buffer
    allocate/register host buffer and register it with io_uring
    create CUDA/NCCL streams, per-window events and executors
    compute_layout(tensor_offsets)
    enter loader RPC/prefetch loop
```

先打开文件的原因是 backend 初始化会决定是否需要 host staging buffer；先分配 host buffer 再注册 fixed buffers，则是 io_uring fixed-buffer 路径的前提。

## 3. `compute_layout` 的输入

Python frontend 首先：

1. 按文件名排序文件；
2. 读取每个 safetensors header；
3. 按 `data_offsets[0]` 排序文件内 tensor；
4. 要求相邻 tensor 的 data offsets 严格连续；
5. 把 safetensors data-relative offset 加上 header 长度，得到绝对文件 offset；
6. 为每个文件追加一个 tensor-data 结束 offset。

传给 C++ 的 `tensor_offsets` 形如：

```text
(file0, tensor0_begin)
(file0, tensor1_begin)
...
(file0, file0_tensor_data_end)
(file1, tensor0_begin)
...
```

每个文件的终止 offset 让相邻两个 offset 可以直接计算 tensor size。跨文件的相邻 pair 只用于结束前一个 chunk、切换文件，不产生 tensor。

0.1.9 的实现还隐含以下输入前提，复现时不能只看 `tensor_offsets` 的元素类型：

- file index 必须从 0 开始连续递增，并且 offsets 按 file、再按文件内地址排列；C++ 通过“最后一个 file index + 1”推导文件数；
- 每个 safetensors 文件至少包含一个 tensor；Python 需要读取排序结果的最后一个元素来追加文件结束 offset；
- 本文针对正常模型文件中的非空 payload。frontend 没有显式拒绝零长度 tensor，但全零长度文件在某些对齐情况下无法形成可供 `last_chunk_id` 等待的 chunk，不应作为 0.1.9 的受支持输入依赖。

行为等价实现可以在入口显式拒绝不满足这些前提的 metadata；如果选择兼容零长度 tensor，则需要单独定义其 completion 和空 view 语义，而不能直接照搬当前 `last_chunk_id` 逻辑。

## 4. `compute_layout` 如何生成 chunk

### 4.1 文件侧布局

一个 chunk 不跨文件。每个文件的第一个 chunk 从第一个 tensor 的绝对文件 offset 向下按页对齐：

```text
chunk_file_offset = round_down(first_tensor_file_offset, A)
left_prefix        = first_tensor_file_offset % A
```

所以第一个 chunk 会包含 safetensors header 尾部或其他不属于 tensor payload 的页内前缀。这个前缀只为满足 direct I/O 对齐，不会暴露给 tensor。

`current_chunk_size` 随连续 tensor 增长；超过 `W` 时切出一个 chunk。除文件尾、device ring wrap 附近外，常规 chunk 大小为 `W`。

完成一个非页对齐 chunk 后：

```text
next_file_offset += round_down(current_chunk_size, A)
remaining_prefix  = current_chunk_size % A
```

`remaining_prefix` 会成为下一个 chunk 的左前缀。这意味着相邻 disk chunk 最多重读 `A - 1` 字节所在的同一文件页。重读的目的仍是保持下一次 I/O 的 file offset 页对齐。

### 4.2 Device 侧布局

chunk 的文件字节按照相对位置映射到 device：

```text
device_address(file_byte) =
    device_buffer
  + chunk.device_buffer_offset
  + (file_byte_offset - chunk.file_offset)
```

在同一个尚未环回的线性布局段中，每完成一个 chunk，device 起点按 `Aw` 向上对齐推进：

```text
next_chunk_device_offset += round_up(chunk.size, Aw)
```

`Aw = page_size * num_threads * world_size` 使一个 padded chunk 能被等分成 `world_size` 个 rank slice，再把每个 rank slice 等分成 `num_threads` 个页对齐 I/O segment。

当下一个完整 tensor 放不进 device buffer 时，`compute_layout` 在 tensor 边界结束当前 chunk，并把后续 tensor 放回 buffer 低地址处。这是 device ring 的“环回”。它不是简单的 `offset % buffer_size`：算法会额外调整新 chunk 起点，使新区域中的第一个 tensor 地址满足 16-byte 对齐。

这里的 16 bytes 不是 I/O 对齐要求。PyTorch 要求 tensor 地址按其 element size
对齐，而 `torch.complex128` 的 element size 为 16 bytes，是可能的最大值。
因此把首个 tensor 对齐到 `F = 16`，即可满足最严格的 element alignment。

只需要显式对齐每个文件的第一个 tensor，以及 device ring wrap 后新连续区域
的第一个 tensor。InstantTensor 当前要求 tensor 按 element size 非递增排列，
并在读取 metadata 时显式验证。对同一连续区域中的 tensor `i`，令 device
地址为 `a_i`，element size 为 `e_i`。支持的 element size 都是最大为 16 的
2 的幂，因此：

```text
a_0 % 16 == 0
tensor_size_i % e_i == 0
e_(i+1) divides e_i
a_(i+1) = a_i + tensor_size_i
```

由归纳法可得每个 tensor 都满足 `a_i % e_i == 0`。文件边界处的 file-page
padding 会打断 payload 的连续映射，而 ring wrap 会在另一段 device 区域重新
放置数据，所以这两个边界之后需要重新对齐首个 tensor。

### 4.3 Tensor 地址和跨 chunk 连续性

每个 tensor 记录：

```text
size
file_index / file_offset
device_buffer_offset
last_chunk_id
prefetch_chunk_id
```

tensor 首地址由所在 chunk 的映射直接得到：

```text
tensor.device_buffer_offset =
    tensor.file_offset
  - chunk.file_offset
  + chunk.device_buffer_offset
```

大 tensor 可以跨多个 chunk。切分发生在完整的 `W` 边界上，而完整 chunk 的 device reservation 也是 `W`，因此 tensor 的后半部分紧接前半部分，device 上没有 gap。只有 tensor 边界、文件边界或 ring wrap 才允许插入 padding。

这就是后续能够从一个首指针直接构造连续 tensor 的原因；运行时不需要对 chunk 做 `cat`、额外拼接或逐 tensor copy。

### 4.4 `prefetch_chunk_id`

device ring 会复用低地址。`compute_layout` 同时追踪：

- 最早仍被认为在 buffer 中的 tensor；
- 当前 device 地址最小的 tensor；
- chunk device offset 的推进和环回位置。

当后续 chunk 的目标区间即将越过或复用旧 tensor 区间时，算法为旧 tensor 写入 `prefetch_chunk_id`。其含义是：

> 用户仍在使用该 tensor 时，loader 最远可以提交到哪个 chunk，而不会覆盖该 tensor 的 device 字节。

运行时 `can_step()` 只允许 `chunk_reading < current_tensor.prefetch_chunk_id` 时继续 prefetch。

### 4.5 行为等价的布局伪代码

下面的伪代码保留了影响布局和 prefetch 安全性的全部状态。复现时可以换容器或拆分函数，但不能省略这些状态转换。

```text
input:
    offsets = [(file_index, absolute_file_offset), ...]
    # 每个文件包含 tensor begin offsets 和一个 data-end offset

state:
    chunks = []
    tensors = array(num_tensors)
    chunk_file_index
    chunk_file_offset
    chunk_device_offset = 0
    current_chunk_size = 0
    tensor_id = 0
    in_buffer_tensor_id = 0
    left_most_tensor_id = 0

tensor_device_offset(file_offset):
    return file_offset - chunk_file_offset + chunk_device_offset

finish_chunk():
    if current_chunk_size == 0:
        return
    chunks.push({
        size: current_chunk_size,
        file_index: chunk_file_index,
        file_offset: chunk_file_offset,
        device_buffer_offset: chunk_device_offset,
    })
    assert chunk_file_offset % A == 0

    # 文件侧仅前进完整页；页内尾部由下一个 chunk 重读
    chunk_file_offset += round_down(current_chunk_size, A)

    # device 侧为 rank/thread 等分保留 Aw 对齐空间
    chunk_device_offset += round_up(current_chunk_size, Aw)
    current_chunk_size %= A

    previous_chunk_id = len(chunks) - 2
    while in_buffer_tensor_id < left_most_tensor_id and
          tensors[in_buffer_tensor_id].device_buffer_offset < chunk_device_offset:
        tensors[in_buffer_tensor_id].prefetch_chunk_id = previous_chunk_id
        in_buffer_tensor_id += 1

reset_device_region(new_left_most_tensor_id):
    assert current_chunk_size < A
    latest_chunk_id = len(chunks) - 1
    while in_buffer_tensor_id < left_most_tensor_id:
        tensors[in_buffer_tensor_id].prefetch_chunk_id = latest_chunk_id
        in_buffer_tensor_id += 1

    # 与当前源码一致：余数为 0 时也会留下 16 bytes，而不是得到 0
    chunk_device_offset = 16 - (current_chunk_size % 16)
    left_most_tensor_id = new_left_most_tensor_id

reset_file(file_index, first_tensor_offset):
    assert current_chunk_size < A
    chunk_file_index = file_index
    chunk_file_offset = round_down(first_tensor_offset, A)
    current_chunk_size = first_tensor_offset % A
    chunk_device_offset =
        round_up(chunk_device_offset, 16)
        + 16
        - (current_chunk_size % 16)

reset_file(offsets[0].file_index, offsets[0].file_offset)

for each adjacent pair (offsets[i], offsets[i + 1]):
    file_index, tensor_file_offset = offsets[i]
    next_file_index, next_offset = offsets[i + 1]

    if file_index != next_file_index:
        finish_chunk()
        reset_file(next_file_index, next_offset)
        continue

    assert current_chunk_size == tensor_file_offset - chunk_file_offset
    tensor_size = next_offset - tensor_file_offset
    assert tensor_size <= device_buffer_size

    # 环回只能发生在 tensor 边界，不能把一个 tensor 拆到 ring 两端
    required_end =
        chunk_device_offset
        + round_up(current_chunk_size + tensor_size, Aw)
    if required_end > device_buffer_size:
        finish_chunk()
        reset_device_region(tensor_id)

    first_device_offset = tensor_device_offset(tensor_file_offset)
    bytes_left = tensor_size

    # 大 tensor 只在完整 W 边界切分，保证 device 字节连续
    while current_chunk_size + bytes_left > W:
        bytes_to_boundary = W - current_chunk_size
        current_chunk_size += bytes_to_boundary
        bytes_left -= bytes_to_boundary
        finish_chunk()

    current_chunk_size += bytes_left
    tensors[tensor_id] = {
        size: tensor_size,
        file_index: file_index,
        file_offset: tensor_file_offset,
        device_buffer_offset: first_device_offset,
        last_chunk_id: len(chunks),   # 当前尚未 finish 的 chunk id
    }
    tensor_id += 1

finish_chunk()

# 第一次处理上一个线性布局段，第二次处理最后一个布局段
reset_device_region(tensor_id)
reset_device_region(tensor_id)
```

复现时最容易漏掉的三个细节是：

1. disk offset 只按完整页前进，因此非页对齐边界会重读文件页；
2. ring wrap 必须发生在 tensor 边界，否则单指针无法表示 tensor；
3. `prefetch_chunk_id` 是从静态 device 地址覆盖关系推导出来的，不是固定的 “tensor chunk id + K”。

## 5. 一个 chunk 的三层映射

对 chunk `c`，运行时先计算：

```text
S  = c.size
Sw = round_up(S, Aw)
Sr = Sw / P                         # padded_rank_size
St = Sr / N                         # padded_thread_size
r0 = Sr * rank                      # rank_offset
rs = clamp(S - r0, 0, Sr)           # rank_size，真正有效的字节数
```

### 5.1 Disk

本 rank 负责：

```text
[c.file_offset + r0, c.file_offset + r0 + rs)
```

该区间再按 thread `i` 分成：

```text
thread_offset = St * i
thread_size   = clamp(S - r0 - thread_offset, 0, St)
read_size     = round_up(thread_size, A)
```

file offset、host address 和 `read_size` 都满足页对齐，支持 `URING` 的 `O_DIRECT`。最后一个 read 可能因为页对齐读到逻辑 chunk 末尾之外；后续 H2D 只复制 `rank_size` 个有效字节。

### 5.2 Host

Host buffer 是按 `io_depth` 循环使用的 pinned staging ring：

```text
window_index  = chunk_id % D
window_offset = window_index * R
thread_dst    = host_buffer + window_offset + thread_offset
```

一个 host window 只对应一个 chunk 的 rank slice，不存其他 rank 的数据。提交 chunk `k` 前，loader 会等待 `k - D` 完成，避免覆盖相同 window。

### 5.3 Device

本 rank 的 H2D 目标为：

```text
rank_dst = device_buffer + c.device_buffer_offset + r0
all_dst  = device_buffer + c.device_buffer_offset
```

单 rank 时，H2D 完成即得到 chunk。多 rank 时，每个 rank 从文件读取不同的连续 slice，然后执行 in-place NCCL all-gather：

```text
rank 0: disk slice 0 -> device slice 0 --+
rank 1: disk slice 1 -> device slice 1 --+--> all-gather --> each GPU has full chunk
...
```

这里的分布式切分是 chunk 字节切分，不是 tensor semantic sharding。一个 tensor 可以跨 rank slice 边界；all-gather 后每张 GPU 上仍得到完整 tensor。

需要区分“有效 payload”和“实际 device 写入跨度”。令：

```text
Sw = round_up(c.size, Aw)
Sr = Sw / P
```

- `P == 1` 时只执行大小为 `c.size` 的 H2D，device reservation 尾部的 padding 不会被这次 copy 写入；
- `P > 1` 时 NCCL 以 `Sr` 为 count 执行 all-gather，因此会写满 `[all_dst, all_dst + Sw)`。其中 `[all_dst + c.size, all_dst + Sw)` 是无 tensor 语义的 padding，内容未定义，不能暴露给 tensor；
- 某个 rank 的有效 disk/H2D slice 可以小于 `Sr`，但 all-gather 的发送和接收粒度仍是完整 `Sr`。布局器必须为完整 `Sw` 保留空间，overwrite 检查也必须把这段 padded NCCL 写入算作潜在覆盖。

## 6. io_uring 加载流程

### 6.1 初始化

`URING` 使用 `O_DIRECT`；`URING_BUFFERED` 使用普通 fd，并调用 `POSIX_FADV_SEQUENTIAL`。两者都会：

- 分配并 CUDA-register pinned host buffer；
- 创建容量为 `io_depth * num_threads` 的 io_uring；
- 注册所有文件为 fixed files；
- 把 host buffer 按最多 1 GiB 的 iovec 注册为 fixed buffers。

单个 read 没有跨越 1 GiB 注册段时使用 `io_uring_prep_read_fixed`，否则退回普通 `io_uring_prep_read`。

### 6.2 提交 SQE

`post_read_chunk_uring()` 为当前 rank 的每个非空 thread segment 创建一个 SQE：

```text
file source -> host window/thread segment
```

Direct 和 buffered io_uring 路径都提交
`round_up(logical_size, PAGE_SIZE)`；buffered I/O 有意复用 direct I/O 的读取
范围，而 H2D 只复制逻辑有效字节。`URING_BUFFERED` 还会设置
`IOSQE_ASYNC`，避免 page cache 命中时在 loader thread 内联执行大块
memcpy；非页对齐的尾段也会设置该标志。

所有 backend 统一使用 `(chunk_id, thread_id)` 作为 segment 坐标，并共享
相同的逻辑长度计算。具有 kernel completion metadata 的 backend 再将其编码为
`segment_id = chunk_id * num_threads + thread_id`：io_uring 存入 SQE
`user_data`，libaio 存入 `iocb->data`。cuFile 和 MMAP 没有对应的 kernel
`user_data`，其 executor request ID 仍是独立机制。`unfinished_cnt` 仍按
chunk 维护。

### 6.3 CQE、H2D 和 NCCL

当前实际线程分工是：

```text
loader thread:
    io_uring_get_sqe -> prepare -> io_uring_submit

cuda_thread (single-thread executor):
    io_uring_wait_cqe / cqe_seen
    -> cudaMemcpyAsync(host rank slice -> device rank slice)
    -> record CUDA event
    -> optional NCCL all-gather on nccl_stream
    -> record completion event

wait_thread (single-thread executor):
    reap cuda_thread task
    -> cudaEventSynchronize
    -> publish chunk completion
```

`cuda_thread` 按 chunk 提交顺序串行处理 completion 和 CUDA/NCCL launch，
但 disk I/O 可以已有多个 chunk 同时在 flight。CQE 可能跨 chunk 乱序返回。
从 segment ID 可同时恢复 `chunk_id` 和 `thread_id`，从而找到正确的
`unfinished_cnt` 和逻辑 segment size。非负 short read 只有在仍覆盖完整逻辑
segment 时才被接受；页对齐 padding 可以没有读满。

`poll_read_chunk()` 和 `wait_read_chunk()` 最终按 chunk id 顺序 reap `wait_thread` 的 completion request，所以公开的 `chunk_read` 始终代表一个连续完成前缀，不会跳过未完成 chunk。

### 6.4 行为等价的运行时状态机

复现加载逻辑至少需要以下单调状态：

```text
current_tensor_index = 0
chunk_reading = -1   # 已提交的最大 chunk id
chunk_read = -1      # 已完整完成的连续前缀的最大 chunk id
```

核心状态机可以写成：

```text
can_step():
    if layout 尚未初始化:
        return false

    tensor = tensors[current_tensor_index]
    limit = tensor.prefetch_chunk_id
    assert chunk_reading <= limit
    return chunk_reading < limit

post_read_chunk():
    chunk_id = chunk_reading + 1
    chunk_reading = chunk_id
    c = chunks[chunk_id]

    # 同时保护 executor queue 容量、host window 和 CUDA event 的复用
    reuse_guard =
        max(chunk_id - MAX_PREFETCH_CHUNKS,
            chunk_id - io_depth)
    if reuse_guard >= 0:
        wait_read_chunk(reuse_guard)

    p = make_runtime_mapping(c, chunk_id)
    completion = submit_uring_h2d_allgather(p)
    chunks[chunk_id].completion = completion

poll_read_chunk():
    next = chunk_read + 1
    while next <= chunk_reading:
        if not try_reap(chunks[next].completion):
            break
        next += 1
    chunk_read = next - 1

wait_read_chunk(target):
    assert target <= chunk_reading
    next = chunk_read + 1
    while next <= target:
        reap(chunks[next].completion)
        next += 1
    chunk_read = next - 1

step():
    post_read_chunk()
    poll_read_chunk()

wait_step(target_chunk):
    while chunk_read < target_chunk:
        if can_step():
            step()
        else:
            wait_read_chunk(target_chunk)

get_tensor_ptr(tensor_index):
    assert tensor_index >= current_tensor_index
    current_tensor_index = tensor_index
    wait_step(tensors[tensor_index].last_chunk_id)
    return device_buffer + tensors[tensor_index].device_buffer_offset
```

submit_uring_h2d_allgather() 的行为顺序为：

```text
1. 根据第 5 节公式，为当前 rank 的每个非空 thread segment 准备 SQE。
2. 编码 segment_id = chunk_id * num_threads + thread_id，并读取
   disk segment -> host window segment。
3. 提交所有 SQE，设置 chunks[chunk_id].unfinished_cnt。
4. 向单线程 CUDA executor 提交任务：
   a. 解码每个 CQE 的 segment ID，拒绝 bytes_read < logical_size，并递减
      对应 chunk 的 unfinished_cnt；
   b. 直到当前 chunk 的 unfinished_cnt 变为 0；
   c. cudaMemcpyAsync(host rank slice -> device rank slice)；
   d. 在 cuda_stream 上记录 window 对应 event；
   e. 若 world_size > 1，让 nccl_stream 等待该 event，执行 in-place all-gather，
      再在 nccl_stream 上重录同一个 event。
5. 向 completion executor 提交任务：
   a. 等待上述 CUDA executor task 已完成 launch；
   b. cudaEventSynchronize(event)；
   c. 将该 task 作为 chunk completion handle 返回。
```

实现可以用 coroutine、future 或其他 event loop 替换当前 executor，但 completion 的语义必须是“disk read、H2D 以及可选 all-gather 均已结束”。只有这种 completion 才能安全地推进 chunk_read 并复用 host window/event。

### 6.5 Loader 主循环与 prefetch 推进

Loader thread 同时处理 Python RPC 和后台 prefetch。等价主循环为：

```text
while not stopping:
    if not can_step() or rpc_queue is not empty:
        request = rpc_queue.pop()      # 必要时阻塞
        result = dispatch(request)
        response_queue.push(result)

    if can_step():
        step()
    else:
        poll_read_chunk()
```

其效果是：

- OPEN 完成后，即使 Python 尚未请求第一个 tensor，也会开始预取；
- prefetch 最远只到当前 current_tensor_index 的 watermark；
- Python 请求下一个 tensor 时会更新 current_tensor_index，从而开放新的安全预取区间；
- RPC 有优先处理机会，tensor 请求不会被无限 prefetch 饿死；
- chunk_read 只按连续完成前缀推进，即使 CQE 乱序，也不会过早宣布后续 tensor 可用。

## 7. Chunk 如何成为 PyTorch tensor

`get_tensor_ptr(index)`：

1. 要求 tensor 基本按 file offset 顺序访问；
2. 等待 `tensor.last_chunk_id` 完成；
3. 返回 `device_buffer + tensor.device_buffer_offset`。

C++ binding 将该地址和 tensor byte size 包装成一维 `int8` DLPack tensor。Python 随后执行：

```python
tensor_int8 = torch.from_dlpack(dl_tensor)
tensor = tensor_int8.view(torch_dtype).view(shape)
```

这里仍然没有数据拼接：DLPack 只是为已经连续的 device 字节建立 view。dtype 和 shape 来自 safetensors metadata。

- `copy=True`：再执行 `tensor.clone()`，返回拥有独立 storage 的 tensor；
- `copy=False`：直接返回 device ring buffer view，后续 prefetch 可能覆盖它。

## 8. Prefetch 和 compute overlap

Loader 自身运行在独立线程中。完成 `OPEN` 后，它会在没有 Python RPC 请求时持续调用 `try_step()`，直到当前 tensor 的 `prefetch_chunk_id` 限制或 inflight 限制阻止继续提交。

主要上限有两层：

1. **Host/inflight 上限**：提交 chunk `k` 前等待
   `max(k - MAX_PREFETCH_CHUNKS, k - io_depth)`，因此实际 inflight 不超过
   `min(io_depth, MAX_PREFETCH_CHUNKS)`，并确保 host window/event 可复用；
2. **Device overwrite 上限**：`can_step()` 不允许超过当前 tensor 的
   `prefetch_chunk_id`。

典型时间线：

```text
loader/io_uring:  [read T0 chunks][read T1 chunks][read T2 chunks]...
cuda/nccl:             [H2D/AG T0][H2D/AG T1][H2D/AG T2]...
user stream:                       [compute/copy T0] [compute/copy T1]...
```

请求 tensor `Ti` 时只保证覆盖 `Ti` 的最后一个 chunk 已完成；后续安全 chunk 可以继续进行 disk I/O、H2D 和 NCCL，与用户对 `Ti` 的 compute/copy 重叠。

Python generator 在每次获取下一个 tensor 前同步当前 CUDA stream。这保证上一轮 `clone()` 或用户在当前 stream 上的消费已完成，然后才允许 loader 根据新的 `current_tensor_index` 扩大 prefetch 窗口。若用户在其他 stream 上使用 `copy=False` view，需要自行建立正确的 stream 同步和生命周期约束。

## 9. 对齐和正确性约束

- 文件内 tensor data offsets 必须连续；frontend 会显式检查。
- 单个 tensor size 不能大于 device buffer；frontend 会至少把显式 buffer 扩大到最大 tensor size。
- 每个文件起点以及每次 ring wrap 后的新区域，其首个 tensor 地址按 16 bytes 对齐，覆盖 PyTorch 最大的 element size。连续布局和已验证的 element size 非递增顺序可以推出同一区域内后续 tensor 均满足各自的对齐；Python 仍在返回前用 `element_size()` 做运行时检查。
- chunk 不跨文件；tensor 不跨文件。
- 完整 `W` chunk 在 device 中连续排列，因此跨 chunk tensor 仍连续。
- disk chunk 可能页级重叠。同一线性 device 布局段中的 chunk reservation 不重叠；ring wrap 后则会有意复用已获准覆盖的旧区间。
- `copy=False` view 的有效期受 device ring overwrite 和 loader close 约束。

## 10. 当前实现中的维护注意点

以下是阅读源码时应特别注意的“代码现状”，不是设计上的抽象：

1. `io_depth` 同时决定 host windows、CUDA events 和 ring queue sizing；它不只是传统意义上的存储队列深度。
2. `concurrency` 在 io_uring 路径中决定每个 full chunk 的 SQE 数量和 chunk world size，不对应一个长期绑定的用户态 I/O thread pool。
3. Device 对齐当前依赖 tensor 按 element size 非递增排布：首 tensor 按 16 bytes 对齐后，整除关系会传播到后续 tensor。InstantTensor frontend 会显式验证这一排列，并在 I/O 前拒绝其他布局；未来版本计划支持这些布局。返回地址的整除检查仍是运行时保护。

这些细节不改变上文描述的数据路径，但修改 buffer sizing、io_uring 并发模型或 dtype 支持时需要考虑。

## 11. 行为等价实现的验收标准

实现者不需要复刻类名或 executor，但应为布局器和加载状态机建立以下检查。

### 11.1 静态布局不变量

对每个 chunk：

```text
0 < chunk.size <= W
chunk.file_offset % A == 0
round_up(chunk.size, Aw) % P == 0
(round_up(chunk.size, Aw) / P) % N == 0
```

对每个 tensor：

```text
tensor.size == next_file_offset - tensor.file_offset
tensor.last_chunk_id 是覆盖 tensor 最后一个字节的 chunk
tensor.device_buffer_offset + tensor.size <= allocated_device_buffer_size
tensor.device_buffer_offset % dtype_itemsize == 0
```

对 tensor 的每个 payload byte delta，必须存在负责该字节的 chunk，并满足：

```text
tensor.device_buffer_offset + delta
==
chunk.device_buffer_offset
+ (tensor.file_offset + delta - chunk.file_offset)
```

若一个 tensor 横跨相邻 chunk，则相邻 payload 片段在 device 上必须首尾相接。disk 页重读和 device padding 不得出现在 tensor payload 中间。

对每个 tensor i，在它已经可见后允许继续预取的区间应满足：

```text
for chunk_id in (tensor[i].last_chunk_id, tensor[i].prefetch_chunk_id]:
    chunk[chunk_id] 的实际 device 写入跨度
    不覆盖 tensor[i] 的 device payload 区间
```

其中 chunk 的实际 device 写入跨度为：

```text
P == 1: [chunk.device_buffer_offset,
         chunk.device_buffer_offset + chunk.size)

P > 1:  [chunk.device_buffer_offset,
         chunk.device_buffer_offset + round_up(chunk.size, Aw))
```

多 rank 时不能只检查有效 payload，因为 NCCL 会写入 padded tail。watermark 可以保守，但不能越过首次可能破坏当前 tensor 的 chunk。

### 11.2 运行时不变量

- 始终满足 `-1 <= chunk_read <= chunk_reading`。
- `chunk_read` 之前的每个 chunk completion 都已被消费，之后不能存在被跳过后仍宣称连续完成的 chunk。
- 同一 host window 或 CUDA event 被 chunk `k + io_depth` 使用前，chunk `k` 已完成。
- 当前 tensor 返回前，`chunk_read >= tensor.last_chunk_id`。
- 当前 tensor 尚可能被用户使用时，不提交超过其 `prefetch_chunk_id` 的 chunk。
- completion 必须覆盖 read、H2D 和可选 all-gather；只完成 SQE 或只完成 H2D launch 都不够。
- 多 rank 下，每个 rank 只从 disk 读取自己的 slice，但 completion 后每个 device chunk 的有效字节必须相同。
- `copy=False` 时，推进到下一个 tensor 前必须确认用户对当前 view 的使用已经结束；0.1.9 通过同步 Python 当前 CUDA stream 实现这一点。

### 11.3 最小场景矩阵

建议至少用以下人工 metadata 测试布局器，不需要真实模型：

| 场景 | 需要验证的行为 |
| --- | --- |
| 第一个 tensor 不在页边界 | 第一个 disk chunk 向下页对齐，tensor device pointer 跳过左前缀 |
| tensor 恰好结束于 `W` | `last_chunk_id` 正确，下一 tensor 从新 full chunk 开始 |
| tensor 大于 `W` | 被切成多个 chunk，但 device payload 完全连续 |
| device buffer 放不下下一个 tensor | 只在 tensor 边界 ring wrap，首地址重新按 16 bytes 对齐 |
| 非页对齐 tensor 边界后结束 chunk | 下一 disk chunk 重读页尾，device 中不重复 tensor payload |
| 两个或更多文件 | chunk 不跨文件，每个文件重新计算页前缀和首 tensor 对齐 |
| `world_size > 1` 且末 chunk 很短 | 后部 rank/thread segment 可以为空，all-gather padded slice 仍可等分 |
| `io_depth = 3` 且提交至少 4 个 chunk | 第 4 个 chunk 复用 window 0 前等待 chunk 0 completion |
| CQE 跨 chunk 乱序 | 解码到正确的 chunk/thread segment，`chunk_read` 仍只推进连续前缀 |
| 页对齐 read 返回 short result | `bytes_read >= logical_size` 时接受；缺少逻辑字节时在 H2D 前拒绝 |
| 小 device ring 多次环回 | 每一代 tensor 的 watermark 都在其字节被覆盖前停止 prefetch |
| tensor dtype itemsize 混合 | 每个返回地址都满足对应 itemsize 对齐，否则明确拒绝 |

### 11.4 端到端判定

给定相同 safetensors 文件、chunk 参数、world size 和 buffer size，一个行为等价实现应产生：

1. 相同 tensor 顺序、shape、dtype 和 payload；
2. 相同的 rank/thread 文件切分范围，允许底层 I/O 合并但不能改变有效字节；
3. 等价的 device ring wrap 点和 tensor 连续区间；
4. 不比 0.1.9 更激进的 overwrite watermark；
5. tensor 返回、host/event 复用和 close 时均不存在未完成的数据依赖。

其中第 4 点允许更保守地少预取，但若目标是复现相同 overlap 性能，则应生成相同的 `prefetch_chunk_id`。
