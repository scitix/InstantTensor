#pragma once

#include <atomic>
#include <any>
#include <cstdint>
#include <functional>
#include <instant_tensor/common.hpp>
#include <instant_tensor/function_executor.hpp>

namespace instanttensor {

using chunk_id_t = ssize_t;

inline size_t rank_logical_size(
    size_t chunk_size, size_t rank_offset, size_t padded_rank_size) {
    return chunk_size > rank_offset
        ? std::min(chunk_size - rank_offset, padded_rank_size)
        : 0;
}

using SingleThreadTaskExecutor = SingleWorkerFunctionExecutor<MAX_IO_DEPTH, MAX_IO_DEPTH>;
using ThreadPoolTaskExecutor = MultiWorkerFunctionExecutor<MAX_IO_DEPTH, MAX_IO_DEPTH>;

struct IOCompletion {
    int request_id;
    std::exception_ptr error;
    bool retry = false;
};

enum class IOSubmitStatus {
    // Dispatched to the kernel/worker; completion will arrive through polling.
    Submitted,
    Retry,
    // The IO stage is complete; CUDA/NCCL still follows the normal pipeline.
    Completed,
};

using IOExecutorBase = SingleWorkerDriverExecutor<std::function<IOSubmitStatus()>, std::any,
                                                MAX_IO_DEPTH, MAX_IO_DEPTH>;

class IOExecutor;
class CUDAExecutor;

// NOTE: edit 
enum Backend {
    AIO,
    AIO_BUFFERED,
    URING,
    URING_BUFFERED,
    CUFILE,
    MMAP,
};

struct BackendStatus {
    bool available;
    string reason;
    string warning;
};

inline bool is_valid_backend(Backend backend) {
    return backend >= Backend::AIO && backend <= Backend::MMAP;
}

inline string backend_to_string(Backend backend) {
    switch(backend) {
        case Backend::AIO: return "AIO";
        case Backend::AIO_BUFFERED: return "AIO_BUFFERED";
        case Backend::URING: return "URING";
        case Backend::URING_BUFFERED: return "URING_BUFFERED";
        case Backend::CUFILE: return "CUFILE";
        case Backend::MMAP: return "MMAP";
    }
    return "UNKNOWN";
}

enum Op {
    OPEN,
    CLOSE,
    GET_TENSOR_PTR,
};

struct OpenArgs {
    vector<string> filenames;
    int device_idx;
    ncclComm_t group_communicator;
    int rank;
    int world_size;
    size_t buffer_size;
    size_t chunk_size;
    size_t concurrency;
    size_t io_depth;
    Backend backend;
    vector<pair<size_t, size_t>> tensor_offsets;
    OpenArgs(const vector<string> &filenames, int device_idx, ncclComm_t group_communicator, int rank,
        int world_size, size_t buffer_size, size_t chunk_size, size_t concurrency, size_t io_depth, Backend backend, const vector<pair<size_t, size_t>>& tensor_offsets)
        : filenames(filenames), device_idx(device_idx), group_communicator(group_communicator), rank(rank), world_size(world_size),
        buffer_size(buffer_size), chunk_size(chunk_size), concurrency(concurrency), io_depth(io_depth), backend(backend), tensor_offsets(tensor_offsets)
        {}
};

struct CloseArgs {
    CloseArgs() {}
};

struct GetTensorArgs {
    size_t tensor_index;
    GetTensorArgs(size_t tensor_index) : tensor_index(tensor_index) {}
};

struct FreeTensorArgs {
    size_t tensor_index;
    FreeTensorArgs(size_t tensor_index) : tensor_index(tensor_index) {}
};

struct RPCRequest {
    int id;
    int op;
    std::any args;
};

struct RPCResponse {
    int id;
    std::any result;
};

struct TensorMetadate {
    size_t size;
    size_t file_index;
    size_t file_offset;
    size_t device_buffer_offset;
    // Empty tensors anchor both IDs to the previous emitted chunk, or -1 if none exists.
    chunk_id_t first_chunk_id;
    chunk_id_t last_chunk_id;
    // Furthest prefetchable chunk without overwriting the tensor's first chunk.
    chunk_id_t prefetch_chunk_id;
};

struct ChunkIOState {
    int io_request_id;
    // Rank-local read range and staging window, fixed before submission and across retries.
    size_t rank_size;
    size_t rank_file_offset;
    size_t window_offset;
    size_t bytes_completed;
    // Loader-owned handle for completion of IO, CUDA and optional NCCL work.
    CUDAExecutor* cuda_executor;
    int cuda_request_id;
};

struct IORequest {
    IOExecutor* executor;
    int wait_handle;
    bool loaded_to_device;
};

struct Chunk {
    size_t size; // size of the chunk in bytes
    size_t file_index;
    size_t file_offset;
    size_t device_buffer_offset;
    ChunkIOState io_state;
};

struct HostBufferCacheEntry {
    void *ptr;
    size_t size;
    std::function<void(void*)> deleter;
};

struct FileInfo {
    string filename;
    int fd;
    bool in_memory;
    off_t size;
    void* mapped_memory;
    CUfileHandle_t cufile_handle;
};

// Transient submission parameters; persistent IO parameters live in chunk.io_state.
struct ChunkIOParams {
    chunk_id_t chunk_id;
    const Chunk &chunk;
    const FileInfo &file;
    size_t window_idx;
    void *rank_dst;
    void *all_dst;
    cudaEvent_t event;
};

} // namespace instanttensor
