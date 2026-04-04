#pragma once

#include <instant_tensor/common.hpp>

namespace instanttensor {

using chunk_id_t = ssize_t;

using AsyncExecutor = SPSCAsyncExecutor<MAX_PREFETCH_CHUNKS, MAX_PREFETCH_CHUNKS>;

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
    size_t num_threads;
    size_t io_depth;
    vector<pair<size_t, size_t>> tensor_offsets;
    OpenArgs(const vector<string> &filenames, int device_idx, ncclComm_t group_communicator, int rank,
        int world_size, size_t buffer_size, size_t chunk_size, size_t num_threads, size_t io_depth, const vector<pair<size_t, size_t>>& tensor_offsets)
        : filenames(filenames), device_idx(device_idx), group_communicator(group_communicator), rank(rank), world_size(world_size),
        buffer_size(buffer_size), chunk_size(chunk_size), num_threads(num_threads), io_depth(io_depth), tensor_offsets(tensor_offsets)
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
    // last chunk that holds the tensor's data
    chunk_id_t last_chunk_id;
    // points to the furthest prefetchable chunk without overwriting current tensor's data
    chunk_id_t prefetch_chunk_id;
};

struct ChunkExtraData {
    size_t aio_unfinished_cnt;
};

struct ChunkRequest {
    AsyncExecutor* executor;
    int wait_handle;
};

struct Chunk {
    size_t size; // size of the chunk in bytes
    size_t file_index;
    size_t file_offset;
    size_t device_buffer_offset;
    ChunkRequest request;
    ChunkExtraData extra_data;
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

} // namespace instanttensor
