#pragma once

#include <instant_tensor/common.hpp>
#include <instant_tensor/types.hpp>
#include <instant_tensor/io_executor.hpp>
#include <instant_tensor/cuda_executor.hpp>
#include <instant_tensor/io_context.hpp>
#include <instant_tensor/io_backend.hpp>

namespace instanttensor {

class Loader {// NOTE: do not use any python object in pure C++ thread
public:
    unique_ptr<SPSCQueue<RPCRequest>> input_queue;
    std::shared_ptr<SPSCQueue<RPCResponse>> output_queue;

    vector<FileInfo> file_info;
    void *device_buffer = nullptr;       // page-aligned pointer for tensor storage
    void *device_buffer_raw = nullptr;   // raw cudaMalloc pointer (for cudaFree)
    void* host_buffer = nullptr; // rank-local staging windows
    HostBufferCacheEntry host_buffer_entry = {nullptr, 0, nullptr};
    vector<TensorMetadate> tensors;
    vector<Chunk> chunks;
    size_t current_tensor_index = 0;
    // Declared before the executors so their workers cannot outlive this storage.
    IOWorkerDriver::Statistics io_stats;
    CUDAWorkerDriver::Statistics cuda_stats;
    // Must outlive IO/CUDA executors and the callbacks that reference it.
    unique_ptr<IOBackend> io_backend;
    unique_ptr<IOExecutor> io_thread;
    unique_ptr<CUDAExecutor> cuda_executor;
    cudaStream_t cuda_stream = nullptr;
    cudaStream_t nccl_stream = nullptr;
    vector<cudaEvent_t> cuda_events;
    size_t world_chunk_alignment = 0;
    size_t rank_chunk_size = 0;
    size_t world_chunk_size = 0;
    size_t rank_alignment = 0;
    // must >= sizeof(dtype) for any dtype, torch.complex128.itemsize == 16
    const size_t first_tensor_alignment = 16;
    // NOTE: cudaHostRegisterMapped can be automatically determined.
    int cuda_host_register_flags = 0;
    // size_t prev_file_index = (size_t)-1;

    int device_idx = -1;
    ncclComm_t group_communicator = nullptr;
    int rank = -1;
    int world_size = 0;
    size_t buffer_size = 0;
    size_t concurrency = 0;
    size_t io_depth = 0;
    Backend backend = Backend::AIO;

    alignas(64)
    atomic<bool> stop = false;
    atomic<chunk_id_t> chunk_reading = -1;
    atomic<chunk_id_t> chunk_read = -1;
    std::exception_ptr fatal_error;

    int loader_task_id = 0;

    int next_loader_task_id();

    // Constructor
    Loader(unique_ptr<SPSCQueue<RPCRequest>> input_queue, std::shared_ptr<SPSCQueue<RPCResponse>> output_queue);

    // Common methods (loader_common.cpp)
    void init_buffer();
    void destroy_buffer();
    void init_threads();
    void destroy_threads();
    void compute_layout(const vector<pair<size_t, size_t>>& tensor_offsets,
                        const vector<bool>& tensor_keep = {});
    void open(OpenArgs args);
    void close(CloseArgs args);
    void post_read_chunk();
    void poll_read_chunk();
    void wait_read_chunk(chunk_id_t chunk_id);
    void step();
    bool can_step();
    void try_step();
    void wait_step(chunk_id_t chunk_id);
    void* get_tensor_ptr(GetTensorArgs args);
    void set_fatal_error(std::exception_ptr error);
    void rethrow_fatal_error() const;
    std::any dispatch(const RPCRequest &m);
    void run();


};

void run_loader(unique_ptr<SPSCQueue<RPCRequest>> input_queue, std::shared_ptr<SPSCQueue<RPCResponse>> output_queue);

} // namespace instanttensor
