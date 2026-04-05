#pragma once

#include <instant_tensor/common.hpp>
#include <instant_tensor/types.hpp>
#include <instant_tensor/io_context.hpp>

namespace instanttensor {

// Parameters computed in post_read_chunk preamble, passed to IO-specific methods
struct ChunkIOParams {
    chunk_id_t chunk_id;
    Chunk &chunk;
    FileInfo &file;
    size_t padded_world_chunk_size;
    size_t padded_rank_size;
    size_t padded_thread_size;
    size_t rank_offset;
    size_t rank_size;
    size_t window_idx;
    size_t window_offset;
    void *rank_dst;
    void *all_dst;
    cudaEvent_t event;
};

class Loader {// NOTE: do not use any python object in pure C++ thread
public:
    unique_ptr<SPSCQueue<RPCRequest>> input_queue;
    unique_ptr<SPSCQueue<RPCResponse>> output_queue;

    vector<FileInfo> file_info;
    bool use_internal_memory_register = false;
    bool use_cufile = false;
    bool need_host_buffer = false;
    bool need_cufile = false;
    bool need_aio = false;
    bool need_worker_threads = false;
    bool need_cuda_thread = false;
    void *device_buffer = nullptr;
    void* host_buffer = nullptr; // per-thread host buffer for in-memory file
    HostBufferCacheEntry host_buffer_entry = {nullptr, 0, nullptr};
    vector<TensorMetadate> tensors;
    vector<Chunk> chunks;
    size_t current_tensor_index = 0;
    vector<unique_ptr<AsyncExecutor>> worker_threads;
    unique_ptr<AsyncExecutor> aio_fallback_thread;
    unique_ptr<AsyncExecutor> cuda_thread;
    unique_ptr<AsyncExecutor> wait_thread;
    std::thread io_depth_sample_thread;
    cudaStream_t cuda_stream = nullptr;
    cudaStream_t nccl_stream = nullptr;
    vector<cudaEvent_t> cuda_events;
    size_t world_chunk_alignment = 0;
    size_t thread_chunk_size = 0;
    size_t rank_chunk_size = 0;
    size_t world_chunk_size = 0;
    size_t thread_alignment = 0;
    size_t rank_alignment = 0;
    size_t world_alignment = 0;
    // must >= sizeof(dtype) for any dtype, torch.complex128.itemsize == 16
    const size_t first_tensor_alignment = 16;
    // NOTE: cudaHostRegisterMapped can be automatically determined.
    int cuda_host_register_flags = 0;
    // size_t prev_file_index = (size_t)-1;

    // aio
    io_context_t aio_ctx = {};
    vector<struct iocb> aio_iocbs;
    vector<struct iocb*> aio_iocb_ptrs;
    vector<struct io_event> aio_events;

    int device_idx = -1;
    ncclComm_t group_communicator = nullptr;
    int rank = -1;
    int world_size = 0;
    size_t buffer_size = 0;
    size_t num_threads = 0;
    size_t io_depth = 0;

    alignas(64)
    atomic<bool> stop = false;
    atomic<chunk_id_t> chunk_reading = -1;
    atomic<chunk_id_t> chunk_read = -1;

    alignas(64)
    atomic<size_t> io_depth_sum = 0;
    atomic<size_t> io_depth_sample = 0;

    // Constructor
    Loader(unique_ptr<SPSCQueue<RPCRequest>> input_queue, unique_ptr<SPSCQueue<RPCResponse>> output_queue);

    // Common methods (loader_common.cpp)
    void open_file();
    void close_file();
    void init_buffer();
    void destroy_buffer();
    void init_threads();
    void destroy_threads();
    void compute_layout(const vector<pair<size_t, size_t>>& tensor_offsets);
    void open(OpenArgs args);
    void close(CloseArgs args);
    void post_read_chunk();
    void poll_read_chunk();
    void wait_read_chunk(chunk_id_t chunk_id);
    void step();
    bool can_step();
    void try_step();
    void* get_tensor_ptr(GetTensorArgs args);
    std::any dispatch(const RPCRequest &m);
    void run();

    // In-memory IO path (loader_io_inmem.cpp)
    void open_file_inmem(FileInfo &f);   // open fd, mmap, optional cudaHostRegister
    void close_file_inmem(FileInfo &f);  // unregister, munmap defer, close fd
    ChunkRequest post_read_chunk_inmem(const ChunkIOParams &p);

    // cuFile IO path (loader_io_cufile.cpp)
    void open_file_cufile(FileInfo &f);  // open fd with O_DIRECT, cuFileHandleRegister
    void close_file_cufile(FileInfo &f); // cuFileHandleDeregister, close fd
    ChunkRequest post_read_chunk_cufile(const ChunkIOParams &p);

    // AIO path (loader_io_aio.cpp)
    void open_file_aio(FileInfo &f);     // open fd with O_DIRECT, fstat
    void open_file_aio_context();        // io_setup, allocate iocb arrays
    void close_file_aio(FileInfo &f);    // close fd
    void close_file_aio_context();       // io_destroy
    ChunkRequest post_read_chunk_aio(const ChunkIOParams &p);
};

void run_loader(unique_ptr<SPSCQueue<RPCRequest>> input_queue, unique_ptr<SPSCQueue<RPCResponse>> output_queue);

} // namespace instanttensor
