#pragma once

#include <instant_tensor/common.hpp>
#include <instant_tensor/types.hpp>
#include <instant_tensor/io_context.hpp>
#include <liburing.h>

namespace instanttensor {

// Parameters computed in post_read_chunk preamble, passed to IO-specific methods
struct ChunkIOParams {
    chunk_id_t chunk_id;
    const Chunk &chunk;
    const FileInfo &file;
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
    bool need_host_buffer = false;
    bool need_worker_threads = false;
    bool need_cuda_thread = false;
    void *device_buffer = nullptr;
    void* host_buffer = nullptr; // per-thread host buffer for in-memory file
    HostBufferCacheEntry host_buffer_entry = {nullptr, 0, nullptr};
    vector<TensorMetadate> tensors;
    vector<Chunk> chunks;
    size_t current_tensor_index = 0;
    vector<unique_ptr<AsyncExecutor>> worker_threads;
    // A special thread to read the last page of a file when the file size is not page aligned, 
    // which results in blocking I/O even with O_DIRECT and libaio/io_uring.
    unique_ptr<AsyncExecutor> last_page_reader_thread; 
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
    bool aio_context_initialized = false;
    io_context_t aio_ctx = {};
    vector<struct iocb> aio_iocbs;
    vector<struct iocb*> aio_iocb_ptrs;
    vector<struct io_event> aio_events;

    // io_uring (loader_io_uring.cpp)
    bool uring_context_initialized = false;
    bool uring_register_file = true;
    bool uring_register_buffer = true; // **important for buffered IO performance**
    struct io_uring uring_ring = {};
    struct io_uring uring_ring_last_page = {};

    int device_idx = -1;
    ncclComm_t group_communicator = nullptr;
    int rank = -1;
    int world_size = 0;
    size_t buffer_size = 0;
    size_t num_threads = 0;
    size_t io_depth = 0;
    Backend backend = Backend::AIO;

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
    static bool cufile_available();
    void open_file_cufile(FileInfo &f);  // open fd with O_DIRECT, cuFileHandleRegister
    void close_file_cufile(FileInfo &f); // cuFileHandleDeregister, close fd
    void register_device_buffer_cufile();
    void deregister_device_buffer_cufile();
    ChunkRequest post_read_chunk_cufile(const ChunkIOParams &p);

    // AIO path (loader_io_aio.cpp)
    void open_file_aio(FileInfo &f);     // open fd with O_DIRECT, fstat
    void close_file_aio(FileInfo &f);    // close fd
    void initialize_aio_context();        // io_setup, allocate iocb arrays
    void destroy_aio_context();       // io_destroy
    ChunkRequest post_read_chunk_aio(const ChunkIOParams &p);

    // io_uring path (loader_io_uring.cpp)
    static bool uring_available();
    void open_file_uring(FileInfo &f);   // open fd (buffered, no O_DIRECT), fadvise
    void close_file_uring(FileInfo &f);  // close fd
    void initialize_uring_context();           // io_uring_queue_init
    void destroy_uring_context();          // io_uring_queue_exit
    void register_host_buffer_uring();
    void deregister_host_buffer_uring();
    ChunkRequest post_read_chunk_uring(const ChunkIOParams &p);
};

void run_loader(unique_ptr<SPSCQueue<RPCRequest>> input_queue, unique_ptr<SPSCQueue<RPCResponse>> output_queue);

} // namespace instanttensor
