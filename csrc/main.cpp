#include <torch/extension.h>
// Pytorch 2.7.1
// NOTE: this file requires compile flag -DUSE_C10D_NCCL to enable
// NOTE: put this before ProcessGroup.hpp

#include <torch/csrc/distributed/c10d/ProcessGroup.hpp>
#include <torch/csrc/distributed/c10d/ProcessGroupNCCL.hpp>
#include <c10/util/intrusive_ptr.h>
#include <pybind11/pybind11.h> // NOTE: the pybind11 headers are under site-packages/torch/include/, not installed by "pip install pybind11"
#include <pybind11/stl.h>
#include <pybind11/detail/type_caster_base.h> // This recursively includes many files under pybind11/detail/

#include <cuda_runtime.h>
#include <nccl.h>
#include <cufile.h>

// #include <boost/asio.hpp>

#include <unistd.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/vfs.h> 
#include <linux/magic.h>
// #include <numa.h>
// #include <liburing.h>
# include <libaio.h>

#include <cstdlib>
#include <unordered_map>
#include <vector>
#include <string>
#include <thread>
#include <future>
#include <chrono>
#include <any>

#include "queue.hpp"
#include "async_executor.hpp"

// namespace c10d {

// // This is a friend function of ProcessGroupNCCL in HackedProcessGroupNCCL.hpp
// // This is no longer needed after PyTorch 2.9 since we have ProcessGroupNCCL::getCommPtr(), but we kept it for compatibility with PyTorch 2.7
// std::shared_ptr<c10d::NCCLComm> hackedGetNCCLComm(c10d::ProcessGroupNCCL* pg, const std::string& deviceKey) {
//     return pg->getNCCLComm(deviceKey);
// }

// // This does not work for protected member function
// // class ProcessGroupNCCLExposer: public c10d::ProcessGroupNCCL{
// // public:
// //     friend std::shared_ptr<c10d::NCCLComm> exposedGetNCCLComm(ProcessGroupNCCL* pg, const std::string& deviceKey) {
// //         return pg->getNCCLComm(deviceKey);
// //     }
// // };

// }

// #include <link.h>
// #include <cstdio>
// static int phdr_callback(struct dl_phdr_info *info, size_t size, void *data) {
//   // info->dlpi_name: so 文件绝对/相对路径，dlpi_addr: 加载基址
//   if (info->dlpi_name && *info->dlpi_name) {
//     fprintf(stderr, "SO: %s @ %p\n", info->dlpi_name, (void*)info->dlpi_addr);
//   }
//   return 0; // 继续迭代
// }
// void dump_loaded_sos() {
//   fprintf(stderr, "==== loaded shared objects ====\n");
//   dl_iterate_phdr(phdr_callback, nullptr);
//   fprintf(stderr, "==================================\n");
// }
#define MAX_PREFETCH_CHUNKS 1024
const size_t PAGE_SIZE = sysconf(_SC_PAGESIZE);// typically 4096

namespace py = pybind11;

using std::string;
using std::vector;
using std::optional;
using std::unordered_map;
using std::unique_ptr;
using std::pair;
using std::atomic;
using std::max;
using std::min;

using chunk_id_t = ssize_t;

[[noreturn]]
void print_and_throw(const std::exception& e) {
    fprintf(stderr, "%s\n", e.what());
    throw e;
}

template <typename T>
T DIV_CEIL(T x, T y) {
    return (x + y - 1) / y;
}

template <typename T>   
T ROUND_UP(T x, T y) {
    return DIV_CEIL(x, y) * y;
}

template <typename T>   
T ROUND_DOWN(T x, T y) {
    return x / y * y;
}

#define CUDA_CHECK(_call) \
    do { \
        cudaError_t _e = (_call); \
        if (_e != cudaSuccess) { \
            print_and_throw(std::runtime_error(string(__FILE__) + ":" + std::to_string(__LINE__) + " '" + #_call + "' -> " + std::to_string(_e) + ":" + cudaGetErrorString(_e))); \
        } \
    } while (0)

#define CUFILE_CHECK(_call) \
    do { \
        CUfileError_t _e = (_call); \
        if (_e.err != CU_FILE_SUCCESS) { \
            print_and_throw(std::runtime_error(string(__FILE__) + ":" + std::to_string(__LINE__) + " '" + #_call + "' -> " + std::to_string(_e.err) + ":" + CUFILE_ERRSTR(_e.err))); \
        } \
    } while (0)

#define NCCL_CHECK(_call) \
    do { \
        ncclResult_t _e = (_call); \
        if (_e != ncclSuccess) { \
            print_and_throw(std::runtime_error(string(__FILE__) + ":" + std::to_string(__LINE__) + " '" + #_call + "' -> " + std::to_string(_e) + ":" + ncclGetErrorString(_e))); \
        } \
    } while (0)

std::optional<string> get_env(const string& name) {
    if (const char* val = std::getenv(name.c_str())) {
        return string(val);
    } else {
        return std::nullopt;
    }
}

// Forward declarations for C++ interface functions
namespace instanttensor {

using AsyncExecutor = SPSCAsyncExecutor<MAX_PREFETCH_CHUNKS, MAX_PREFETCH_CHUNKS>;

enum Op {
    OPEN,
    CLOSE,
    GET_TENSOR_PTR,
};

struct OpenArgs {
    vector<string> filenames;
    int device_idx;
    // std::optional<ncclUniqueId> group_handle;
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
    // nothing
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
    std::any args; // NOTE: std::any is not POD type, and is copy-constructible
    // std::promise<std::any> *promise; // NOTE: std::promise is not POD type, but is not copy-constructible
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
    chunk_id_t last_chunk_id; // last chunk that holds the tensor's data
    chunk_id_t prefetch_chunk_id; // maximum chunk that is prefetchable without overwrite the tensor's data
};

struct ChunkExtraData {
    size_t aio_unfinished_cnt;
};

struct ChunkRequest {
    AsyncExecutor* executor;
    int wait_handle;
};

struct Chunk {
    size_t size; // size of the chunk in the file, unpadded
    size_t file_index;
    size_t file_offset;
    size_t device_buffer_offset;
    ChunkRequest request;
    ChunkExtraData extra_data;
};

// template <typename Executor, typename F>
// auto post_with_future(Executor &ex, F&& f)
//   -> std::future<std::invoke_result_t<std::decay_t<F>>>
// {
//     using Fun      = std::decay_t<F>;
//     using Result   = std::invoke_result_t<Fun>;
//     std::packaged_task<Result()> task(std::forward<Fun>(f));
//     auto fut = task.get_future();
//     boost::asio::post(ex,
//         [task = std::move(task)]() mutable {
//             task();
//         });
//     return fut;
// }

// Decide when to call cudaHostRegister, only affects in-memory file (e.g., tmpfs and ramfs).
enum MemoryRegisterMode {
    // The default mode. Provide best performance when the CPU has enough cores for memory copy. It registers an external buffer and needs an extra memory-to-memory copy.
    EXTERNAL,
    // May better than EXTERNAL when the CPU is weak. It directly registers the mmapped memory.
    INTERNAL,
};

bool file_in_memory(const string& filename) {
    struct statfs buf;
    if (statfs(filename.c_str(), &buf)==0) {
        return buf.f_type == TMPFS_MAGIC || buf.f_type == RAMFS_MAGIC;
    }
    return false;
}

class CufileContext {
public:
    CufileContext() {
        auto t0 = std::chrono::high_resolution_clock::now();
        CUFILE_CHECK(cuFileDriverOpen());// // This costs a long time (~2s)
        auto t1 = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> dt = t1 - t0;
        fprintf(stderr, "cuFile Driver Opened in %.2f seconds\n", dt.count());
    }
    ~CufileContext() {
        CUFILE_CHECK(cuFileDriverClose());
        fprintf(stderr, "cuFile Driver Closed\n");
    }
};

class CufileContextInitializer {
    unique_ptr<CufileContext> context;
    std::mutex mutex;
public:
    void initialize() {
        std::lock_guard<std::mutex> lock(mutex);
        if(!context) context = std::make_unique<CufileContext>();
    }
};

struct HostBufferCacheEntry {
    void *ptr;
    size_t size;
    std::function<void(void*)> deleter;
};

class HostBufferCache {// NOTE: do not cache too many entries since get() is O(n)
    std::mutex mutex;
    vector<HostBufferCacheEntry> cached_host_buffers;
public:
    ~HostBufferCache() {
        std::lock_guard<std::mutex> lock(mutex);
        for(auto& entry : cached_host_buffers) {
            entry.deleter(entry.ptr);
        }
    }
    void put(HostBufferCacheEntry entry) {
        std::lock_guard<std::mutex> lock(mutex);
        cached_host_buffers.emplace_back(std::move(entry));
    }
    HostBufferCacheEntry get(size_t size) {
        std::lock_guard<std::mutex> lock(mutex);
        for(size_t i = 0; i < cached_host_buffers.size(); i++) {
            if (cached_host_buffers[i].size >= size) {
                HostBufferCacheEntry entry = std::move(cached_host_buffers[i]);
                cached_host_buffers.erase(cached_host_buffers.begin() + i);
                return entry;
            }
        }
        return HostBufferCacheEntry{NULL, 0, nullptr};
    }
};

class Munmaper {
    std::mutex mutex;
    vector<pair<void*, size_t>> mapped_memory;
public:
    ~Munmaper() noexcept(false) {
        std::lock_guard<std::mutex> lock(mutex);
        for(auto& ptr : mapped_memory) {
            int ret = munmap(ptr.first, ptr.second);
            if (ret != 0) {
                throw std::runtime_error("Failed to munmap: " + std::string(strerror(errno)));
            }
        }
    }
    void add(void* ptr, size_t size) {
        std::lock_guard<std::mutex> lock(mutex);
        mapped_memory.emplace_back(std::make_pair(ptr, size));
    }
};

unique_ptr<CufileContextInitializer> cufile_context_initializer = std::make_unique<CufileContextInitializer>();
unique_ptr<HostBufferCache> host_buffer_cache = std::make_unique<HostBufferCache>();
unique_ptr<Munmaper> munmaper = std::make_unique<Munmaper>();

struct FileInfo {
    string filename;
    int fd;
    bool in_memory;
    off_t size;
    void* mapped_memory;
    CUfileHandle_t cufile_handle;
};

static bool _use_cufile(){
    return get_env("INSTANTTENSOR_USE_CUFILE").value_or("0") == "1";
}

static bool _use_internal_memory_register(){
    return get_env("INSTANTTENSOR_USE_INTERNAL_MEMORY_REGISTER").value_or("0") == "1";
}

class Loader {// NOTE: do not use any python object in pure C++ thread
public:
    unique_ptr<SPSCQueue<RPCRequest>> input_queue;
    unique_ptr<SPSCQueue<RPCResponse>> output_queue;

    bool stop = false;
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
    unique_ptr<AsyncExecutor> cuda_thread;
    unique_ptr<AsyncExecutor> wait_thread;
    cudaStream_t cuda_stream = nullptr;
    cudaStream_t nccl_stream = nullptr;
    vector<cudaEvent_t> cuda_events;
    size_t world_chunk_alignment = 0;
    chunk_id_t chunk_reading = -1;
    chunk_id_t chunk_read = -1;
    size_t thread_chunk_size = 0;
    size_t rank_chunk_size = 0;
    size_t world_chunk_size = 0;
    size_t thread_alignment = 0;
    size_t rank_alignment = 0;
    size_t world_alignment = 0;
    size_t first_tensor_alignment = 16; // must >= sizeof(dtype) for any dtype, torch.complex128.itemsize == 16
    // NOTE: Never use cudaHostRegisterReadOnly for a host-writable buffer.
    //       cudaHostRegisterReadOnly should only be used for mapped weight memory.
    // NOTE: cudaHostRegisterMapped can be automatically determined.
    int cuda_host_register_flags = 0;


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

    Loader(unique_ptr<SPSCQueue<RPCRequest>> input_queue, unique_ptr<SPSCQueue<RPCResponse>> output_queue) {
        this->input_queue = std::move(input_queue);
        this->output_queue = std::move(output_queue);
        this->use_internal_memory_register = _use_internal_memory_register();
        this->use_cufile = _use_cufile();
    }

    // void init_comm() {
    //     if (this->group_handle.has_value()) {
    //         // convert bytes to string
    //         if (this->world_size <= 1) {
    //             print_and_throw(std::runtime_error("world_size must be greater than 1 when group_handle is provided"));
    //         }
    //         ncclUniqueId nccl_uid = this->group_handle.value();
    //         ncclCommInitRank(&this->comm, this->world_size, nccl_uid, this->rank);
    //         fprintf(stderr, "ncclCommInitRank: %d/%d\n", this->rank, this->world_size);
    //     }
    //     else {
    //         if (this->world_size != 1) {
    //             print_and_throw(std::runtime_error("world_size must be 1 when group_handle is not provided"));
    //         }
    //         this->comm = NULL;
    //     }
    // }

    // void destroy_comm() {
    //     if (this->comm) {
    //         ncclCommDestroy(this->comm);
    //     }
    // }

    void open_file() {
        this->file_info.resize(this->file_info.size());
        for(size_t i = 0; i < this->file_info.size(); i++) {
            FileInfo& f = this->file_info[i];
            auto& filename = f.filename;
            f.in_memory = instanttensor::file_in_memory(filename);

            if (f.in_memory) {
                f.fd = ::open(filename.c_str(), O_RDONLY);
                if (f.fd < 0) {
                    throw std::runtime_error("Failed to open file: " + filename);
                }

                struct stat st;
                if (fstat(f.fd, &st) < 0) { throw std::runtime_error("Failed to fstat file: " + filename); }
                f.size = st.st_size;
                // NOTE: 1. Let cudaHostRegister lock the pages (MAP_LOCKED), which is faster than mmap.
                //       2. mmap (without MAP_LOCKED) is fast, however, 
                //          cudaHostRegister is much slower (regardless of whether mmap locks the memory),
                //          so we perform mmap here but let cudaHostRegister overlapped with reading.
                //       3. Flag cudaHostRegisterReadOnly reduce the time consumption of cudaHostRegister.
                f.mapped_memory = mmap(NULL, f.size, PROT_READ, MAP_SHARED, f.fd, 0);
                if (f.mapped_memory == MAP_FAILED) {
                    throw std::runtime_error("Failed to mmap file: " + filename);
                }
                if (this->use_internal_memory_register) {
                    // NOTE: this requires cudaHostRegisterReadOnly since the file is read-only
                    CUDA_CHECK(cudaHostRegister(f.mapped_memory, f.size, this->cuda_host_register_flags | cudaHostRegisterReadOnly));
                }
                else {
                    this->need_host_buffer = true;
                    this->need_worker_threads = true;
                    this->need_cuda_thread = true;
                }
            }
            else {
                f.fd = ::open(filename.c_str(), O_RDONLY | O_DIRECT);
                if (f.fd < 0) {
                    throw std::runtime_error("Failed to open file: " + filename);
                }

                struct stat st;
                if (fstat(f.fd, &st) < 0) { throw std::runtime_error("Failed to fstat file: " + filename); }
                f.size = st.st_size;
                
                if (this->use_cufile) {
                    this->need_cufile = true;
                    this->need_worker_threads = true;
                    this->need_cuda_thread = true;

                    cufile_context_initializer->initialize();

                    CUfileDescr_t descr = {};
                    descr.type = CU_FILE_HANDLE_TYPE_OPAQUE_FD;
                    descr.handle.fd = f.fd;
                    CUFILE_CHECK(cuFileHandleRegister(&f.cufile_handle, &descr));
                }
                else {// aio
                    this->need_aio = true;
                    this->need_host_buffer = true;
                    this->need_cuda_thread = true;
                }
            }
            // Cannot close here since cuFileHandleRegister requires fd to be different in value (int),
            // and closing here cause the OS to reuse fd value.
            // ::close(fd);
        }
        if(this->need_aio) {
            // fprintf(stderr, "aio setup %zu * %zu = %zu\n", this->io_depth, this->num_threads, this->io_depth * this->num_threads);
            int ret = io_setup(this->io_depth * this->num_threads, &this->aio_ctx);
            if(ret < 0){
                print_and_throw(std::runtime_error("Failed to setup aio: " + std::string(strerror(-ret))));
            }
            this->aio_iocbs.resize(this->io_depth * this->num_threads);
            this->aio_iocb_ptrs.resize(this->io_depth * this->num_threads);
            for(size_t i = 0; i < this->io_depth * this->num_threads; i++) {
                this->aio_iocb_ptrs[i] = &this->aio_iocbs[i];
            }
            this->aio_events.resize(this->io_depth * this->num_threads);
        }
    }

    void close_file() {
        for(size_t i = 0; i < this->file_info.size(); i++) {
            FileInfo& f = this->file_info[i];
            if (f.in_memory) {
                if (this->use_internal_memory_register) {
                    CUDA_CHECK(cudaHostUnregister(f.mapped_memory));
                }
                // int ret = munmap(f.mapped_memory, f.size); // very slow, reclaim the page table at exit
                // if (ret != 0) {
                //     throw std::runtime_error("Failed to munmap: " + std::string(strerror(errno)));
                // }
                munmaper->add(f.mapped_memory, f.size);
            }
            else {
                if(this->use_cufile) {
                    cuFileHandleDeregister(f.cufile_handle);
                }
                else {// aio
                    // pass
                }
            }
            ::close(f.fd);
        }
        if(this->need_aio) {
            int ret = io_destroy(this->aio_ctx);
            if(ret < 0){
                print_and_throw(std::runtime_error("Failed to destroy aio: " + std::string(strerror(-ret))));
            }
        }
    }

    void init_buffer() {
        this->thread_alignment = PAGE_SIZE;
        this->rank_alignment = this->thread_alignment * this->num_threads;
        this->world_chunk_alignment = this->rank_alignment * this->world_size;
        if(this->thread_chunk_size % this->thread_alignment != 0) {
            size_t new_chunk_size = ROUND_UP(this->thread_chunk_size, this->thread_alignment); 
            fprintf(stderr, "Enlarge thread_chunk_size from %zu to %zu to align to %zu\n", this->thread_chunk_size, new_chunk_size, this->thread_alignment);
            this->thread_chunk_size = new_chunk_size;
        }
        this->rank_chunk_size = this->thread_chunk_size * this->num_threads;
        this->world_chunk_size = this->rank_chunk_size * this->world_size;

        size_t padded_size = 3 * (this->first_tensor_alignment + this->world_chunk_alignment); // can be any value that >= 0
        this->buffer_size += padded_size;
        CUDA_CHECK(cudaMalloc(&this->device_buffer, this->buffer_size));

        if (this->need_cufile) {
            CUFILE_CHECK(cuFileBufRegister(this->device_buffer, this->buffer_size, 0));
        }
        if (this->need_host_buffer) {
            size_t host_buffer_size = this->io_depth * this->rank_chunk_size;
            this->host_buffer_entry = host_buffer_cache->get(host_buffer_size);
            if (this->host_buffer_entry.ptr == NULL) {
                // NOTE: method1 costs a long time in memory allocation
                // method1:
                // CUDA_CHECK(cudaHostAlloc(&this->host_buffer_entry.ptr, host_buffer_size, cudaHostAllocMapped | cudaHostAllocWriteCombined));// does WriteCombined really help?
                
                // method2:
                // CUDA_CHECK(cudaHostAlloc(&this->host_buffer_entry.ptr, host_buffer_size, cudaHostAllocMapped));

                // method3:
                this->host_buffer_entry.ptr = aligned_alloc(this->thread_alignment, host_buffer_size); //numa_alloc_interleaved(host_buffer_size);
                if (this->host_buffer_entry.ptr == NULL) {
                    throw std::runtime_error("Failed to allocate host buffer: " + std::string(strerror(errno)));
                }
                
                CUDA_CHECK(cudaHostRegister(this->host_buffer_entry.ptr, host_buffer_size, this->cuda_host_register_flags));// | cudaHostRegisterReadOnly));

                this->host_buffer_entry.size = host_buffer_size;
                this->host_buffer_entry.deleter = [=](void *ptr) {
                    // method1/2
                    // CUDA_CHECK(cudaFreeHost(ptr));

                    // method3
                    CUDA_CHECK(cudaHostUnregister(ptr));

                    // if (this->need_aio) {
                    //     // pass
                    // }
                    free(ptr);
                    // numa_free(ptr, host_buffer_size);
                };
            }
            this->host_buffer = (char*)this->host_buffer_entry.ptr;
        }
    }

    void destroy_buffer() {
        if (this->need_cufile) {
            CUFILE_CHECK(cuFileBufDeregister(this->device_buffer));
        }
        CUDA_CHECK(cudaFree(this->device_buffer));
        if (this->need_host_buffer) {
            host_buffer_cache->put(std::move(this->host_buffer_entry));
        }
    }

    void init_threads() {
        auto set_device_func = [=]() { CUDA_CHECK(cudaSetDevice(this->device_idx)); };

        if(this->need_worker_threads) {
            while(this->worker_threads.size() < this->num_threads) {
                this->worker_threads.emplace_back(std::make_unique<AsyncExecutor>());
            }
        }
        if(this->need_cuda_thread) {
            if(!this->cuda_thread) {
                this->cuda_thread = std::make_unique<AsyncExecutor>();
            }
        }

        if(!this->cuda_stream) {
            CUDA_CHECK(cudaStreamCreateWithFlags(&this->cuda_stream, cudaStreamNonBlocking));
        }
        if(!this->nccl_stream) {
            CUDA_CHECK(cudaStreamCreateWithFlags(&this->nccl_stream, cudaStreamNonBlocking));
        }
        if(!this->wait_thread) {
            this->wait_thread = std::make_unique<AsyncExecutor>();
        }

        for(auto &thread : this->worker_threads) {
            thread->post(set_device_func);
        }
        if(this->cuda_thread) {
            this->cuda_thread->post(set_device_func);
        }
        if(this->wait_thread) {
            this->wait_thread->post(set_device_func);
        }
        this->cuda_events.resize(this->io_depth);
        for(size_t i = 0; i < this->io_depth; i++) {
            CUDA_CHECK(cudaEventCreateWithFlags(&this->cuda_events[i], cudaEventDisableTiming));
        }
    }

    void destroy_threads() {
        for (auto& thread : this->worker_threads) {
            thread->join();
        }
        if (this->cuda_thread) {
            this->cuda_thread->join();
        }
        if (this->wait_thread) {
            this->wait_thread->join();
        }
        if (this->cuda_stream) {
            CUDA_CHECK(cudaStreamDestroy(this->cuda_stream));
        }
        if (this->nccl_stream) {
            CUDA_CHECK(cudaStreamDestroy(this->nccl_stream));
        }
    }

    /*
     * @brief Compute: 1) addresses of the tensors in the buffer, 2) each chunk's range, 3) prefetch chunk for each tensor 
     */
    void compute_layout(const vector<pair<size_t, size_t>>& tensor_offsets) {// list[(file_index, tensor_offset)]
        // NOTE: One chunk is a group of contiguous tensors, may have left and right non-tensor paddings.
        //       It may correspond to at most three types of storage: file, host buffer, device buffer.
        //       The size and offset of the file storage and the host buffer of a chunk is aligned to thread_alignment (== PAGE_SIZE) 
        //       since libaio with O_DIRECT requres page-aligned I/O. (cuFile does not require this)
        //       The size of the device buffer of a chunk is aligned to world_chunk_alignment for ncclAllGather, and the offset is set to let tensors aligned (see below).
        // NOTE: We set the device buffer offset of the first chunk of a file properly 
        //       to let the device buffer offset of the first tensor be aligned to first_tensor_alignment.
        //       Since for any dtype (FP32/FP16/...), first_tensor_alignment % sizeof(dtype) == 0, the first tensor is always dtype-aligned.
        //       Consecutive tensors in current and consecutive chunks are then also dtype-aligned since the tensors 
        //       are listed in the order of decreasing dtype size (e.g., FP32->FP16/BF16->FP8/INT8) in safetensors files.
        size_t chunk_file_index = 0;
        size_t chunk_file_offset = 0; // The chunk offset from the beginning of the file, aligned to thread_alignment (typically == PAGE_SIZE)
        size_t chunk_device_buffer_offset = 0; // TODO: rename. The chunk offset from the beginning of the device buffers, aligned to world_chunk_alignment
        size_t unchunked_file_size = 0;
        size_t tensor_id = 0;
        size_t in_buffer_tensor_id = 0;
        size_t left_most_tensor_id = 0;
        size_t num_files = tensor_offsets[tensor_offsets.size() - 1].first + 1;
        size_t num_tensors = tensor_offsets.size() - num_files; // for each file, num_offsets = num_tensors + 1
        this->tensors.resize(num_tensors);

        auto tensor_device_buffer_offset = [&](size_t tensor_file_offset) -> size_t {// TODO: rename
            return tensor_file_offset - chunk_file_offset + chunk_device_buffer_offset;
            // tensor_file_offset - chunk_file_offset == unchunked_file_size
        };

        auto finish_chunk = [&](size_t chunk_size) {
            if(chunk_size > 0) {
                if(unchunked_file_size < chunk_size) {
                    throw std::runtime_error("Internal error: Unchunked size is smaller than the chunk size.");
                }
                // ROUND_UP/DOWN make real effect only at the right most chunk
                this->chunks.push_back(Chunk{chunk_size, chunk_file_index, chunk_file_offset, chunk_device_buffer_offset, {}, {}});

                if(chunk_file_offset % this->thread_alignment != 0) {
                    throw std::runtime_error("Internal error: Chunk alignment error.");
                }

                chunk_file_offset += ROUND_DOWN(chunk_size, this->thread_alignment);// may reread the last file page
                chunk_device_buffer_offset += ROUND_UP(chunk_size, this->world_chunk_alignment); 
                unchunked_file_size -= ROUND_DOWN(chunk_size, this->thread_alignment); // "-= chunk_size" and then "+= left_padding"

                chunk_id_t prev_chunk_id = (chunk_id_t)this->chunks.size() - 2;
                while(in_buffer_tensor_id < left_most_tensor_id 
                    && this->tensors[in_buffer_tensor_id].device_buffer_offset < chunk_device_buffer_offset) {
                    this->tensors[in_buffer_tensor_id].prefetch_chunk_id = prev_chunk_id;
                    in_buffer_tensor_id++;
                } 
            }
        };
        auto reset_chunk_buffer_offset = [&](size_t new_left_most_tensor_id) {
            if(unchunked_file_size >= this->thread_alignment) {
                print_and_throw(std::runtime_error("Internal error: Unchunked page detected when resetting chunk buffer offset."));
            }
            chunk_id_t latest_chunk_id = (chunk_id_t)this->chunks.size() - 1;
            while(in_buffer_tensor_id < left_most_tensor_id) {
                this->tensors[in_buffer_tensor_id].prefetch_chunk_id = latest_chunk_id;
                in_buffer_tensor_id++;
            }
            chunk_device_buffer_offset = this->first_tensor_alignment - unchunked_file_size % this->first_tensor_alignment; // make the first tensor address aligned to first_tensor_alignment
            left_most_tensor_id = new_left_most_tensor_id;
        };
        auto reset_chunk_file = [&](size_t file_index, size_t file_offset) {
            if(unchunked_file_size >= this->thread_alignment) {
                print_and_throw(std::runtime_error("Internal error: Unchunked page detected when resetting chunk file offset."));
            }
            chunk_file_index = file_index;
            chunk_file_offset = ROUND_DOWN(file_offset, this->thread_alignment);
            unchunked_file_size = file_offset % this->thread_alignment;
            chunk_device_buffer_offset = ROUND_UP(chunk_device_buffer_offset, this->first_tensor_alignment) + this->first_tensor_alignment - unchunked_file_size % this->first_tensor_alignment; // make the first tensor address aligned to first_tensor_alignment
        };


        reset_chunk_file(tensor_offsets[0].first, tensor_offsets[0].second);

        for (size_t i = 0; i < tensor_offsets.size() - 1; i++) {
            size_t file_index = tensor_offsets[i].first;
            size_t tensor_file_offset = tensor_offsets[i].second;
            size_t next_file_index = tensor_offsets[i + 1].first;
            size_t next_tensor_offset = tensor_offsets[i + 1].second;
            if (file_index != next_file_index) {
                finish_chunk(unchunked_file_size);
                reset_chunk_file(next_file_index, next_tensor_offset);
                continue;
            }
            if(unchunked_file_size != tensor_file_offset - chunk_file_offset) {
                throw std::runtime_error("Internal error: Unchunked file size mismatch.");
            }
            size_t tensor_size = next_tensor_offset - tensor_file_offset;
            if (tensor_size > this->buffer_size) {
                print_and_throw(std::runtime_error("Internal error: Buffer size is smaller than a single tensor size. This should be forbidden by the python frontend."));
            }
            if (chunk_device_buffer_offset + ROUND_UP(unchunked_file_size + tensor_size, this->world_chunk_alignment) > this->buffer_size) {
                finish_chunk(unchunked_file_size);
                reset_chunk_buffer_offset(tensor_id);
                // place the tensor at the beginning of the buffer
            }
            size_t _tensor_device_buffer_offset = tensor_device_buffer_offset(tensor_file_offset);
            unchunked_file_size += tensor_size;
            while (unchunked_file_size > this->world_chunk_size) {
                finish_chunk(this->world_chunk_size);
            }
            chunk_id_t current_chunk_id = (chunk_id_t)this->chunks.size();
            this->tensors[tensor_id] = TensorMetadate{tensor_size, file_index, tensor_file_offset, _tensor_device_buffer_offset, current_chunk_id, 0};

            tensor_id ++;
        }
        finish_chunk(unchunked_file_size);
        reset_chunk_buffer_offset(tensor_id);// set prefetch_chunk_id of the second-to-last layer tensors to the last chunk
        reset_chunk_buffer_offset(tensor_id);// set prefetch_chunk_id of the last layer tensors to the last chunk

        // if(this->rank == 0) {
        //     size_t min_alignment = this->tensors[0].file_offset & -this->tensors[0].file_offset;
        //     for(size_t i = 0; i < this->tensors.size(); i++) {
        //         min_alignment = std::min(min_alignment, this->tensors[i].file_offset & -this->tensors[i].file_offset);
        //     }
        //     fprintf(stderr, "min_alignment = %zu\n", min_alignment);
        //     // for(size_t i = 0; i < this->tensors.size(); i++) {
        //     //     fprintf(stderr, "tensor %zu: size = %zu, file_index = %zu, file_offset = %zu, buffer_offset = %zu, last_chunk_id = %zu, prefetch_chunk_id = %zu\n", 
        //     //         i, this->tensors[i].size, this->tensors[i].file_index, this->tensors[i].file_offset, this->tensors[i].device_buffer_offset, this->tensors[i].last_chunk_id, this->tensors[i].prefetch_chunk_id);
        //     // }
        //     // for(size_t i = 0; i < this->chunks.size(); i++) {
        //     //     fprintf(stderr, "chunk %zu: size = %zu, file_index = %zu, file_offset = %zu, buffer_offset = %zu, file_page = %zu-%zu, buffer_page = %zu-%zu\n", 
        //     //         i, this->chunks[i].size, this->chunks[i].file_index, this->chunks[i].file_offset, this->chunks[i].device_buffer_offset,
        //     //         this->chunks[i].file_offset / PAGE_SIZE, (this->chunks[i].file_offset + this->chunks[i].size - 1) / PAGE_SIZE,
        //     //         this->chunks[i].device_buffer_offset / PAGE_SIZE, (this->chunks[i].device_buffer_offset + this->chunks[i].size - 1) / PAGE_SIZE
        //     //     );
        //     // }
        // }
    }

    void open(OpenArgs args) {
        this->file_info.resize(args.filenames.size());
        for(size_t i = 0; i < args.filenames.size(); i++) {
            this->file_info[i].filename = args.filenames[i];
        }
        this->device_idx = args.device_idx;
        this->group_communicator = args.group_communicator;
        this->rank = args.rank;
        this->world_size = args.world_size;
        this->buffer_size = args.buffer_size;
        this->thread_chunk_size = args.chunk_size;
        this->num_threads = args.num_threads;
        this->io_depth = args.io_depth;

        if (this->world_size > 1 && this->group_communicator == NULL) {
            print_and_throw(std::runtime_error("Internal error: A communicatior should be provided if world_size > 1"));
        }
        if (this->world_size == 1 && this->group_communicator != NULL) {
            print_and_throw(std::runtime_error("Internal error: A communicatior should not be provided if world_size == 1"));
        }

        auto t0 = std::chrono::high_resolution_clock::now();
        cudaSetDevice(this->device_idx);
        auto t1 = std::chrono::high_resolution_clock::now();

        // this->init_comm();
        auto t2 = std::chrono::high_resolution_clock::now();
        this->open_file();// NOTE: this should be called before init_buffer/init_threads/
        auto t3 = std::chrono::high_resolution_clock::now();
        this->init_buffer();
        auto t4 = std::chrono::high_resolution_clock::now();
        this->init_threads();
        auto t5 = std::chrono::high_resolution_clock::now();
        this->compute_layout(args.tensor_offsets);
        auto t6 = std::chrono::high_resolution_clock::now();
        fprintf(stderr, "Config: rank=%d/%d, num_threads=%zu, buffer_size=%zu, chunk_size=%zu, io_depth=%zu, device=%d, communicator=%p\n", 
            this->rank, this->world_size, this->num_threads, this->buffer_size, this->thread_chunk_size, this->io_depth, this->device_idx, (void*)(this->group_communicator));
        std::chrono::duration<double> d1 = t1 - t0;
        std::chrono::duration<double> d2 = t2 - t1;
        std::chrono::duration<double> d3 = t3 - t2;
        std::chrono::duration<double> d4 = t4 - t3;
        std::chrono::duration<double> d5 = t5 - t4;
        std::chrono::duration<double> d6 = t6 - t5;
        fprintf(stderr, "Open time: device=%f, comm=%f, file=%f, buffer=%f, threads=%f, layout=%f\n", d1.count(), d2.count(), d3.count(), d4.count(), d5.count(), d6.count());
    }

    void close(CloseArgs args) {
        auto t0 = std::chrono::high_resolution_clock::now();
        this->destroy_threads();
        auto t1 = std::chrono::high_resolution_clock::now();
        this->destroy_buffer();
        auto t2 = std::chrono::high_resolution_clock::now();
        this->close_file();
        auto t3 = std::chrono::high_resolution_clock::now();
        // this->destroy_comm();
        auto t4 = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> d1 = t1 - t0;
        std::chrono::duration<double> d2 = t2 - t1;
        std::chrono::duration<double> d3 = t3 - t2;
        std::chrono::duration<double> d4 = t4 - t3;
        fprintf(stderr, "Close time: threads=%f, buffer=%f, file=%f, comm=%f\n", d1.count(), d2.count(), d3.count(), d4.count());
        this->stop = true;
    }

    void post_read_chunk() {
        this->chunk_reading++;
        chunk_id_t chunk_id = this->chunk_reading;
        Chunk &chunk = this->chunks[chunk_id];
        // fprintf(stderr, "post_read_chunk: chunk_reading = %zd, chunk_read = %zd, file_offset = %zu, buffer_offset = %zu\n", this->chunk_reading, this->chunk_read, chunk.file_offset, chunk.device_buffer_offset);

        // only the right most chunk may not be aligned and need padding
        size_t padded_world_chunk_size = ROUND_UP(chunk.size, this->world_chunk_alignment);
        size_t padded_rank_size = padded_world_chunk_size / this->world_size;
        size_t padded_thread_size = padded_rank_size / this->num_threads;
        size_t rank_offset = padded_rank_size * this->rank;
        size_t rank_size = std::min((size_t)std::max((ssize_t)(chunk.size - rank_offset), (ssize_t)0), padded_rank_size);
        size_t window_idx = chunk_id % this->io_depth;
        size_t window_offset = window_idx * this->rank_chunk_size;
        void *rank_dst = (char*)this->device_buffer + chunk.device_buffer_offset + rank_offset;
        void *all_dst = (char*)this->device_buffer + chunk.device_buffer_offset;
        
        cudaEvent_t event = this->cuda_events[window_idx];// NOTE: cudaEvent_t is a pointer type

        // std::future<void> completion_future;
        int completion_req_id = -1;
        AsyncExecutor *completion_thread = nullptr;

        chunk_id_t wait_chunk_id = max(chunk_id - (chunk_id_t)MAX_PREFETCH_CHUNKS, chunk_id - (chunk_id_t)this->io_depth);
        if(wait_chunk_id >= 0) {
            // wait for the AsyncExecutor to be available to post tasks
            // wait for existing usage of host_buffer
            this->wait_read_chunk(wait_chunk_id);
        }

        // NOTE: We use AsyncExecutor + std::future instead of faster "SPSCQueue" 
        //       since these operations are async and not on critical path of get_tensor_ptr()
        FileInfo &f = this->file_info[chunk.file_index];
        if (f.in_memory) {
            if (!this->use_internal_memory_register) {
                // vector<cudaEvent_t> events(this->num_threads);
                // vector<std::future<void>> futures(this->num_threads);
                vector<int> req_ids(this->num_threads);
                size_t worker_cnt = 0;
                for(size_t i = 0; i < this->num_threads; i++) {
                    size_t thread_offset = padded_thread_size * i;
                    size_t thread_size = std::min((size_t)std::max((ssize_t)(chunk.size - rank_offset - thread_offset), (ssize_t)0), padded_thread_size);
                    if(padded_thread_size > this->thread_chunk_size) {
                        print_and_throw(std::runtime_error("Internal error: padded_thread_size > chunk_size."));
                    }

                    if(thread_size == 0) continue;
                    // CUDA_CHECK(cudaEventCreateWithFlags(&events[i], cudaEventDisableTiming));

                    // void *dst = (char*)this->device_buffer + chunk.buffer_offset + rank_offset + thread_offset;
                    void *thread_src = (char*)f.mapped_memory + chunk.file_offset + rank_offset + thread_offset;
                    // fprintf(stderr, "src = %p, mmaped_memory = %p\n", src, this->file_info[chunk.file_index].mapped_memory);
                    void *thread_mid = (char*)this->host_buffer + window_offset + thread_offset;// window_idx * this->thread_chunk_size;
                    auto memcpy_func = [=]() {
                        // fprintf(stderr, "memcpy_func %zd-%zu: thread_src = %p, thread_mid = %p, thread_size = %zu\n", chunk_id, i, thread_src, thread_mid, thread_size);
                        memcpy(thread_mid, thread_src, thread_size);
                    };
                    // futures[i] = post_with_future(*(this->worker_threads[i]), std::move(memcpy_func));
                    req_ids[i] = this->worker_threads[i]->post(std::move(memcpy_func));
                    worker_cnt++;
                }

                void *rank_mid = (char*)this->host_buffer + window_offset;
                auto cuda_func = [=]() {
                    // fprintf(stderr, "cudamemcpy_func %zd\n", chunk_id);
                    for(size_t i = 0; i < worker_cnt; i++) {
                        this->worker_threads[i]->pop(req_ids[i]);
                    }
                    CUDA_CHECK(cudaMemcpyAsync(rank_dst, rank_mid, rank_size, cudaMemcpyHostToDevice, this->cuda_stream));// 240GB/s for 8 GPUs
                    CUDA_CHECK(cudaEventRecord(event, this->cuda_stream));
                    if(this->world_size > 1) {
                        CUDA_CHECK(cudaStreamWaitEvent(this->nccl_stream, event));
                        // ncclResult_t ncclAllGather(const void* sendbuff, void* recvbuff, size_t sendcount, ncclDataType_t datatype, ncclComm_t comm, cudaStream_t stream)
                        // In-place operation will happen if sendbuff == recvbuff + rank * sendcount.
                        NCCL_CHECK(ncclAllGather(rank_dst, all_dst, padded_rank_size, ncclUint8, this->group_communicator, this->nccl_stream));// 320GB/s for 8 GPUs
                        CUDA_CHECK(cudaEventRecord(event, this->nccl_stream));
                    }
                };
                int cuda_req_id = this->cuda_thread->post(std::move(cuda_func));

                auto wait_func = [=]() mutable {
                    // fprintf(stderr, "wait %zd\n", chunk_id);
                    this->cuda_thread->pop(cuda_req_id);
                    CUDA_CHECK(cudaEventSynchronize(event));
                };
                // completion_future = post_with_future(*(this->wait_thread), std::move(wait_func));
                completion_req_id = this->wait_thread->post(std::move(wait_func));
                completion_thread = this->wait_thread.get();
            }
            else if (this->use_internal_memory_register) {
                void *rank_src = (char*)f.mapped_memory + chunk.file_offset + rank_offset;
                CUDA_CHECK(cudaMemcpyAsync(rank_dst, rank_src, rank_size, cudaMemcpyHostToDevice, this->cuda_stream));// use default stream 0
                CUDA_CHECK(cudaEventRecord(event, this->cuda_stream));
                if(this->world_size > 1) {
                    CUDA_CHECK(cudaStreamWaitEvent(this->nccl_stream, event));
                    NCCL_CHECK(ncclAllGather(rank_dst, all_dst, padded_rank_size, ncclUint8, this->group_communicator, this->nccl_stream));
                    CUDA_CHECK(cudaEventRecord(event, this->nccl_stream));
                }
                
                auto wait_func = [=]() {
                    CUDA_CHECK(cudaEventSynchronize(event));
                };
                // completion_future = post_with_future(*(this->wait_thread), std::move(wait_func));
                completion_req_id = this->wait_thread->post(std::move(wait_func));
                completion_thread = this->wait_thread.get();
            }
        }
        else {
            // std::vector<std::future<ssize_t>> futures(this->num_threads);
            if(this->use_cufile) {
                vector<int> req_ids(this->num_threads);
                std::vector<ssize_t> expect_return(this->num_threads);
                size_t worker_cnt = 0;
                for(size_t i = 0; i < this->num_threads; i++) {
                    size_t thread_offset = padded_thread_size * i;
                    size_t thread_size = std::min((size_t)std::max((ssize_t)(chunk.size - rank_offset - thread_offset), (ssize_t)0), padded_thread_size);

                    if(thread_size == 0) continue;
                    
                    auto read_weight = [=]() -> ssize_t {
                        // fprintf(stderr, "read_weight: file_offset = %zu, buffer_offset = %zu, thread_size = %zu\n", chunk.file_offset + rank_offset + thread_offset, chunk.device_buffer_offset + rank_offset + thread_offset, thread_size);
                        ssize_t ret = cuFileRead(f.cufile_handle, this->device_buffer, thread_size, 
                            chunk.file_offset + rank_offset + thread_offset, chunk.device_buffer_offset + rank_offset + thread_offset);
                        if(ret == -1) {
                            perror("cuFileRead");
                            throw std::runtime_error("");
                        }
                        else if(ret < 0) {
                            std::cerr << CUFILE_ERRSTR(-ret) << '\n';
                        }
                        return ret;
                    };
                    // method1
                    // futures[i] = std::async(std::launch::async, read_weight);

                    // method2: 15% faster than method1
                    // futures[i] = post_with_future(*(this->worker_threads[i]), std::move(read_weight));
                    req_ids[i] = this->worker_threads[i]->post(std::move(read_weight));
                    expect_return[i] = thread_size;
                    worker_cnt++;
                }

                auto cuda_func = [=, req_ids=std::move(req_ids), expect_return=std::move(expect_return)]() {
                    for(size_t i = 0; i < worker_cnt; i++) {
                        ssize_t bytes_read;
                        this->worker_threads[i]->pop(req_ids[i], bytes_read);
                        if(bytes_read < 0) {// NOTE: bytes_read == expect_return for non-right-most chunk
                            fprintf(stderr, "chunk_id=%zd, rank=%d, thread_id=%zu, bytes_read=%zd, expect_read=%zd\n", 
                                chunk_id, this->rank, i, bytes_read, expect_return[i]);
                            print_and_throw(std::runtime_error("Internal error: bytes_read(" + std::to_string(bytes_read) + ") != thread_size(" + std::to_string(expect_return[i]) + ")."));
                        }
                    }
                    if(this->world_size > 1) {
                        // ncclResult_t ncclAllGather(const void* sendbuff, void* recvbuff, size_t sendcount, ncclDataType_t datatype, ncclComm_t comm, cudaStream_t stream)
                        // In-place operation will happen if sendbuff == recvbuff + rank * sendcount.
                        NCCL_CHECK(ncclAllGather(rank_dst, all_dst, padded_rank_size, ncclUint8, this->group_communicator, this->nccl_stream));// 320GB/s for 8 GPUs
                        CUDA_CHECK(cudaEventRecord(event, this->nccl_stream));
                    }
                };
                int cuda_req_id = this->cuda_thread->post(std::move(cuda_func));

                auto wait_func = [=]() mutable {
                    this->cuda_thread->pop(cuda_req_id);
                    CUDA_CHECK(cudaEventSynchronize(event));
                };

                // method1
                // completion_future = std::async(std::launch::async, wait_func);

                // method2
                // completion_future = post_with_future(*(this->wait_thread), std::move(wait_func));
                completion_req_id = this->wait_thread->post(std::move(wait_func));
                completion_thread = this->wait_thread.get();
            }
            else {// aio
                size_t submit_cnt = 0;
                for(size_t i = 0; i < this->num_threads; i++) {
                    size_t thread_offset = padded_thread_size * i;
                    size_t thread_size = std::min((size_t)std::max((ssize_t)(chunk.size - rank_offset - thread_offset), (ssize_t)0), padded_thread_size);
                    if(thread_size == 0) continue;
                    struct iocb *iocb = this->aio_iocb_ptrs[window_idx * this->num_threads + i];
                    // NOTE: aio needs to use padded_thread_size
                    io_prep_pread(iocb, f.fd, (char*)this->host_buffer + window_offset + thread_offset, padded_thread_size, chunk.file_offset + rank_offset + thread_offset);
                    iocb->data = (void*)chunk_id;
                    submit_cnt ++;
                }
                // this->chunks[chunk_id].extra_data.aio_unfinished_cnt = 0;
                this->chunks[chunk_id].extra_data.aio_unfinished_cnt = submit_cnt;

                int ret = io_submit(this->aio_ctx, submit_cnt, this->aio_iocb_ptrs.data() + window_idx * this->num_threads);
                if(ret < 0){
                    print_and_throw(std::runtime_error("Failed to submit aio: " + std::string(strerror(-ret))));
                }

                void *rank_mid = (char*)this->host_buffer + window_offset;
                auto cuda_func = [=]() {
                    // disk to host
                    size_t &unfinished_cnt = this->chunks[chunk_id].extra_data.aio_unfinished_cnt;
                    while(unfinished_cnt > 0) {
                        int got = io_getevents(this->aio_ctx, unfinished_cnt, unfinished_cnt, this->aio_events.data(), NULL);
                        // got may < min_nr and >= 0 if interrupted
                        if(got < 0){
                            print_and_throw(std::runtime_error("Failed to get aio events: " + std::string(strerror(-got))));
                        }
                        for(int i = 0; i < got; i++) {
                            if(this->aio_events[i].res < 0) {
                                print_and_throw(std::runtime_error("Failed to get aio events: " + std::string(strerror(-this->aio_events[i].res))));
                            }
                            // NOTE: event_chunk_id can be different from chunk_id
                            chunk_id_t event_chunk_id = (chunk_id_t)this->aio_events[i].data;
                            this->chunks[event_chunk_id].extra_data.aio_unfinished_cnt --;
                        }
                    }
                    
                    // host to device
                    CUDA_CHECK(cudaMemcpyAsync(rank_dst, rank_mid, rank_size, cudaMemcpyHostToDevice, this->cuda_stream));// 240GB/s for 8 GPUs
                    CUDA_CHECK(cudaEventRecord(event, this->cuda_stream));
                    if(this->world_size > 1) {
                        CUDA_CHECK(cudaStreamWaitEvent(this->nccl_stream, event));
                        NCCL_CHECK(ncclAllGather(rank_dst, all_dst, padded_rank_size, ncclUint8, this->group_communicator, this->nccl_stream));// 320GB/s for 8 GPUs
                        CUDA_CHECK(cudaEventRecord(event, this->nccl_stream));
                    }
                };

                int cuda_req_id = this->cuda_thread->post(std::move(cuda_func));

                auto wait_func = [=]() mutable {
                    this->cuda_thread->pop(cuda_req_id);
                    CUDA_CHECK(cudaEventSynchronize(event));
                };

                completion_req_id = this->wait_thread->post(std::move(wait_func));
                completion_thread = this->wait_thread.get();
            }
        }
        // this->chunk_futures.push_back(std::move(completion_future));
        this->chunks[chunk_id].request = ChunkRequest{completion_thread, completion_req_id};
    }

    void poll_read_chunk() {
        chunk_id_t next_chunck_id = this->chunk_read + 1;
        while(next_chunck_id <= this->chunk_reading
            && this->chunks[next_chunck_id].request.executor->test(this->chunks[next_chunck_id].request.wait_handle)) {
            next_chunck_id++;
        }
        this->chunk_read = next_chunck_id - 1;
    }

    void wait_read_chunk(chunk_id_t chunk_id) {
        chunk_id_t next_chunck_id = this->chunk_read + 1;
        if(chunk_id > this->chunk_reading) {
            print_and_throw(std::runtime_error("Internal error: chunk_id out of range."));
        }
        while(next_chunck_id <= chunk_id) {
            this->chunks[next_chunck_id].request.executor->wait(this->chunks[next_chunck_id].request.wait_handle);
            next_chunck_id++;
        }
        this->chunk_read = next_chunck_id - 1;
    }

    void step() {
        this->post_read_chunk();
        this->poll_read_chunk();
    }

    bool can_step() {
        if(this->tensors.size() == 0) { // not opened
            return false;
        }
        if(this->current_tensor_index >= this->tensors.size()) {
            print_and_throw(std::runtime_error("Internal error: current_tensor_index (" + std::to_string(this->current_tensor_index) + ") out of range (" + std::to_string(this->tensors.size()) + ")."));
        }
        TensorMetadate& tensor = this->tensors[this->current_tensor_index];
        chunk_id_t prefetch_chunk_id = tensor.prefetch_chunk_id;

        if(this->chunk_reading > prefetch_chunk_id) {
            print_and_throw(std::runtime_error("Internal error: chunk_reading is greater than prefetch_chunk_id."));
        }
        return this->chunk_reading < prefetch_chunk_id;
    }

    void try_step() {
        if(this->can_step()) {
            this->step();
        }
        else {
            this->poll_read_chunk();
        }
    }

    void* get_tensor_ptr(GetTensorArgs args) {
        // for experiment
        // static void *data_ptr = nullptr;
        // if(data_ptr == nullptr) {
        //     CUDA_CHECK(cudaMalloc(&data_ptr, 1024*1024*1024));
        // }
        // return data_ptr;


        auto index = args.tensor_index;

        // if (index == this->tensors.size()) {// going out of bounds by one position is allowed
        //     return NULL;
        // }

        if (index >= this->tensors.size()) {
            print_and_throw(std::runtime_error("Internal error: index out of range."));
        }

        if (index < this->current_tensor_index) {
            print_and_throw(std::runtime_error("Should call get_tensor(key) with keys in order of keys()."));
        }
        if (index > this->current_tensor_index + 1) { 
            fprintf(stderr, "WARNING: index jumped from %zu to %zu\n", this->current_tensor_index, index);
        }
        this->current_tensor_index = index;

        TensorMetadate& tensor = this->tensors[this->current_tensor_index];
        chunk_id_t last_chunk_id = tensor.last_chunk_id;

        while (this->chunk_read < last_chunk_id) {
            if(this->can_step()) {
                this->step();
            }
            else {
                this->wait_read_chunk(last_chunk_id);
            }
        }

        return (char*)this->device_buffer + this->tensors[index].device_buffer_offset;
    }

    // void free_tensor(FreeTensorArgs args) {
    //     // TODO:
    // }

    std::any dispatch(const RPCRequest &m) {
        switch (m.op) {
            case OPEN:
                this->open(std::any_cast<OpenArgs>(m.args));
                return std::any();
            case CLOSE:
                this->close(std::any_cast<CloseArgs>(m.args));
                return std::any();
            case GET_TENSOR_PTR:
                void* ret = this->get_tensor_ptr(std::any_cast<GetTensorArgs>(m.args));
                return std::any(ret);
        }
        print_and_throw(std::runtime_error("Invalid operation"));
    }

    void run() {
        while (!this->stop) {
            if (!this->can_step() || !this->input_queue->empty()) {
                // auto t0 = std::chrono::high_resolution_clock::now();
                RPCRequest m; 
                this->input_queue->pop(m);
                std::any ret = this->dispatch(m);
                this->output_queue->push(RPCResponse{m.id, std::move(ret)});
                // if (m.promise) {
                //     m.promise->set_value(ret);
                //     delete m.promise;
                // }
                // auto t1 = std::chrono::high_resolution_clock::now();
                // std::chrono::duration<double> d1 = t1 - t0;
                // std::cout << "time: " << d1.count() << " seconds" << std::endl;
            }
            this->try_step();
        }
    }
};

void run_loader(unique_ptr<SPSCQueue<RPCRequest>> input_queue, unique_ptr<SPSCQueue<RPCResponse>> output_queue) {
    Loader loader(std::move(input_queue), std::move(output_queue));
    loader.run();
}

// forcely cast the Python object to the C++ holder pointer
template <typename Holder>
Holder *get_holder(pybind11::handle obj) {
    using namespace pybind11::detail;
    // 1) get the PyTypeObject* of the Python object
    //    use C-API macro Py_TYPE:
    PyTypeObject *pytype = reinterpret_cast<PyTypeObject*>(Py_TYPE(obj.ptr()));
    // 2) look up the registration table, get the corresponding pybind11::detail::type_info*
    type_info *ti = get_type_info(pytype);
    if (!ti) {
        throw std::runtime_error("get_holder: Python type not registered with pybind11");
    }
    // 3) cast to instance* and use it to query value_and_holder
    instance *inst = reinterpret_cast<instance *>(obj.ptr());
    // throw_if_missing=true: if not found or holder not constructed, will throw exception
    auto vh = inst->get_value_and_holder(ti, /*throw_if_missing=*/true);
    // 4) get the Holder reference, the holder is placed in vh.vh[1] when placement-new
    Holder &h = vh.holder<Holder>();
    return &h;
}

// Copied and modified from ProcessGroupNCCL.getCommPtr() of PyTorch 2.9 to adapt to PyTorch 2.7
ncclComm_t get_nccl_comm_from_pg(c10d::ProcessGroup* pg) {
    // 1. get CUDA backend
    // auto backend = pg->getBackend(c10::DeviceType::CUDA);// no "d" in "c10"
    auto backend = pg->getBackend(c10d::ProcessGroup::BackendType::NCCL);// this may fail and exit(1)
    // 2. convert to ProcessGroupNCCL
    auto* nccl_pg = dynamic_cast<c10d::ProcessGroupNCCL*>(backend.get());
    TORCH_CHECK(nccl_pg != nullptr, "ProcessGroup backend must be NCCL");
    int64_t comm_addr = nccl_pg->getCommPtr();
    ncclComm_t comm = reinterpret_cast<ncclComm_t>(comm_addr);
    return comm;
}

struct RPCHandle {
    int latest_req_id;
    SPSCQueue<RPCRequest>* input_queue;
    SPSCQueue<RPCResponse>* output_queue;
};

class LoaderManager {
public:
    unordered_map<int, RPCHandle> loader_handle_to_queue;
    int loader_id_iter = 0;
    int device_idx = 0;

    int call(int remote_handle, int opcode, std::any args) {
        // NOTE: promise+future costs ~4us, thus we use spsc_queue to reduce time overhead of result returning
        auto& handle = this->loader_handle_to_queue[remote_handle];
        ++handle.latest_req_id;
        int req_id = handle.latest_req_id;
        handle.input_queue->push(RPCRequest{req_id, opcode, std::move(args)});
        return req_id;
    }

    std::any get_result(int remote_handle, int req_id) {
        // NOTE: assume the requests are returned in order, otherwise we need to use a map to store the requests and responses
        auto& handle = this->loader_handle_to_queue[remote_handle];
        RPCResponse response;
        handle.output_queue->pop(response);
        while (response.id != req_id) {
            handle.output_queue->pop(response);
        }
        return response.result;
    }

    py::bytes create_group_constructor() 
    {
        fprintf(stderr, "create_process_constructor\n");
        ncclUniqueId nccl_uid;
        NCCL_CHECK(ncclGetUniqueId(&nccl_uid));
        string blob(reinterpret_cast<char*>(&nccl_uid), sizeof(nccl_uid));
        return py::bytes(blob);
    }

    void* create_group_communicator(py::bytes group_constructor, int world_size, int rank) {
        string group_constructor_str = group_constructor.cast<string>();
        ncclUniqueId nccl_uid;
        std::memcpy(&nccl_uid, group_constructor_str.data(), sizeof(nccl_uid));
        ncclComm_t comm;
        NCCL_CHECK(ncclCommInitRank(&comm, world_size, nccl_uid, rank));
        fprintf(stderr, "create_group_communicator: world_size = %d, rank = %d, comm = %p\n", world_size, rank, comm);
        return (void*)comm; // ncclComm_t is a pointer to struct ncclComm
    }

    // async open
    int open(const vector<string> &filenames, 
            int device_idx, 
            py::object process_group,
            size_t buffer_size, 
            size_t chunk_size, 
            size_t num_threads,
            size_t io_depth,
            const vector<pair<size_t, size_t>>& tensor_offsets)
    {
        // fprintf(stderr, "open\n");
        this->device_idx = device_idx;

        int loader_handle = loader_id_iter++;
        SPSCQueue<RPCRequest> *input_queue = new SPSCQueue<RPCRequest>(); // deleted by Loader
        SPSCQueue<RPCResponse> *output_queue = new SPSCQueue<RPCResponse>(); // deleted by Loader
        this->loader_handle_to_queue[loader_handle] = RPCHandle{0, input_queue, output_queue};
        std::thread loader_thread(run_loader, std::unique_ptr<SPSCQueue<RPCRequest>>(input_queue), std::unique_ptr<SPSCQueue<RPCResponse>>(output_queue));
        loader_thread.detach();

        // std::optional<ncclUniqueId> nccl_uid_opt = std::nullopt;
        // if (group_handle.has_value()) {
        //     string group_handle_str = group_handle.value().cast<string>();
        //     ncclUniqueId nccl_uid;
        //     std::memcpy(&nccl_uid, group_handle_str.data(), sizeof(nccl_uid));
        //     nccl_uid_opt = nccl_uid;
        // }
        ncclComm_t nccl_group_communicator = nullptr;
        int rank = 0;
        int world_size = 1;
        if(!process_group.is_none()) {
            // This returns NULL since pybind11 cannot find the mapping
            // c10::intrusive_ptr<c10d::ProcessGroup> process_group_ptr = process_group.cast<c10::intrusive_ptr<c10d::ProcessGroup>>();
            // c10d::ProcessGroup* process_group_ptr = unwrap_pybind11_ptr<c10d::ProcessGroup>(process_group);
            c10::intrusive_ptr<c10d::ProcessGroup> process_group_ptr = *(get_holder<c10::intrusive_ptr<c10d::ProcessGroup>>(process_group));
            nccl_group_communicator = get_nccl_comm_from_pg(process_group_ptr.get());
            rank = process_group_ptr->getRank();
            world_size = process_group_ptr->getSize();
        }

        // async
        int req_id = this->call(loader_handle, OPEN, std::make_any<OpenArgs>(filenames, device_idx, nccl_group_communicator, rank, world_size, buffer_size, chunk_size, num_threads, io_depth, tensor_offsets));
        this->get_result(loader_handle, req_id);
        return loader_handle;
    }

    // async close
    void close(int loader_handle) {
        // fprintf(stderr, "close\n");
        // async
        int req_id = this->call(loader_handle, CLOSE, std::make_any<CloseArgs>());
        this->get_result(loader_handle, req_id);
        // SPSCQueue is freed by Loader
        this->loader_handle_to_queue.erase(loader_handle);
    }

    // void free_tensor(int loader_handle, int tensor_index) {
    //     this->loader_handle_to_queue[loader_handle]->push(RPCRequest{FREE_TENSOR, std::make_any<FreeTensorArgs>(tensor_index), NULL});
    // }

    // sync get_tensor
    at::Tensor get_tensor(int loader_handle, int tensor_index, vector<int64_t> shape, c10::ScalarType dtype) {
        // inline Tensor from_blob(
        //     void* data,
        //     IntArrayRef sizes,
        //     IntArrayRef strides,
        //     int64_t storage_offset,
        //     const std::function<void(void*)>& deleter,
        //     const TensorOptions& options = {},
        //     const std::optional<Device> target_device = std::nullopt);
        
        // inline Tensor from_blob(
        //     void* data,
        //     IntArrayRef sizes,
        //     const TensorOptions& options = {});

        // device can be set by at::TensorOptions().device(at::Device(DeviceType::CUDA, static_cast<DeviceIndex>(device)))

        // NOTE: IntArrayRef requires element type of int64_t

        // auto t0 = std::chrono::high_resolution_clock::now();


        int req_id = this->call(loader_handle, GET_TENSOR_PTR, std::make_any<GetTensorArgs>(tensor_index));
        std::any data_ptr_any = this->get_result(loader_handle, req_id);
        void *data_ptr = std::any_cast<void*>(data_ptr_any);


        // auto t1 = std::chrono::high_resolution_clock::now();
        // std::chrono::duration<double> d1 = t1 - t0;
        // std::cout << "time: " << d1.count() << " seconds" << std::endl;

        // for experiment

        // NOTE: this experimental branch provides ~200GB/s performance on a single GPU
        // auto t0 = std::chrono::high_resolution_clock::now();
        // size_t size = 1;
        // for(auto s: shape) {
        //     size *= s;
        // }
        // size *= at::elementSize(dtype);
        // static void *data_ptr = NULL;
        // if(data_ptr == NULL) {
        //     CUDA_CHECK(cudaMalloc(&data_ptr, 1024*1024*1024));
        // }

        // auto t1 = std::chrono::high_resolution_clock::now();
        auto ret = at::from_blob(data_ptr, shape, at::TensorOptions().dtype(dtype).device(at::Device(at::DeviceType::CUDA, static_cast<at::DeviceIndex>(this->device_idx)))); // target_device can be inferred
        // auto t2 = std::chrono::high_resolution_clock::now();
        // std::chrono::duration<double> d1 = t1 - t0;
        // std::chrono::duration<double> d2 = t2 - t1;
        // std::cout << "time: " << d1.count() << " seconds, " << d2.count() << " seconds" << std::endl;
        return ret;
    }

};


unique_ptr<LoaderManager> manager;

void init() {
    if(_use_cufile()) {
        cufile_context_initializer->initialize();
    }
    if(!manager) {
        manager = std::make_unique<LoaderManager>();
    }
}

py::bytes create_group_constructor() 
{
    if(!manager) {
        manager = std::make_unique<LoaderManager>();
    }
    return manager->create_group_constructor();
}

void* create_group_communicator(py::bytes group_constructor, int world_size, int rank) {
    if(!manager) {
        manager = std::make_unique<LoaderManager>();
    }
    return manager->create_group_communicator(group_constructor, world_size, rank);
}

int open(const vector<string> &filenames, 
        int device_idx, 
        py::object process_group,
        size_t buffer_size, 
        size_t chunk_size,
        size_t num_threads,
        size_t io_depth,
        const vector<pair<size_t, size_t>>& tensor_offsets)
{
    if(!manager) {
        manager = std::make_unique<LoaderManager>();
    }
    return manager->open(filenames, device_idx, process_group, buffer_size, chunk_size, num_threads, io_depth, tensor_offsets);
}

void close(int loader_handle) {
    if(!manager) {
        print_and_throw(std::runtime_error("Internal error: manager is not initialized"));
    }
    manager->close(loader_handle);
}

at::Tensor get_tensor(int loader_handle, int tensor_index, vector<int64_t> shape, c10::ScalarType dtype) {
    if(!manager) {
        print_and_throw(std::runtime_error("Internal error: manager is not initialized"));
    }
    return manager->get_tensor(loader_handle, tensor_index, shape, dtype);
}

// Clean global cuda-related objects before python exit and cudaDeviceReset() is called 
// to avoid cuda error when the program exits. This is registered by atexit.register() in safe_open_impl.py.
void cleanup() {
    instanttensor::cufile_context_initializer.reset();
    instanttensor::host_buffer_cache.reset();
    instanttensor::munmaper.reset();
}

}



// pybind11 interface
PYBIND11_MODULE(_C, m) {
    m.doc() = "InstantTensor C++ extension module";

    // unnecessary binding
    // py::module dist = py::module::import("torch.distributed.distributed_c10d");
    // py::class_<
    //     c10d::ProcessGroup,
    //     c10::intrusive_ptr<c10d::ProcessGroup>
    // >(dist.attr("ProcessGroup")); // tell pybind11 to map ProcessGroup (python) to c10d::ProcessGroup (C++) with holder c10::intrusive_ptr
    
    // py::module dist = py::module::import("torch._C._distributed_c10d");
    // py::class_<c10d::ProcessGroup, c10::intrusive_ptr<c10d::ProcessGroup>>(
    //     dist.attr("ProcessGroup")
    // );

    m.def("create_group_constructor", &instanttensor::create_group_constructor,
          "Create a constructor of group communicator for distributed loading");

    m.def("create_group_communicator", &instanttensor::create_group_communicator,
          "Create a group communicator from a constructor for distributed loading",
          pybind11::arg("group_constructor"),
          pybind11::arg("world_size"),
          pybind11::arg("rank"));

    m.def("init", &instanttensor::init,
          "Initialize the cufile driver. This will be called lazily in open() if cufile is available for the specified file, "
          "but can be explicitly called to control the timing of initialization.");

    m.def("open", &instanttensor::open,
          "Open a safetensors file for loading",
          pybind11::arg("filenames"),
          pybind11::arg("device_idx"),
          pybind11::arg("process_group"),
          pybind11::arg("buffer_size"),
          pybind11::arg("chunk_size"),
          pybind11::arg("num_threads"),
          pybind11::arg("io_depth"),
          pybind11::arg("tensor_offsets"));
    
    m.def("close", &instanttensor::close,
          "Close the loader handle and cleanup resources",
          pybind11::arg("loader_handle"));
    
    m.def("get_tensor", &instanttensor::get_tensor,
          "Get a tensor by index from the opened file",
          pybind11::arg("loader_handle"),
          pybind11::arg("tensor_index"),
          pybind11::arg("shape"),
          pybind11::arg("dtype"));

    m.def("file_in_memory", &instanttensor::file_in_memory,
          "Check if the file is in memory",
          pybind11::arg("filenames"));

    m.def("cleanup", &instanttensor::cleanup,
          "Clean up global cuda-related objects before python exit and cudaDeviceReset() is called");
}