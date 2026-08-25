#include <instant_tensor/loader.hpp>

namespace instanttensor {

bool Loader::cufile_available(){// best-effort check
    return cufile_binding::init();
}

void Loader::open_file_cufile(FileInfo &f) {
    f.fd = ::open(f.filename.c_str(), O_RDONLY | O_DIRECT);
    if (f.fd < 0) {
        throw std::runtime_error("Failed to open file: " + f.filename);
    }
    struct stat st;
    if (fstat(f.fd, &st) < 0) { throw std::runtime_error("Failed to fstat file: " + f.filename); }
    f.size = st.st_size;

    this->need_worker_threads = true;
    this->need_cuda_thread = true;

    cufile_context_initializer->initialize();

    CUfileDescr_t descr = {};
    descr.type = CU_FILE_HANDLE_TYPE_OPAQUE_FD;
    descr.handle.fd = f.fd;
    CUFILE_CHECK(cuFileHandleRegister(&f.cufile_handle, &descr));
    // Cannot close fd here since cuFileHandleRegister requires fd to be different in value (int),
    // and closing here cause the OS to reuse fd value.
}

void Loader::close_file_cufile(FileInfo &f) {
    cuFileHandleDeregister(f.cufile_handle);
    ::close(f.fd);
}

void Loader::register_device_buffer_cufile() {
    CUFILE_CHECK(cuFileBufRegister(this->device_buffer, this->buffer_size, 0));
}

void Loader::deregister_device_buffer_cufile() {
    CUFILE_CHECK(cuFileBufDeregister(this->device_buffer));
}

ChunkRequest Loader::post_read_chunk_cufile(const ChunkIOParams &p) {
    chunk_id_t chunk_id = p.chunk_id;
    int read_req_id = EXECUTOR_STOP_REQUEST_ID;
    ssize_t expected_bytes = p.rank_size;
    if(p.rank_size > 0) {
        CUfileHandle_t cufile_handle = p.file.cufile_handle;
        size_t file_offset = p.chunk.file_offset + p.rank_offset;
        size_t buf_offset = p.chunk.device_buffer_offset + p.rank_offset;
        size_t rank_size = p.rank_size;
        auto read_chunk = [=]() -> ssize_t {
            ssize_t ret = cuFileRead(cufile_handle, this->device_buffer, rank_size,
                file_offset, buf_offset);
            if(ret == -1) {
                perror("cuFileRead");
                throw std::runtime_error("");
            }
            else if(ret < 0) {
                std::cerr << CUFILE_ERRSTR(-ret) << '\n';
            }
            return ret;
        };
        read_req_id = this->next_executor_request_id();
        this->worker_threads->submit(read_req_id, std::move(read_chunk));
    }

    void *rank_dst = p.rank_dst;
    void *all_dst = p.all_dst;
    size_t padded_rank_size = p.padded_rank_size;
    cudaEvent_t event = p.event;
    int rank = this->rank;
    auto cuda_func = [=]() {
        if(read_req_id != EXECUTOR_STOP_REQUEST_ID) {
            ssize_t bytes_read;
            this->worker_threads->reap(read_req_id, bytes_read);
            if(bytes_read != expected_bytes) {// bytes_read < 0 on error
                fprintf(stderr, "chunk_id=%zd, rank=%d, bytes_read=%zd, expect_read=%zd\n",
                    chunk_id, rank, bytes_read, expected_bytes);
                print_and_throw(std::runtime_error("Internal error: bytes_read(" + std::to_string(bytes_read) + ") != rank_size(" + std::to_string(expected_bytes) + ")."));
            }
        }
        if(this->world_size > 1) {
            NCCL_CHECK(ncclAllGather(rank_dst, all_dst, padded_rank_size, ncclInt8, this->group_communicator, this->nccl_stream));// 320GB/s for 8 GPUs
            CUDA_CHECK(cudaEventRecord(event, this->nccl_stream));
        }
    };
    int cuda_req_id = this->next_executor_request_id();
    this->cuda_thread->submit(cuda_req_id, std::move(cuda_func));

    auto wait_func = [=]() mutable {
        this->cuda_thread->reap(cuda_req_id);
        CUDA_CHECK(cudaEventSynchronize(event));
    };

    int completion_req_id = this->next_executor_request_id();
    this->wait_thread->submit(completion_req_id, std::move(wait_func));
    SingleThreadTaskExecutor *completion_thread = this->wait_thread.get();

    return ChunkRequest{completion_thread, completion_req_id};
}

} // namespace instanttensor
