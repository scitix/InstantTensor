#include <instant_tensor/loader.hpp>

namespace instanttensor {

void Loader::open_file_cufile(FileInfo &f) {
    f.fd = ::open(f.filename.c_str(), O_RDONLY | O_DIRECT);
    if (f.fd < 0) {
        throw std::runtime_error("Failed to open file: " + f.filename);
    }
    struct stat st;
    if (fstat(f.fd, &st) < 0) { throw std::runtime_error("Failed to fstat file: " + f.filename); }
    f.size = st.st_size;

    this->need_cufile = true;
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

ChunkRequest Loader::post_read_chunk_cufile(const ChunkIOParams &p) {
    chunk_id_t chunk_id = p.chunk_id;
    vector<int> req_ids(this->num_threads);
    std::vector<ssize_t> expect_return(this->num_threads);
    size_t worker_cnt = 0;
    for(size_t i = 0; i < this->num_threads; i++) {
        size_t thread_offset = p.padded_thread_size * i;
        size_t thread_size = std::min((size_t)std::max((ssize_t)(p.chunk.size - p.rank_offset - thread_offset), (ssize_t)0), p.padded_thread_size);

        if(thread_size == 0) continue;

        CUfileHandle_t cufile_handle = p.file.cufile_handle;
        size_t file_offset = p.chunk.file_offset + p.rank_offset + thread_offset;
        size_t buf_offset = p.chunk.device_buffer_offset + p.rank_offset + thread_offset;
        auto read_weight = [=]() -> ssize_t {
            ssize_t ret = cuFileRead(cufile_handle, this->device_buffer, thread_size,
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
        req_ids[i] = this->worker_threads[i]->post(std::move(read_weight));
        expect_return[i] = thread_size;
        worker_cnt++;
    }

    void *rank_dst = p.rank_dst;
    void *all_dst = p.all_dst;
    size_t padded_rank_size = p.padded_rank_size;
    cudaEvent_t event = p.event;
    int rank = this->rank;
    auto cuda_func = [=, req_ids=std::move(req_ids), expect_return=std::move(expect_return)]() {
        for(size_t i = 0; i < worker_cnt; i++) {
            ssize_t bytes_read;
            this->worker_threads[i]->pop(req_ids[i], bytes_read);
            if(bytes_read != expect_return[i]) {// bytes_read < 0 on error
                fprintf(stderr, "chunk_id=%zd, rank=%d, thread_id=%zu, bytes_read=%zd, expect_read=%zd\n",
                    chunk_id, rank, i, bytes_read, expect_return[i]);
                print_and_throw(std::runtime_error("Internal error: bytes_read(" + std::to_string(bytes_read) + ") != thread_size(" + std::to_string(expect_return[i]) + ")."));
            }
        }
        if(this->world_size > 1) {
            NCCL_CHECK(ncclAllGather(rank_dst, all_dst, padded_rank_size, ncclInt8, this->group_communicator, this->nccl_stream));// 320GB/s for 8 GPUs
            CUDA_CHECK(cudaEventRecord(event, this->nccl_stream));
        }
    };
    int cuda_req_id = this->cuda_thread->post(std::move(cuda_func));

    auto wait_func = [=]() mutable {
        this->cuda_thread->pop(cuda_req_id);
        CUDA_CHECK(cudaEventSynchronize(event));
    };

    int completion_req_id = this->wait_thread->post(std::move(wait_func));
    AsyncExecutor *completion_thread = this->wait_thread.get();

    return ChunkRequest{completion_thread, completion_req_id};
}

} // namespace instanttensor
