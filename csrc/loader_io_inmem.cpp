#include <instant_tensor/loader.hpp>

namespace instanttensor {

void Loader::open_file_inmem(FileInfo &f) {
    f.fd = ::open(f.filename.c_str(), O_RDONLY);
    if (f.fd < 0) {
        throw std::runtime_error("Failed to open file: " + f.filename);
    }
    struct stat st;
    if (fstat(f.fd, &st) < 0) { throw std::runtime_error("Failed to fstat file: " + f.filename); }
    f.size = st.st_size;

    f.mapped_memory = mmap(NULL, f.size, PROT_READ, MAP_SHARED, f.fd, 0);
    if (f.mapped_memory == MAP_FAILED) {
        throw std::runtime_error("Failed to mmap file: " + f.filename);
    }
    if (this->use_internal_memory_register) {
        // NOTE: this requires cudaHostRegisterReadOnly since the file is read-only
        CUDA_CHECK(cudaHostRegister(f.mapped_memory, f.size, this->cuda_host_register_flags | cudaHostRegisterReadOnly));
    }
    else {
        this->need_host_buffer = true;
        this->need_worker_threads = true;
    }
}

void Loader::close_file_inmem(FileInfo &f) {
    if (this->use_internal_memory_register) {
        CUDA_CHECK(cudaHostUnregister(f.mapped_memory));
    }
    // NOTE: Since munmap is very slow, we defer it to exit time
    munmaper->add(f.mapped_memory, f.size);
    ::close(f.fd);
}

IORequest Loader::post_read_chunk_inmem(const ChunkIOParams &p) {
    chunk_id_t chunk_id = p.chunk_id;
    if (!this->use_internal_memory_register) {
        ChunkExtraData &initial_state = this->chunks[chunk_id].extra_data;
        initial_state.pending_worker_request_id = EXECUTOR_STOP_REQUEST_ID;
        void *rank_src = (char*)p.file.mapped_memory + p.chunk.file_offset + p.rank_offset;
        void *rank_mid = (char*)this->host_buffer + p.window_offset;
        size_t rank_size = p.rank_size;
        auto io_start = [=]() {
            ChunkExtraData &state = this->chunks[chunk_id].extra_data;
            if (rank_size > 0) {
                state.pending_worker_request_id = this->next_io_worker_task_id();
                this->worker_threads->submit(state.pending_worker_request_id, [=]() {
                    memcpy(rank_mid, rank_src, rank_size);
                });
            }
        };
        auto io_func = [=]() -> bool {
            ChunkExtraData &state = this->chunks[chunk_id].extra_data;
            if (state.pending_worker_request_id != EXECUTOR_STOP_REQUEST_ID) {
                std::any ignored;
                if (!this->worker_threads->try_reap(state.pending_worker_request_id, ignored)) {
                    return false;
                }
                state.pending_worker_request_id = EXECUTOR_STOP_REQUEST_ID;
            }
            return true;
        };
        int io_req_id = this->next_loader_task_id();
        this->io_thread->submit(io_req_id, IOOperation{std::move(io_start), std::move(io_func)});
        return IORequest{this->io_thread.get(), io_req_id, false};
    }
    else {
        void *rank_src = (char*)p.file.mapped_memory + p.chunk.file_offset + p.rank_offset;
        void *rank_dst = p.rank_dst;
        size_t rank_size = p.rank_size;
        CUDA_CHECK(cudaMemcpyAsync(rank_dst, rank_src, rank_size, cudaMemcpyHostToDevice, this->cuda_stream));// use default stream 0
        auto io_func = []() -> bool { return true; };
        int io_req_id = this->next_loader_task_id();
        this->io_thread->submit(io_req_id, IOOperation{[] {}, std::move(io_func)});
        return IORequest{this->io_thread.get(), io_req_id, true};
    }
}

} // namespace instanttensor
