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
    if (!this->use_internal_memory_register) {
        void *rank_src = (char*)p.file.mapped_memory + p.chunk.io_state.rank_file_offset;
        void *rank_mid = (char*)this->host_buffer + p.chunk.io_state.window_offset;
        size_t rank_size = p.chunk.io_state.rank_size;
        int io_req_id = this->next_loader_task_id();
        auto io_start = [=]() {
            if (rank_size > 0) {
                this->worker_threads->submit(io_req_id, [=]() {
                    memcpy(rank_mid, rank_src, rank_size);
                });
            } else {
                return IOSubmitStatus::Completed;
            }
            return IOSubmitStatus::Submitted;
        };
        this->io_thread->submit(io_req_id, std::move(io_start));
        return IORequest{this->io_thread.get(), io_req_id, false};
    }
    else {
        void *rank_src = (char*)p.file.mapped_memory + p.chunk.io_state.rank_file_offset;
        void *rank_dst = p.rank_dst;
        size_t rank_size = p.chunk.io_state.rank_size;
        CUDA_CHECK(cudaMemcpyAsync(rank_dst, rank_src, rank_size, cudaMemcpyHostToDevice, this->cuda_stream));// use default stream 0
        int io_req_id = this->next_loader_task_id();
        this->io_thread->submit(io_req_id, []() { return IOSubmitStatus::Completed; });
        return IORequest{this->io_thread.get(), io_req_id, true};
    }
}

} // namespace instanttensor
