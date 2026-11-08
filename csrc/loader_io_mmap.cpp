#include <instant_tensor/loader.hpp>

namespace instanttensor {

class MmapBackend final : public WorkerIOBackend {
public:
    MmapBackend(Loader &loader, bool register_memory)
        : WorkerIOBackend(loader), use_internal_memory_register(register_memory) {}
    bool needs_host_buffer() const override { return !this->use_internal_memory_register; }
    void init_threads() override {
        if (!this->use_internal_memory_register) this->WorkerIOBackend::init_threads();
    }
    void open_files() override;
    void close_files() override;
    IORequest post_read_chunk(const ChunkIOParams &p) override;

    bool use_internal_memory_register;
};

void MmapBackend::open_files() {
    for (auto &f : this->loader.file_info) {
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
            CUDA_CHECK(cudaHostRegister(f.mapped_memory, f.size, this->loader.cuda_host_register_flags | cudaHostRegisterReadOnly));
        }
    }
}

void MmapBackend::close_files() {
    for (auto &f : this->loader.file_info) {
        if (this->use_internal_memory_register) {
            CUDA_CHECK(cudaHostUnregister(f.mapped_memory));
        }
        // NOTE: Since munmap is very slow, we defer it to exit time
        munmaper->add(f.mapped_memory, f.size);
        ::close(f.fd);
    }
}

IORequest MmapBackend::post_read_chunk(const ChunkIOParams &p) {
    if (!this->use_internal_memory_register) {
        void *rank_src = (char*)p.file.mapped_memory + p.chunk.io_state.rank_file_offset;
        void *rank_mid = (char*)this->loader.host_buffer + p.chunk.io_state.window_offset;
        size_t rank_size = p.chunk.io_state.rank_size;
        int io_req_id = this->loader.next_loader_task_id();
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
        this->loader.io_thread->submit(io_req_id, std::move(io_start));
        return IORequest{this->loader.io_thread.get(), io_req_id, false};
    }
    else {
        void *rank_src = (char*)p.file.mapped_memory + p.chunk.io_state.rank_file_offset;
        void *rank_dst = p.rank_dst;
        size_t rank_size = p.chunk.io_state.rank_size;
        CUDA_CHECK(cudaMemcpyAsync(rank_dst, rank_src, rank_size, cudaMemcpyHostToDevice, this->loader.cuda_stream));// use default stream 0
        int io_req_id = this->loader.next_loader_task_id();
        this->loader.io_thread->submit(io_req_id, []() { return IOSubmitStatus::Completed; });
        return IORequest{this->loader.io_thread.get(), io_req_id, true};
    }
}

std::unique_ptr<IOBackend> make_mmap_backend(Loader &loader) {
    return std::make_unique<MmapBackend>(loader, _env_use_internal_memory_register());
}

} // namespace instanttensor
