#include <instant_tensor/loader.hpp>

namespace instanttensor {

class MmapBackend final : public WorkerIOBackend {
public:
    using WorkerIOBackend::WorkerIOBackend;
    bool needs_host_buffer() const override { return true; }
    void open_files() override;
    void close_files() override;
    IORequest post_read_chunk(const ChunkIOParams &p) override;
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
    }
}

void MmapBackend::close_files() {
    for (auto &f : this->loader.file_info) {
        // NOTE: Since munmap is very slow, we defer it to exit time
        munmaper->add(f.mapped_memory, f.size);
        ::close(f.fd);
    }
}

IORequest MmapBackend::post_read_chunk(const ChunkIOParams &p) {
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

std::unique_ptr<IOBackend> make_mmap_backend(Loader &loader) {
    return std::make_unique<MmapBackend>(loader);
}

} // namespace instanttensor
