#include <instant_tensor/loader.hpp>
#include <cerrno>
#include <cstring>
#include <fcntl.h>
#include <mutex>
#include <sys/mman.h>
#include <sys/stat.h>

namespace instanttensor {
namespace {

class Munmaper {
    std::mutex mutex;
    vector<pair<void*, size_t>> mapped_memory;
public:
    ~Munmaper() noexcept(false) {
        std::lock_guard<std::mutex> lock(this->mutex);
        for(auto& ptr : this->mapped_memory) {
            int ret = munmap(ptr.first, ptr.second);
            if (ret != 0) {
                throw std::runtime_error("Failed to munmap: " + std::string(strerror(errno)));
            }
        }
    }
    void add(void* ptr, size_t size) {
        std::lock_guard<std::mutex> lock(this->mutex);
        this->mapped_memory.emplace_back(std::make_pair(ptr, size));
    }
};

auto munmaper = std::make_unique<Munmaper>();

} // namespace

void cleanup_mmap() {
    munmaper.reset();
}

class MmapBackend final : public WorkerIOBackend {
public:
    using WorkerIOBackend::WorkerIOBackend;
    vector<void*> file_mappings;
    bool needs_host_buffer() const override { return true; }
    void open_files() override;
    void close_files() override;
    IORequest post_read_chunk(const ChunkIOParams &p) override;
};

void MmapBackend::open_files() {
    this->file_mappings.resize(this->loader.file_info.size());
    for (size_t i = 0; i < this->loader.file_info.size(); ++i) {
        FileInfo &f = this->loader.file_info[i];
        f.fd = ::open(f.filename.c_str(), O_RDONLY);
        if (f.fd < 0) {
            throw std::runtime_error("Failed to open file: " + f.filename);
        }
        struct stat st;
        if (fstat(f.fd, &st) < 0) { throw std::runtime_error("Failed to fstat file: " + f.filename); }
        f.size = st.st_size;

        this->file_mappings[i] = mmap(NULL, f.size, PROT_READ, MAP_SHARED, f.fd, 0);
        if (this->file_mappings[i] == MAP_FAILED) {
            throw std::runtime_error("Failed to mmap file: " + f.filename);
        }
    }
}

void MmapBackend::close_files() {
    for (size_t i = 0; i < this->file_mappings.size(); ++i) {
        const FileInfo &f = this->loader.file_info[i];
        // NOTE: Since munmap is very slow, we defer it to exit time
        munmaper->add(this->file_mappings[i], f.size);
        ::close(f.fd);
    }
}

IORequest MmapBackend::post_read_chunk(const ChunkIOParams &p) {
    void *rank_src = (char*)this->file_mappings[p.chunk.file_index] + p.chunk.io_state.rank_file_offset;
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
