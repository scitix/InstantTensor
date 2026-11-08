#include <instant_tensor/loader.hpp>
#include <cerrno>
#include <cstring>
#include <fcntl.h>
#include <libaio.h>
#include <sys/stat.h>

namespace instanttensor {

class AIOBackend final : public IOBackend {
public:
    using IOBackend::IOBackend;
    ~AIOBackend() override { this->destroy_threads(); }
    bool needs_host_buffer() const override { return true; }
    void open_files() override;
    void close_files() override;
    void destroy_context();
    void init_threads() override;
    void destroy_threads() override;
    void abort() override;
    IORequest post_read_chunk(const ChunkIOParams &p) override;
    void poll(std::vector<IOCompletion> &completed) override;
    IOSubmitStatus submit_read(chunk_id_t id);
    IOSubmitStatus schedule_read(chunk_id_t id);
    void drain_submissions();
    int next_io_worker_task_id();

    bool aio_context_initialized = false;
    io_context_t aio_ctx = {};
    vector<struct iocb> aio_iocbs;
    vector<struct iocb*> aio_iocb_ptrs;
    vector<struct io_event> aio_events;
    std::unordered_map<int, chunk_id_t> aio_last_page_submissions;
    int io_worker_task_id = 0;
    unique_ptr<SingleThreadTaskExecutor> last_page_reader_thread;
private:
    void open_file(FileInfo &file);
};

int AIOBackend::next_io_worker_task_id() {
    int request_id = this->io_worker_task_id;
    this->io_worker_task_id = static_cast<int>((static_cast<unsigned int>(request_id) + 1U) & 0x7fffffffU);
    return request_id;
}

void AIOBackend::init_threads() {
    this->last_page_reader_thread = std::make_unique<SingleThreadTaskExecutor>();
}

void AIOBackend::destroy_threads() {
    if (this->last_page_reader_thread) {
        this->drain_submissions();
        this->last_page_reader_thread->join();
    }
}

void AIOBackend::abort() {
    if (!this->aio_context_initialized) return;
    this->destroy_threads();
    this->destroy_context();
    this->aio_last_page_submissions.clear();
}

void AIOBackend::open_file(FileInfo &f) {
    int open_flags = O_RDONLY;
    if (this->loader.backend == Backend::AIO) {// != AIO_BUFFERED
        open_flags |= O_DIRECT;
    }
    f.fd = ::open(f.filename.c_str(), open_flags);
    if (f.fd < 0) {
        throw std::runtime_error("Failed to open file: " + f.filename);
    }
    struct stat st;
    if (fstat(f.fd, &st) < 0) { throw std::runtime_error("Failed to fstat file: " + f.filename); }
    f.size = st.st_size;

    if (this->loader.backend == Backend::AIO_BUFFERED) {
        posix_fadvise(f.fd, 0, 0, POSIX_FADV_SEQUENTIAL);
    }
}

void AIOBackend::open_files() {
    int ret = io_setup(this->loader.io_depth, &this->aio_ctx);
    if(ret < 0){
        print_and_throw(std::runtime_error("Failed to setup aio: " + std::string(strerror(-ret))));
    }
    this->aio_context_initialized = true;
    this->aio_iocbs.resize(this->loader.io_depth);
    this->aio_iocb_ptrs.resize(this->loader.io_depth);
    for(size_t i = 0; i < this->loader.io_depth; i++) {
        this->aio_iocb_ptrs[i] = &this->aio_iocbs[i];
    }
    this->aio_events.resize(this->loader.io_depth);
    for (auto &f : this->loader.file_info) {
        this->open_file(f);
    }
}

void AIOBackend::close_files() {
    for (auto &f : this->loader.file_info) {
        ::close(f.fd);
    }
    this->destroy_context();
}

void AIOBackend::destroy_context() {
    if (!this->aio_context_initialized) return;
    int ret = io_destroy(this->aio_ctx);
    if(ret < 0){
        print_and_throw(std::runtime_error("Failed to destroy aio: " + std::string(strerror(-ret))));
    }
    this->aio_context_initialized = false;
}

IOSubmitStatus AIOBackend::submit_read(chunk_id_t id) {
    Chunk &chunk = this->loader.chunks[id];
    const ChunkIOState &state = chunk.io_state;
    size_t read_offset = ROUND_DOWN(state.bytes_completed, this->loader.rank_alignment);
    size_t remaining_size = state.rank_size - read_offset;
    size_t window_idx = id % this->loader.io_depth;
    struct iocb *iocb = this->aio_iocb_ptrs[window_idx];
    io_prep_pread(iocb, this->loader.file_info[chunk.file_index].fd,
        (char*)this->loader.host_buffer + state.window_offset + read_offset,
        ROUND_UP(remaining_size, this->loader.rank_alignment),
        state.rank_file_offset + read_offset);
    iocb->data = reinterpret_cast<void*>(static_cast<uintptr_t>(id));
    int ret = io_submit(this->aio_ctx, 1, this->aio_iocb_ptrs.data() + window_idx);
    if (ret == -EAGAIN || ret == -EINTR || ret == 0) {
        return IOSubmitStatus::Retry;
    }
    if (ret != 1) {
        throw std::runtime_error("Failed to submit aio: " + std::string(strerror(-ret)));
    }
    return IOSubmitStatus::Submitted;
}

IOSubmitStatus AIOBackend::schedule_read(chunk_id_t id) {
    Chunk &chunk = this->loader.chunks[id];
    ChunkIOState &state = chunk.io_state;
    if (state.bytes_completed >= state.rank_size) {
        return IOSubmitStatus::Completed;
    } else if (state.rank_size % this->loader.rank_alignment != 0) {
        int worker_request_id = this->next_io_worker_task_id();
        this->last_page_reader_thread->submit(worker_request_id,
            [this, id]() { return this->submit_read(id); });
        this->aio_last_page_submissions.emplace(worker_request_id, id);
    } else {
        return this->submit_read(id);
    }
    return IOSubmitStatus::Submitted;
}

void AIOBackend::poll(std::vector<IOCompletion> &completed) {
    SingleThreadTaskExecutor::ResultItem result;
    while (this->last_page_reader_thread->try_reap_any(result)) {
        chunk_id_t id = this->aio_last_page_submissions.at(result.request_id);
        this->aio_last_page_submissions.erase(result.request_id);
        ChunkIOState &state = this->loader.chunks[id].io_state;
        if (result.value.type() == typeid(std::exception_ptr)) {
            completed.push_back({state.io_request_id,
                std::any_cast<std::exception_ptr>(result.value)});
        } else if (std::any_cast<IOSubmitStatus>(result.value) == IOSubmitStatus::Retry) {
            completed.push_back({state.io_request_id, {}, true});
        }
    }
    timespec timeout{0, 0};
    int got = io_getevents(this->aio_ctx, 0, this->loader.io_depth, this->aio_events.data(), &timeout);
    if (got == -EAGAIN || got == -EINTR) return;
    if (got < 0) {
        throw std::runtime_error("Failed to get aio events: " + std::string(strerror(-got)));
    }
    for (int i = 0; i < got; ++i) {
        const auto& event = this->aio_events[i];
        chunk_id_t id = static_cast<chunk_id_t>(reinterpret_cast<uintptr_t>(event.data));
        // libaio declares res unsigned on some architectures; kernel errors are signed.
        ssize_t result = static_cast<ssize_t>(event.res);
        completed.push_back(this->process_read_result(id, result));
    }
}

void AIOBackend::drain_submissions() {
    // IO completion can precede helper return. Consume late results before join
    // so the helper cannot be held up by its result queue during shutdown.
    SingleThreadTaskExecutor::ResultItem result;
    while (!this->aio_last_page_submissions.empty()) {
        if (this->last_page_reader_thread->try_reap_any(result)) {
            this->aio_last_page_submissions.erase(result.request_id);
        } else {
            std::this_thread::yield();
        }
    }
}

IORequest AIOBackend::post_read_chunk(const ChunkIOParams &p) {
    chunk_id_t id = p.chunk_id;
    ChunkIOState &state = this->loader.chunks[id].io_state;
    int request_id = this->loader.next_loader_task_id();
    state.io_request_id = request_id;
    state.bytes_completed = 0;
    this->loader.io_thread->submit(request_id,
        [this, id]() { return this->schedule_read(id); });
    return IORequest{this->loader.io_thread.get(), request_id, false};
}

std::unique_ptr<IOBackend> make_aio_backend(Loader &loader) {
    return std::make_unique<AIOBackend>(loader);
}

} // namespace instanttensor
