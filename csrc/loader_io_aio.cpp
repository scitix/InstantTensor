#include <instant_tensor/loader.hpp>

namespace instanttensor {

void Loader::open_file_aio(FileInfo &f) {
    int open_flags = O_RDONLY;
    if (this->backend == Backend::AIO) {// != AIO_BUFFERED
        open_flags |= O_DIRECT;
    }
    f.fd = ::open(f.filename.c_str(), open_flags);
    if (f.fd < 0) {
        throw std::runtime_error("Failed to open file: " + f.filename);
    }
    struct stat st;
    if (fstat(f.fd, &st) < 0) { throw std::runtime_error("Failed to fstat file: " + f.filename); }
    f.size = st.st_size;

    if (this->backend == Backend::AIO_BUFFERED) {
        posix_fadvise(f.fd, 0, 0, POSIX_FADV_SEQUENTIAL);
    }

    this->need_host_buffer = true;
}

void Loader::initialize_aio_context() {
    int ret = io_setup(this->io_depth, &this->aio_ctx);
    if(ret < 0){
        print_and_throw(std::runtime_error("Failed to setup aio: " + std::string(strerror(-ret))));
    }
    this->aio_context_initialized = true;
    this->aio_iocbs.resize(this->io_depth);
    this->aio_iocb_ptrs.resize(this->io_depth);
    for(size_t i = 0; i < this->io_depth; i++) {
        this->aio_iocb_ptrs[i] = &this->aio_iocbs[i];
    }
    this->aio_events.resize(this->io_depth);
}

void Loader::close_file_aio(FileInfo &f) {
    ::close(f.fd);
}

void Loader::destroy_aio_context() {
    if (!this->aio_context_initialized) return;
    int ret = io_destroy(this->aio_ctx);
    if(ret < 0){
        print_and_throw(std::runtime_error("Failed to destroy aio: " + std::string(strerror(-ret))));
    }
    this->aio_context_initialized = false;
}

IOSubmitStatus Loader::submit_read_aio(chunk_id_t id) {
    Chunk &chunk = this->chunks[id];
    const ChunkIOState &state = chunk.io_state;
    size_t read_offset = ROUND_DOWN(state.bytes_completed, this->rank_alignment);
    size_t remaining_size = state.rank_size - read_offset;
    size_t window_idx = id % this->io_depth;
    struct iocb *iocb = this->aio_iocb_ptrs[window_idx];
    io_prep_pread(iocb, this->file_info[chunk.file_index].fd,
        (char*)this->host_buffer + state.window_offset + read_offset,
        ROUND_UP(remaining_size, this->rank_alignment),
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

IOSubmitStatus Loader::schedule_read_aio(chunk_id_t id) {
    Chunk &chunk = this->chunks[id];
    ChunkIOState &state = chunk.io_state;
    if (state.bytes_completed >= state.rank_size) {
        return IOSubmitStatus::Completed;
    } else if (state.rank_size % this->rank_alignment != 0) {
        int worker_request_id = this->next_io_worker_task_id();
        this->last_page_reader_thread->submit(worker_request_id,
            [this, id]() { return this->submit_read_aio(id); });
        this->aio_last_page_submissions.emplace(worker_request_id, id);
    } else {
        return this->submit_read_aio(id);
    }
    return IOSubmitStatus::Submitted;
}

void Loader::poll_aio(std::vector<IOCompletion> &completed) {
    SingleThreadTaskExecutor::ResultItem result;
    while (this->last_page_reader_thread->try_reap_any(result)) {
        chunk_id_t id = this->aio_last_page_submissions.at(result.request_id);
        this->aio_last_page_submissions.erase(result.request_id);
        ChunkIOState &state = this->chunks[id].io_state;
        if (result.value.type() == typeid(std::exception_ptr)) {
            completed.push_back({state.io_request_id,
                std::any_cast<std::exception_ptr>(result.value)});
        } else if (std::any_cast<IOSubmitStatus>(result.value) == IOSubmitStatus::Retry) {
            completed.push_back({state.io_request_id, {}, true});
        }
    }
    timespec timeout{0, 0};
    int got = io_getevents(this->aio_ctx, 0, this->io_depth, this->aio_events.data(), &timeout);
    if (got == -EAGAIN || got == -EINTR) return;
    if (got < 0) {
        throw std::runtime_error("Failed to get aio events: " + std::string(strerror(-got)));
    }
    for (int i = 0; i < got; ++i) {
        const auto& event = this->aio_events[i];
        chunk_id_t id = static_cast<chunk_id_t>(reinterpret_cast<uintptr_t>(event.data));
        // libaio declares res unsigned on some architectures; kernel errors are signed.
        ssize_t result = static_cast<ssize_t>(event.res);
        completed.push_back(this->complete_native_read(id, result));
    }
}

void Loader::drain_aio_submissions() {
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

IORequest Loader::post_read_chunk_aio(const ChunkIOParams &p) {
    chunk_id_t id = p.chunk_id;
    ChunkIOState &state = this->chunks[id].io_state;
    int request_id = this->next_loader_task_id();
    state.io_request_id = request_id;
    state.bytes_completed = 0;
    this->io_thread->submit(request_id,
        [this, id]() { return this->schedule_read_aio(id); });
    return IORequest{this->io_thread.get(), request_id, false};
}

} // namespace instanttensor
