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
    int ret = io_destroy(this->aio_ctx);
    if(ret < 0){
        print_and_throw(std::runtime_error("Failed to destroy aio: " + std::string(strerror(-ret))));
    }
}

IORequest Loader::post_read_chunk_aio(const ChunkIOParams &p) {
    chunk_id_t chunk_id = p.chunk_id;
    ChunkExtraData &initial_state = this->chunks[chunk_id].extra_data;
    initial_state.total_logical_size = p.rank_size;
    initial_state.bytes_completed = 0;
    initial_state.request_file_offset = p.chunk.file_offset + p.rank_offset;
    initial_state.request_buffer_offset = p.window_offset;
    initial_state.request_logical_size = p.rank_size;
    initial_state.pending_worker_request_id = EXECUTOR_STOP_REQUEST_ID;

    auto submit_chunk = [this](chunk_id_t id) {
        Chunk &chunk = this->chunks[id];
        ChunkExtraData &state = chunk.extra_data;
        if (state.request_logical_size == 0) {
            return;
        }
        size_t window_idx = id % this->io_depth;
        struct iocb *iocb = this->aio_iocb_ptrs[window_idx];
        io_prep_pread(iocb, this->file_info[chunk.file_index].fd,
                      (char*)this->host_buffer + state.request_buffer_offset,
                      ROUND_UP(state.request_logical_size, this->rank_alignment),
                      state.request_file_offset);
        iocb->data = reinterpret_cast<void*>(static_cast<uintptr_t>(id));
        while (true) {
            int ret = io_submit(this->aio_ctx, 1, this->aio_iocb_ptrs.data() + window_idx);
            if (ret < 0) {
                if (ret == -EAGAIN || ret == -EINTR) {
                    if (!this->io_retry_warning_emitted) {
                        fprintf(stderr, "[InstantTensor][WARN] retrying AIO submit for chunk %zd after %s\n",
                            id, strerror(-ret));
                        this->io_retry_warning_emitted = true;
                    }
                    std::this_thread::yield();
                    continue;
                }
                print_and_throw(std::runtime_error(
                    "Failed to submit aio: " + std::string(strerror(-ret))));
            }
            if (ret == 1) {
                break;
            }
            if (!this->io_retry_warning_emitted) {
                fprintf(stderr, "[InstantTensor][WARN] retrying AIO submit for chunk %zd after zero submission\n", id);
                this->io_retry_warning_emitted = true;
            }
            std::this_thread::yield();
        }
    };

    auto schedule_chunk = [this, submit_chunk](chunk_id_t id) {
        ChunkExtraData &state = this->chunks[id].extra_data;
        bool unaligned_last_page =
            state.request_logical_size % this->rank_alignment != 0;
        if (unaligned_last_page) {
            state.pending_worker_request_id = this->next_io_worker_task_id();
            int request_id = state.pending_worker_request_id;
            this->last_page_reader_thread->submit(request_id, [submit_chunk, id]() {
                submit_chunk(id);
            });
        } else {
            submit_chunk(id);
        }
    };

    auto io_func = [=]() mutable -> bool {
        ChunkExtraData &state = this->chunks[chunk_id].extra_data;
        if (state.pending_worker_request_id != EXECUTOR_STOP_REQUEST_ID) {
            std::any ignored;
            if (!this->last_page_reader_thread->try_reap(
                    state.pending_worker_request_id, ignored)) {
                return false;
            }
            state.pending_worker_request_id = EXECUTOR_STOP_REQUEST_ID;
        }
        timespec timeout{0, 0};
        int got = io_getevents(this->aio_ctx, 0, this->io_depth,
                               this->aio_events.data(), &timeout);
        if (got < 0) {
            if (got == -EAGAIN || got == -EINTR) {
                if (!this->io_retry_warning_emitted) {
                    fprintf(stderr, "[InstantTensor][WARN] retrying AIO completion poll for chunk %zd after %s\n",
                        chunk_id, strerror(-got));
                    this->io_retry_warning_emitted = true;
                }
                return false;
            }
            print_and_throw(std::runtime_error(
                "Failed to get aio events: " + std::string(strerror(-got))));
        }
        for (int i = 0; i < got; i++) {
            chunk_id_t event_chunk_id = static_cast<chunk_id_t>(
                reinterpret_cast<uintptr_t>(this->aio_events[i].data));
            Chunk &event_chunk = this->chunks[event_chunk_id];
            ChunkExtraData &event_state = event_chunk.extra_data;
            if (this->aio_events[i].res < 0) {
                int error = static_cast<int>(-this->aio_events[i].res);
                if (error == EAGAIN || error == EINTR) {
                    if (!this->io_retry_warning_emitted) {
                        fprintf(stderr, "[InstantTensor][WARN] retrying AIO read for chunk %zd after %s\n",
                            event_chunk_id, strerror(error));
                        this->io_retry_warning_emitted = true;
                    }
                    schedule_chunk(event_chunk_id);
                    continue;
                }
                print_and_throw(std::runtime_error(
                    "Failed to get aio events: " + std::string(strerror(error))));
            }
            size_t padded_world_size = ROUND_UP(event_chunk.size, this->world_chunk_alignment);
            size_t padded_rank_size = padded_world_size / this->world_size;
            size_t event_rank_offset = padded_rank_size * this->rank;
            size_t logical_size = rank_logical_size(
                event_chunk.size, event_rank_offset, padded_rank_size);
            size_t original_file_offset = event_chunk.file_offset + event_rank_offset;
            size_t bytes_read = static_cast<size_t>(this->aio_events[i].res);
            size_t covered = event_state.request_file_offset - original_file_offset + bytes_read;
            event_state.bytes_completed = std::min(
                logical_size, std::max(event_state.bytes_completed, covered));
            if (event_state.bytes_completed < logical_size) {
                struct stat st;
                if (fstat(this->file_info[event_chunk.file_index].fd, &st) != 0 ||
                    static_cast<size_t>(st.st_size) < event_chunk.file_offset +
                        event_rank_offset + logical_size) {
                    print_and_throw(std::runtime_error(
                        "Unexpected AIO short read at EOF: chunk_id=" +
                        std::to_string(event_chunk_id) + ", bytes_read=" +
                        std::to_string(bytes_read) +
                        ", logical_size=" + std::to_string(logical_size)));
                }
                size_t retry_offset = ROUND_DOWN(
                    original_file_offset + event_state.bytes_completed,
                    this->rank_alignment);
                event_state.request_file_offset = retry_offset;
                event_state.request_buffer_offset =
                    (event_chunk_id % this->io_depth) * this->rank_chunk_size +
                    (retry_offset - original_file_offset);
                event_state.request_logical_size =
                    original_file_offset + logical_size - retry_offset;
                schedule_chunk(event_chunk_id);
                continue;
            }
        }
        return state.bytes_completed >= state.total_logical_size;
    };
    int io_req_id = this->next_loader_task_id();
    this->io_thread->submit(io_req_id, IOOperation{
        [=]() { schedule_chunk(chunk_id); }, std::move(io_func)});
    return IORequest{this->io_thread.get(), io_req_id, false};
}

} // namespace instanttensor
