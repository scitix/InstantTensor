#include <instant_tensor/loader.hpp>
#include <liburing.h>
#include <sys/utsname.h>

namespace instanttensor {

// The loader thread prepares and submits SQEs. cuda_thread consumes CQEs before
// launching H2D and NCCL work. Each side has a single caller for its ring API.

#define IO_URING_REGISTER_BUFFER_SIZE (1<<30) // 1GiB buffer size limit

// ─── file open / close ───────────────────────────────────────────────────────

namespace {

struct KernelVersion {
    long major;
    long minor;
    string release;
};

std::optional<KernelVersion> running_kernel_version() {
    struct utsname uts;
    if (uname(&uts) != 0) {
        return std::nullopt;
    }

    char *end = nullptr;
    long major = strtol(uts.release, &end, 10);
    if (end == uts.release || *end != '.') {
        return std::nullopt;
    }

    const char *minor_start = end + 1;
    long minor = strtol(minor_start, &end, 10);
    if (end == minor_start) {
        return std::nullopt;
    }

    return KernelVersion{major, minor, uts.release};
}

bool kernel_version_at_least(
    const KernelVersion &version, long required_major, long required_minor) {
    return version.major > required_major ||
           (version.major == required_major && version.minor >= required_minor);
}

bool probe_supports_fixed_buffer_read(struct io_uring *ring, string *reason) {
    struct io_uring_probe *probe = io_uring_get_probe_ring(ring);
    if (!probe) {
        *reason = "Could not query the supported io_uring opcodes.";
        return false;
    }

    bool supports_read = io_uring_opcode_supported(probe, IORING_OP_READ);
    bool supports_read_fixed =
        io_uring_opcode_supported(probe, IORING_OP_READ_FIXED);
    io_uring_free_probe(probe);
    if (!supports_read || !supports_read_fixed) {
        *reason = "The required io_uring READ and READ_FIXED opcodes are unavailable.";
        return false;
    }
    return true;
}

bool supports_fixed_file_registration(struct io_uring *ring, string *reason) {
    int fd = ::open("/dev/null", O_RDONLY);
    if (fd < 0) {
        int error = errno;
        *reason = "Could not open /dev/null while probing io_uring fixed-file "
                  "registration: " + std::string(strerror(error)) + ".";
        return false;
    }

    int ret = io_uring_register_files(ring, &fd, 1);
    if (ret == 0) {
        io_uring_unregister_files(ring);
    }
    ::close(fd);
    if (ret < 0) {
        *reason = "io_uring fixed-file registration failed: " +
                  std::string(strerror(-ret)) + ".";
    }
    return ret == 0;
}

bool supports_fixed_buffer_registration(struct io_uring *ring, string *reason) {
    char buffer[4096];
    struct iovec iov = {buffer, sizeof(buffer)};

    int ret = io_uring_register_buffers(ring, &iov, 1);
    if (ret == 0) {
        io_uring_unregister_buffers(ring);
    } else {
        *reason = "io_uring fixed-buffer registration failed: " +
                  std::string(strerror(-ret)) + ".";
    }
    return ret == 0;
}

} // namespace

BackendStatus Loader::uring_status(){// best-effort check
    auto kernel_version = running_kernel_version();
    if (!kernel_version) {
        return {false, "Could not determine the Linux kernel version.", ""};
    }

    // IORING_OP_READ, IOSQE_ASYNC, and IORING_REGISTER_PROBE require 5.6.
    if (!kernel_version_at_least(*kernel_version, 5, 6)) {
        return {
            false,
            "io_uring requires Linux kernel 5.6 or newer; the detected kernel "
            "version is " + kernel_version->release + ".",
            "",
        };
    }

    string warning;
    if (!kernel_version_at_least(*kernel_version, 5, 15)) {
        warning = "io_uring on Linux " + kernel_version->release +
                  " may be unstable; Linux 5.15 or newer is recommended.";
    }

    struct io_uring ring = {};
    struct io_uring_params params = {};
    // params.flags = IORING_SETUP_SQPOLL; // SQPOLL is not very necessary for us
    // params.sq_thread_idle = 1000;

    int ret = io_uring_queue_init_params(2, &ring, &params);
    if (ret < 0) {
        return {
            false,
            "io_uring queue initialization failed: " +
                std::string(strerror(-ret)) + ".",
            warning,
        };
    }

    // io_uring_probe reports opcodes; SQPOLL/fixed files are setup/register capabilities.
    string reason;
    bool supported = probe_supports_fixed_buffer_read(&ring, &reason) &&
                     supports_fixed_file_registration(&ring, &reason) &&
                     supports_fixed_buffer_registration(&ring, &reason);
    io_uring_queue_exit(&ring);
    return {supported, reason, warning};
}

void Loader::open_file_uring(FileInfo &f) {
    // URING uses O_DIRECT; URING_BUFFERED uses the page cache and io-wq
    // workers for asynchronous buffered reads.
    int open_flags = O_RDONLY;
    if (this->backend == Backend::URING) {
        open_flags |= O_DIRECT;
    }
    f.fd = ::open(f.filename.c_str(), open_flags);
    if (f.fd < 0) {
        throw std::runtime_error("Failed to open file: " + f.filename);
    }
    struct stat st;
    if (fstat(f.fd, &st) < 0) {
        throw std::runtime_error("Failed to fstat file: " + f.filename);
    }
    f.size = st.st_size;

    // Hint sequential access so the VFS read-ahead fills the page cache ahead
    // of our reads and reduces the time spent in the io-wq workers.
    if (this->backend == Backend::URING_BUFFERED) {
        posix_fadvise(f.fd, 0, 0, POSIX_FADV_SEQUENTIAL);
    }

    this->need_host_buffer = true;
}

void Loader::initialize_uring_context() {
    int ret = io_uring_queue_init((unsigned)this->io_depth, &this->uring_ring, 0);
    if (ret < 0) {
        throw std::runtime_error(
            "io_uring_queue_init failed: " + std::string(strerror(-ret)));
    }
    // Since the SQ and CQ of uring both operate in SPSC mode, we use an extra ring to submit IO for the last page.
    // ret = io_uring_queue_init((unsigned)this->io_depth, &this->uring_ring_last_page, 0);
    // if (ret < 0) {
    //     throw std::runtime_error(
    //         "io_uring_queue_init failed: " + std::string(strerror(-ret)));
    // }

    // vector<unsigned int>num_workers = {32, 0};
    // io_uring_register_iowq_max_workers(&this->uring_ring, num_workers.data());
    // fprintf(stderr, "io_uring_register_iowq_max_workers: %d, %d\n", num_workers[0], num_workers[1]);

    if(this->uring_register_file) {
        vector<int> fds;
        for(const auto &f : this->file_info) {
            fds.push_back(f.fd);
        }
        ret = io_uring_register_files(&this->uring_ring, fds.data(), fds.size());
        if (ret < 0) {
            throw std::runtime_error(
                "io_uring_register failed: " + std::string(strerror(-ret)));
        }
    }
}

void Loader::close_file_uring(FileInfo &f) {
    ::close(f.fd);
}

void Loader::destroy_uring_context() {
    if(this->uring_register_file) {
        io_uring_unregister_files(&this->uring_ring);
    }

    // io_uring_queue_exit(&this->uring_ring_last_page);
    io_uring_queue_exit(&this->uring_ring);
}

void Loader::register_host_buffer_uring() {
    if(this->uring_register_buffer) {
        void *ptr = this->host_buffer_entry.ptr;
        size_t size = this->host_buffer_entry.size;
        vector<struct iovec> iovs;
        while(size > 0) {
            size_t iov_size = std::min(size, (size_t)IO_URING_REGISTER_BUFFER_SIZE);
            iovs.push_back({ptr, iov_size});
            ptr = (char*)ptr + iov_size;
            size -= iov_size;
        }

        int ret = io_uring_register_buffers(&this->uring_ring, iovs.data(), iovs.size());
        if (ret < 0) {
            throw std::runtime_error(
                "io_uring_register_buffers failed: " + std::string(strerror(-ret)));
        }
    }
}

void Loader::deregister_host_buffer_uring() {
    if(this->uring_register_buffer) {
        io_uring_unregister_buffers(&this->uring_ring);
    }
}

// ─── chunk read ──────────────────────────────────────────────────────────────

IORequest Loader::post_read_chunk_uring(const ChunkIOParams &p) {
    chunk_id_t chunk_id = p.chunk_id;
    ChunkExtraData &initial_state = this->chunks[chunk_id].extra_data;
    initial_state.total_logical_size = p.rank_size;
    initial_state.bytes_completed = 0;
    initial_state.request_file_offset = p.chunk.file_offset + p.rank_offset;
    initial_state.request_buffer_offset = p.window_offset;
    initial_state.request_logical_size = p.rank_size;

    auto submit_chunk = [this](chunk_id_t id) {
        Chunk &chunk = this->chunks[id];
        ChunkExtraData &state = chunk.extra_data;
        if (state.request_logical_size == 0) {
            return;
        }
        size_t rank_size_aligned = ROUND_UP(state.request_logical_size, this->rank_alignment);
        bool unaligned_last_page = state.request_logical_size != rank_size_aligned;
        void *buf = (char*)this->host_buffer + state.request_buffer_offset;
        struct io_uring_sqe *sqe = io_uring_get_sqe(&this->uring_ring);
        if (!sqe) {
            throw std::runtime_error("io_uring SQ full");
        }
        int file_handle = this->uring_register_file
            ? static_cast<int>(chunk.file_index)
            : this->file_info[chunk.file_index].fd;
        int buffer_index = -1;
        if (this->uring_register_buffer) {
            size_t left = ((char*)buf - (char*)this->host_buffer_entry.ptr) /
                IO_URING_REGISTER_BUFFER_SIZE;
            size_t right = ((char*)buf + rank_size_aligned - 1 -
                (char*)this->host_buffer_entry.ptr) / IO_URING_REGISTER_BUFFER_SIZE;
            if (left == right) {
                buffer_index = static_cast<int>(left);
            }
        }
        if (buffer_index != -1) {
            io_uring_prep_read_fixed(sqe, file_handle, buf,
                static_cast<unsigned>(rank_size_aligned),
                state.request_file_offset, buffer_index);
        } else {
            io_uring_prep_read(sqe, file_handle, buf,
                static_cast<unsigned>(rank_size_aligned),
                state.request_file_offset);
        }
        if (this->uring_register_file) {
            sqe->flags |= IOSQE_FIXED_FILE;
        }
        if (this->backend == Backend::URING_BUFFERED || unaligned_last_page) {
            sqe->flags |= IOSQE_ASYNC;
        }
        io_uring_sqe_set_data64(sqe, static_cast<uint64_t>(id));
        size_t submitted = 0;
        while (submitted < 1) {
            int ret = io_uring_submit(&this->uring_ring);
            if (ret < 0) {
                if (ret == -EAGAIN || ret == -EINTR) {
                    if (!this->io_retry_warning_emitted) {
                        fprintf(stderr, "[InstantTensor][WARN] retrying io_uring submit for chunk %zd after %s\n",
                            id, strerror(-ret));
                        this->io_retry_warning_emitted = true;
                    }
                    std::this_thread::yield();
                    continue;
                }
                throw std::runtime_error(
                    "io_uring_submit failed: " + std::string(strerror(-ret)));
            }
            submitted += ret;
            if (ret == 0) {
                if (!this->io_retry_warning_emitted) {
                    fprintf(stderr, "[InstantTensor][WARN] retrying io_uring submit for chunk %zd after zero submission\n", id);
                    this->io_retry_warning_emitted = true;
                }
                std::this_thread::yield();
            }
        }
    };

    // Consume CQEs and publish a complete host read to the common CUDA path.
    auto io_func = [=]() -> bool {
        ChunkExtraData &state = this->chunks[chunk_id].extra_data;
        for (size_t i = 0; i < this->io_depth &&
             state.bytes_completed < state.total_logical_size; ++i) {
            struct io_uring_cqe *cqe;
            int ret = io_uring_peek_cqe(&this->uring_ring, &cqe);
            if (ret == -EAGAIN) {
                break;
            }
            if (ret == -EINTR) {
                if (!this->io_retry_warning_emitted) {
                    fprintf(stderr, "[InstantTensor][WARN] retrying io_uring completion poll for chunk %zd after EINTR\n", chunk_id);
                    this->io_retry_warning_emitted = true;
                }
                return false;
            }
            if (ret != 0) {
                throw std::runtime_error(
                    "io_uring_peek_cqe failed: " + std::string(strerror(-ret)));
            }
            chunk_id_t cqe_chunk_id = static_cast<chunk_id_t>(io_uring_cqe_get_data64(cqe));
            Chunk &cqe_chunk = this->chunks[cqe_chunk_id];
            ChunkExtraData &event_state = cqe_chunk.extra_data;
            if (cqe->res < 0) {
                int error = -cqe->res;
                io_uring_cqe_seen(&this->uring_ring, cqe);
                if (error == EAGAIN || error == EINTR) {
                    if (!this->io_retry_warning_emitted) {
                        fprintf(stderr, "[InstantTensor][WARN] retrying io_uring read for chunk %zd after %s\n",
                            cqe_chunk_id, strerror(error));
                        this->io_retry_warning_emitted = true;
                    }
                    submit_chunk(cqe_chunk_id);
                    continue;
                }
                std::string msg =
                    "io_uring read error for chunk id: " + std::to_string(cqe_chunk_id) + ", error: " + std::string(strerror(error));
                throw std::runtime_error(msg);
            }
            size_t padded_world_size = ROUND_UP(cqe_chunk.size, this->world_chunk_alignment);
            size_t padded_rank_size = padded_world_size / this->world_size;
            size_t logical_size = rank_logical_size(
                cqe_chunk.size, padded_rank_size * this->rank, padded_rank_size);
            size_t original_file_offset = cqe_chunk.file_offset +
                padded_rank_size * this->rank;
            if(static_cast<size_t>(cqe->res) < logical_size) {
                int bytes_read = cqe->res;
                io_uring_cqe_seen(&this->uring_ring, cqe);
                struct stat st;
                if (fstat(this->file_info[cqe_chunk.file_index].fd, &st) != 0 ||
                    static_cast<size_t>(st.st_size) < cqe_chunk.file_offset +
                        padded_rank_size * this->rank + logical_size) {
                    throw std::runtime_error(
                        "Unexpected io_uring short read at EOF: chunk_id=" +
                        std::to_string(cqe_chunk_id) + ", bytes_read=" +
                        std::to_string(bytes_read) + ", logical_size=" +
                        std::to_string(logical_size));
                }
                size_t covered = event_state.request_file_offset - original_file_offset +
                    static_cast<size_t>(bytes_read);
                event_state.bytes_completed = std::min(
                    logical_size, std::max(event_state.bytes_completed, covered));
                size_t retry_offset = ROUND_DOWN(
                    original_file_offset + event_state.bytes_completed,
                    this->rank_alignment);
                event_state.request_file_offset = retry_offset;
                event_state.request_buffer_offset =
                    (cqe_chunk_id % this->io_depth) * this->rank_chunk_size +
                    (retry_offset - original_file_offset);
                event_state.request_logical_size =
                    original_file_offset + logical_size - retry_offset;
                submit_chunk(cqe_chunk_id);
                continue;
            }
            event_state.bytes_completed = logical_size;
            io_uring_cqe_seen(&this->uring_ring, cqe);
        }
        return state.bytes_completed >= state.total_logical_size;
    };
    int io_req_id = this->next_loader_task_id();
    this->io_thread->submit(io_req_id, IOOperation{
        [=]() { submit_chunk(chunk_id); }, std::move(io_func)});
    return IORequest{this->io_thread.get(), io_req_id, false};
}

} // namespace instanttensor
