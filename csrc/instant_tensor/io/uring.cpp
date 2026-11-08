#include <instant_tensor/loader.hpp>
#include <cerrno>
#include <cstring>
#include <fcntl.h>
#include <sys/stat.h>
#include <liburing.h>
#include <sys/utsname.h>

namespace instanttensor {

class UringBackend final : public IOBackend {
public:
    using IOBackend::IOBackend;
    bool needs_host_buffer() const override { return true; }
    void open_files() override;
    void close_files() override;
    void destroy_context();
    void init_buffer() override;
    void destroy_buffer() override;
    void abort() override;
    IORequest post_read_chunk(const ChunkIOParams &p) override;
    void poll(std::vector<IOCompletion> &completed) override;
    IOSubmitStatus submit_read(chunk_id_t id);

    bool uring_context_initialized = false;
    bool uring_register_file = true;
    bool uring_register_buffer = true; // Important for buffered IO performance.
    struct io_uring uring_ring = {};
    size_t uring_reads_pending = 0;
private:
    void open_file(FileInfo &file);
};

// The IO worker owns both SQ and CQ. CUDA sees only complete logical reads.

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

BackendStatus uring_status(){// best-effort check
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

void UringBackend::open_file(FileInfo &f) {
    // URING uses O_DIRECT; URING_BUFFERED uses the page cache and io-wq
    // workers for asynchronous buffered reads.
    int open_flags = O_RDONLY;
    if (this->loader.backend == Backend::URING) {
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
    if (this->loader.backend == Backend::URING_BUFFERED) {
        posix_fadvise(f.fd, 0, 0, POSIX_FADV_SEQUENTIAL);
    }
}

void UringBackend::open_files() {
    int ret = io_uring_queue_init((unsigned)this->loader.io_depth, &this->uring_ring, 0);
    if (ret < 0) {
        throw std::runtime_error(
            "io_uring_queue_init failed: " + std::string(strerror(-ret)));
    }
    this->uring_context_initialized = true;
    for (auto &f : this->loader.file_info) {
        this->open_file(f);
    }
    if(this->uring_register_file) {
        vector<int> fds;
        for(const auto &f : this->loader.file_info) {
            fds.push_back(f.fd);
        }
        int ret = io_uring_register_files(&this->uring_ring, fds.data(), fds.size());
        if (ret < 0) {
            throw std::runtime_error(
                "io_uring_register failed: " + std::string(strerror(-ret)));
        }
    }
}

void UringBackend::close_files() {
    for (auto &f : this->loader.file_info) {
        ::close(f.fd);
    }
    this->destroy_context();
}

void UringBackend::destroy_context() {
    if (!this->uring_context_initialized) return;
    if(this->uring_register_file) {
        io_uring_unregister_files(&this->uring_ring);
    }

    io_uring_queue_exit(&this->uring_ring);
    this->uring_context_initialized = false;
}

void UringBackend::abort() {
    if (!this->uring_context_initialized) return;
    // close(ring_fd) alone need not synchronously finish kernel teardown.
    // Drain accepted reads before unpinning or releasing their host windows.
    while (this->uring_reads_pending != 0) {
        struct io_uring_cqe *cqe;
        int ret = io_uring_wait_cqe(&this->uring_ring, &cqe);
        if (ret == -EINTR || ret == -EAGAIN) {
            std::this_thread::yield();
            continue;
        }
        if (ret != 0) {
            throw std::runtime_error("Failed to drain io_uring: " + std::string(strerror(-ret)));
        }
        io_uring_cqe_seen(&this->uring_ring, cqe);
        --this->uring_reads_pending;
    }
    this->destroy_buffer();
    this->destroy_context();
}

void UringBackend::init_buffer() {
    if(this->uring_register_buffer) {
        void *ptr = this->loader.host_buffer_entry.ptr;
        size_t size = this->loader.host_buffer_entry.size;
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

void UringBackend::destroy_buffer() {
    if(this->uring_context_initialized && this->uring_register_buffer) {
        io_uring_unregister_buffers(&this->uring_ring);
    }
}

// ─── chunk read ──────────────────────────────────────────────────────────────

IOSubmitStatus UringBackend::submit_read(chunk_id_t id) {
    Chunk &chunk = this->loader.chunks[id];
    const ChunkIOState &state = chunk.io_state;
    if (state.bytes_completed >= state.rank_size) {
        return IOSubmitStatus::Completed;
    }
    size_t read_offset = ROUND_DOWN(state.bytes_completed, this->loader.rank_alignment);
    size_t remaining_size = state.rank_size - read_offset;
    size_t rank_size_aligned = ROUND_UP(remaining_size, this->loader.rank_alignment);
    bool unaligned_last_page = remaining_size != rank_size_aligned;
    void *buf = (char*)this->loader.host_buffer + state.window_offset + read_offset;
    struct io_uring_sqe *sqe = io_uring_get_sqe(&this->uring_ring);
    if (!sqe) {
        throw std::runtime_error("io_uring SQ full");
    }
    int file_handle = this->uring_register_file
        ? static_cast<int>(chunk.file_index)
        : this->loader.file_info[chunk.file_index].fd;
    int buffer_index = -1;
    if (this->uring_register_buffer) {
        size_t left = ((char*)buf - (char*)this->loader.host_buffer_entry.ptr) /
            IO_URING_REGISTER_BUFFER_SIZE;
        size_t right = ((char*)buf + rank_size_aligned - 1 -
            (char*)this->loader.host_buffer_entry.ptr) / IO_URING_REGISTER_BUFFER_SIZE;
        if (left == right) {
            buffer_index = static_cast<int>(left);
        }
    }
    if (buffer_index != -1) {
        io_uring_prep_read_fixed(sqe, file_handle, buf,
            static_cast<unsigned>(rank_size_aligned),
            state.rank_file_offset + read_offset, buffer_index);
    } else {
        io_uring_prep_read(sqe, file_handle, buf,
            static_cast<unsigned>(rank_size_aligned),
            state.rank_file_offset + read_offset);
    }
    if (this->uring_register_file) {
        sqe->flags |= IOSQE_FIXED_FILE;
    }
    if (this->loader.backend == Backend::URING_BUFFERED || unaligned_last_page) {
        sqe->flags |= IOSQE_ASYNC;
    }
    io_uring_sqe_set_data64(sqe, static_cast<uint64_t>(id));
    // submit() may already have flushed the SQE; retry without preparing another.
    while (true) {
        int ret = io_uring_submit(&this->uring_ring);
        if (ret == -EAGAIN || ret == -EINTR || ret == 0) {
            std::this_thread::yield();
            continue;
        }
        if (ret != 1) {
            throw std::runtime_error("io_uring_submit failed: " + std::string(strerror(-ret)));
        }
        break;
    }
    ++this->uring_reads_pending;
    return IOSubmitStatus::Submitted;
}

void UringBackend::poll(std::vector<IOCompletion> &completed) {
    struct io_uring_cqe *cqes[MAX_IO_DEPTH];
    unsigned count = io_uring_peek_batch_cqe(
        &this->uring_ring, cqes, static_cast<unsigned>(this->loader.io_depth));
    if (count == 0) {
        // Batch peek hides CQ flush errors; preserve the single-peek error path.
        int ret = io_uring_peek_cqe(&this->uring_ring, &cqes[0]);
        if (ret == -EAGAIN || ret == -EINTR) return;
        if (ret != 0) {
            throw std::runtime_error("io_uring_peek_cqe failed: " + std::string(strerror(-ret)));
        }
        count = 1;
    }
    completed.reserve(completed.size() + count);
    for (unsigned i = 0; i < count; ++i) {
        chunk_id_t id = static_cast<chunk_id_t>(io_uring_cqe_get_data64(cqes[i]));
        completed.push_back(this->process_read_result(id, cqes[i]->res));
    }
    io_uring_cq_advance(&this->uring_ring, count);
    this->uring_reads_pending -= count;
}

IORequest UringBackend::post_read_chunk(const ChunkIOParams &p) {
    chunk_id_t id = p.chunk_id;
    ChunkIOState &state = this->loader.chunks[id].io_state;
    int request_id = this->loader.next_loader_task_id();
    state.io_request_id = request_id;
    state.bytes_completed = 0;
    this->loader.io_thread->submit(request_id,
        [this, id]() { return this->submit_read(id); });
    return IORequest{this->loader.io_thread.get(), request_id, false};
}

std::unique_ptr<IOBackend> make_uring_backend(Loader &loader) {
    return std::make_unique<UringBackend>(loader);
}

} // namespace instanttensor
