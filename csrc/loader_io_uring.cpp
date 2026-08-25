#include <instant_tensor/loader.hpp>
#include <liburing.h>
#include <sys/utsname.h>

namespace instanttensor {

// The loader thread prepares and submits SQEs. cuda_thread consumes CQEs before
// launching H2D and NCCL work. Each side has a single caller for its ring API.

#define IO_URING_REGISTER_BUFFER_SIZE (1<<30) // 1GiB buffer size limit

// ─── file open / close ───────────────────────────────────────────────────────

namespace {

bool kernel_at_least_5_15() {
    struct utsname uts;
    if (uname(&uts) != 0) {
        return false;
    }

    char *end = nullptr;
    long major = strtol(uts.release, &end, 10);
    if (end == uts.release || *end != '.') {
        return false;
    }

    const char *minor_start = end + 1;
    long minor = strtol(minor_start, &end, 10);
    if (end == minor_start) {
        return false;
    }

    return major > 5 || (major == 5 && minor >= 15);
}

bool probe_supports_fixed_buffer_read(struct io_uring *ring) {
    struct io_uring_probe *probe = io_uring_get_probe_ring(ring);
    if (!probe) {
        return false;
    }

    bool supported = io_uring_opcode_supported(probe, IORING_OP_READ) &&
                     io_uring_opcode_supported(probe, IORING_OP_READ_FIXED);
    io_uring_free_probe(probe);
    return supported;
}

bool supports_fixed_file_registration(struct io_uring *ring) {
    int fd = ::open("/dev/null", O_RDONLY);
    if (fd < 0) {
        return false;
    }

    int ret = io_uring_register_files(ring, &fd, 1);
    if (ret == 0) {
        io_uring_unregister_files(ring);
    }
    ::close(fd);
    return ret == 0;
}

bool supports_fixed_buffer_registration(struct io_uring *ring) {
    char buffer[4096];
    struct iovec iov = {buffer, sizeof(buffer)};

    int ret = io_uring_register_buffers(ring, &iov, 1);
    if (ret == 0) {
        io_uring_unregister_buffers(ring);
    }
    return ret == 0;
}

} // namespace

bool Loader::uring_available(){// best-effort check
    if (!kernel_at_least_5_15()) { // io_uring with kernel < 5.15 is not reliable
        return false;
    }

    struct io_uring ring = {};
    struct io_uring_params params = {};
    // params.flags = IORING_SETUP_SQPOLL; // SQPOLL is not very necessary for us
    // params.sq_thread_idle = 1000;

    int ret = io_uring_queue_init_params(2, &ring, &params);
    if (ret < 0) {
        return false;
    }

    // io_uring_probe reports opcodes; SQPOLL/fixed files are setup/register capabilities.
    bool supported = // (ring.flags & IORING_SETUP_SQPOLL) &&
                     probe_supports_fixed_buffer_read(&ring) &&
                     supports_fixed_file_registration(&ring) &&
                     supports_fixed_buffer_registration(&ring);
    io_uring_queue_exit(&ring);
    return supported;
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
    this->need_cuda_thread = true;
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

ChunkRequest Loader::post_read_chunk_uring(const ChunkIOParams &p) {
    chunk_id_t chunk_id = p.chunk_id;
    size_t rank_size_aligned = ROUND_UP(p.rank_size, this->rank_alignment);
    bool unaligned_last_page = p.rank_size != rank_size_aligned;

    struct io_uring *selected_ring = &this->uring_ring; // unaligned_last_page ? &this->uring_ring_last_page : &this->uring_ring;

    size_t submit_cnt = 0;
    if(p.rank_size > 0) {
        void *buf = (char*)this->host_buffer + p.window_offset;
        struct io_uring_sqe *sqe = io_uring_get_sqe(selected_ring);
        if (!sqe) {
            throw std::runtime_error(
                "io_uring SQ full");
        }
        // Buffered I/O intentionally reuses the page-rounded direct-I/O size.
        // H2D copies only the logically valid rank bytes.
        int file_handle = this->uring_register_file ? p.chunk.file_index : p.file.fd;

        int buffer_index = -1;
        if(this->uring_register_buffer) {
            size_t left_buffer_index = ((char*)buf - (char*)this->host_buffer_entry.ptr) / IO_URING_REGISTER_BUFFER_SIZE;
            size_t right_buffer_index = ((char*)buf + rank_size_aligned - 1 - (char*)this->host_buffer_entry.ptr) / IO_URING_REGISTER_BUFFER_SIZE;
            if(left_buffer_index == right_buffer_index) {
                buffer_index = (int)left_buffer_index;
            }
        }
        if(buffer_index != -1) {
            io_uring_prep_read_fixed(sqe, file_handle, buf,
                (unsigned)rank_size_aligned, p.chunk.file_offset + p.rank_offset, buffer_index);
        }
        else {
            io_uring_prep_read(sqe, file_handle, buf,
                (unsigned)rank_size_aligned, p.chunk.file_offset + p.rank_offset);
        }
        
        if(this->uring_register_file) {
            sqe->flags |= IOSQE_FIXED_FILE;
        }

        // Normal operation for io_uring is to try and issue an sqe as
        // non-blocking first (do IO in the current thread without sleeping or waiting), 
        // and if that fails, execute it in an async manner (do IO in another thread). 
        // However, the non-blocking path will perform "inline memory copy" 
        // if the page cache is available under buffered IO (w/o O_DIRECT).
        // Such inline copy is time consuming and does not benefit from multithreading.
        // Thus we mark it as async, indicating that we prefer to do IO in another thread instead of inline.
        // This significantly improves the performance if the page cache is available.
        // Such optimization also applies to the last page of a chunk if 
        // the file size is not aligned to pages and the last-page IO has to be blocking.
        if(this->backend == Backend::URING_BUFFERED || unaligned_last_page) {
            sqe->flags |= IOSQE_ASYNC;
            // or io_uring_sqe_set_flags(sqe, sqe->flags | IOSQE_ASYNC)
        }
        // sqe->flags |= IOSQE_ASYNC;
        io_uring_sqe_set_data64(sqe, static_cast<uint64_t>(chunk_id));
        submit_cnt = 1;
    }
    this->chunks[chunk_id].extra_data.unfinished_cnt = submit_cnt;

    auto uring_func = [=]() {
        size_t submitted = 0;
        while(submitted < submit_cnt) {
            int ret = io_uring_submit(selected_ring);
            if(ret < 0) {
                throw std::runtime_error(
                    "io_uring_submit failed: " + std::string(strerror(-ret)));
            }
            submitted += ret;
            if(ret == 0) {
                fprintf(stderr, "io_uring_submit returned 0, chunk_id: %zu, submitted: %zu, submit_cnt: %zu\n", chunk_id, submitted, submit_cnt);
            }
        }
    };

    if(submit_cnt > 0) {
        uring_func();
    }

    // ── cuda_func: runs on cuda_thread ───────────────────────────────────────
    // Consumes CQEs, then launches DMA host->GPU.
    void  *rank_mid         = (char*)this->host_buffer + p.window_offset;
    void  *rank_dst         = p.rank_dst;
    void  *all_dst          = p.all_dst;
    size_t rank_size        = p.rank_size;
    size_t padded_rank_size = p.padded_rank_size;
    cudaEvent_t event       = p.event;

    auto cuda_func = [=]() {
        size_t &unfinished_cnt = this->chunks[chunk_id].extra_data.unfinished_cnt;
        while(unfinished_cnt > 0) {
            struct io_uring_cqe *cqe;
            int ret = io_uring_wait_cqe(selected_ring, &cqe);
            if (ret != 0) {
                throw std::runtime_error(
                    "io_uring_wait_cqe failed: " + std::string(strerror(-ret)));
            }
            chunk_id_t cqe_chunk_id = static_cast<chunk_id_t>(io_uring_cqe_get_data64(cqe));
            if (cqe->res < 0) {
                std::string msg =
                    "io_uring read error for chunk id: " + std::to_string(cqe_chunk_id) + ", error: " + std::string(strerror(-cqe->res));
                io_uring_cqe_seen(selected_ring, cqe);
                throw std::runtime_error(msg);
            }
            const Chunk &cqe_chunk = this->chunks[cqe_chunk_id];
            size_t padded_world_size = ROUND_UP(cqe_chunk.size, this->world_chunk_alignment);
            size_t padded_rank_size = padded_world_size / this->world_size;
            size_t logical_size = rank_logical_size(
                cqe_chunk.size, padded_rank_size * this->rank, padded_rank_size);
            if(static_cast<size_t>(cqe->res) < logical_size) {
                std::string msg =
                    "Unexpected io_uring short read: chunk_id=" + std::to_string(cqe_chunk_id)
                    + ", bytes_read=" + std::to_string(cqe->res)
                    + ", logical_size=" + std::to_string(logical_size);
                io_uring_cqe_seen(selected_ring, cqe);
                throw std::runtime_error(msg);
            }
            this->chunks[cqe_chunk_id].extra_data.unfinished_cnt --;
            io_uring_cqe_seen(selected_ring, cqe);
        }

        CUDA_CHECK(cudaMemcpyAsync(rank_dst, rank_mid, rank_size,
                                   cudaMemcpyHostToDevice, this->cuda_stream));
        CUDA_CHECK(cudaEventRecord(event, this->cuda_stream));
        if (this->world_size > 1) {
            CUDA_CHECK(cudaStreamWaitEvent(this->nccl_stream, event));
            NCCL_CHECK(ncclAllGather(rank_dst, all_dst, padded_rank_size,
                                     ncclInt8, this->group_communicator,
                                     this->nccl_stream));
            CUDA_CHECK(cudaEventRecord(event, this->nccl_stream));
        }
    };

    int cuda_req_id = this->next_executor_request_id();
    this->cuda_thread->submit(cuda_req_id, std::move(cuda_func));

    // ── wait_func: runs on wait_thread ───────────────────────────────────────
    auto wait_func = [=]() mutable {
        this->cuda_thread->reap(cuda_req_id);
        CUDA_CHECK(cudaEventSynchronize(event));
    };

    int completion_req_id = this->next_executor_request_id();
    this->wait_thread->submit(completion_req_id, std::move(wait_func));
    return ChunkRequest{this->wait_thread.get(), completion_req_id};
}

} // namespace instanttensor
