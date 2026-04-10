#include <instant_tensor/loader.hpp>
#include <liburing.h>

namespace instanttensor {

// ─── Threading model ─────────────────────────────────────────────────────────
// ALL io_uring calls (get_sqe, submit, wait_cqe, cqe_seen) run exclusively on
// uring_thread (an SPSCAsyncExecutor).  The main loader thread is the only
// producer; cuda_thread is the only consumer (via uring_thread->pop).
//
// This mirrors the existing aio_fallback_thread pattern and avoids any
// concurrent access to the liburing userspace state, which is not thread-safe.
//
// Within each chunk, the uring_func submits num_threads SQEs simultaneously
// and then waits for all their CQEs.  The kernel's io-wq dispatches each SQE
// to a separate worker thread, so all num_threads page-cache fills happen in
// parallel — unlike libaio, where io_submit serialises the iocbs through
// generic_file_read_iter one at a time.

// ─── file open / close ───────────────────────────────────────────────────────

void Loader::open_file_uring(FileInfo &f) {
    // Buffered fd — no O_DIRECT.  Page-cache reads are done by kernel io-wq
    // workers, which is what makes them truly async with io_uring.
    f.fd = ::open(f.filename.c_str(), O_RDONLY);
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
    posix_fadvise(f.fd, 0, 0, POSIX_FADV_SEQUENTIAL);

    this->need_uring       = true;
    this->need_host_buffer = true;
    this->need_cuda_thread = true;
}

void Loader::open_uring_context() {
    // Ring depth must cover io_depth × num_threads in-flight SQEs.
    // io_uring_queue_init rounds up internally, but we round up ourselves too
    // to ensure the ring can hold all in-flight entries without blocking.
    unsigned needed = (unsigned)(this->io_depth * this->num_threads);
    unsigned ring_entries = 1;
    while (ring_entries < needed) ring_entries <<= 1;
    this->uring_ring_size = ring_entries;

    int ret = io_uring_queue_init(ring_entries, &this->uring_ring, 0);
    if (ret < 0) {
        throw std::runtime_error(
            "io_uring_queue_init failed: " + std::string(strerror(-ret)));
    }
}

void Loader::close_file_uring(FileInfo &f) {
    ::close(f.fd);
}

void Loader::close_uring_context() {
    io_uring_queue_exit(&this->uring_ring);
}

// ─── chunk read ──────────────────────────────────────────────────────────────

// Helper struct capturing the per-segment read parameters, built on the main
// thread and moved into the uring_func lambda.
struct UringReadOp {
    int    fd;
    void  *buf;
    size_t size;
    size_t file_offset;
};

ChunkRequest Loader::post_read_chunk_uring(const ChunkIOParams &p) {
    // Build the per-thread read descriptors on the main thread (no io_uring
    // calls here — those happen exclusively on uring_thread below).
    std::vector<UringReadOp> ops;
    ops.reserve(this->num_threads);

    for (size_t i = 0; i < this->num_threads; i++) {
        size_t thread_offset = p.padded_thread_size * i;
        size_t thread_size   = std::min(
            (size_t)std::max((ssize_t)(p.chunk.size - p.rank_offset - thread_offset), (ssize_t)0),
            p.padded_thread_size);
        if (thread_size == 0) continue;

        ops.push_back(UringReadOp{
            p.file.fd,
            (char*)this->host_buffer + p.window_offset + thread_offset,
            thread_size,
            p.chunk.file_offset + p.rank_offset + thread_offset,
        });
    }

    // ── uring_func: runs on uring_thread ─────────────────────────────────────
    // Submits all SQEs then waits for exactly that many CQEs.
    // Because uring_thread is sequential and chunk N's SQEs are the only ones
    // in-flight at this point, every CQE must belong to chunk N — no cross-
    // chunk dispatch needed.
    auto uring_func = [this, ops = std::move(ops)]() {
        const size_t submit_cnt = ops.size();
        if (submit_cnt == 0) return;

        for (const auto &op : ops) {
            struct io_uring_sqe *sqe = io_uring_get_sqe(&this->uring_ring);
            if (!sqe) {
                throw std::runtime_error(
                    "io_uring SQ full (ring_entries=" +
                    std::to_string(this->uring_ring_size) + ")");
            }
            // Buffered read: no alignment constraints, use exact byte range.
            io_uring_prep_read(sqe, op.fd, op.buf,
                               (unsigned)op.size, op.file_offset);
        }

        int submitted = io_uring_submit(&this->uring_ring);
        if (submitted < 0) {
            throw std::runtime_error(
                "io_uring_submit failed: " + std::string(strerror(-submitted)));
        }

        // Wait for all submitted reads to complete.
        // io_uring_wait_cqe blocks until at least one CQE is available;
        // the kernel's io-wq workers are doing the actual I/O in parallel.
        for (size_t done = 0; done < submit_cnt; done++) {
            struct io_uring_cqe *cqe;
            int ret = io_uring_wait_cqe(&this->uring_ring, &cqe);
            if (ret < 0) {
                throw std::runtime_error(
                    "io_uring_wait_cqe failed: " + std::string(strerror(-ret)));
            }
            if (cqe->res < 0) {
                std::string msg =
                    "io_uring read error: " + std::string(strerror(-cqe->res));
                io_uring_cqe_seen(&this->uring_ring, cqe);
                throw std::runtime_error(msg);
            }
            io_uring_cqe_seen(&this->uring_ring, cqe);
        }
        // All reads for this chunk are complete; host_buffer is populated.
    };

    int uring_req_id = this->uring_thread->post(std::move(uring_func));

    // ── cuda_func: runs on cuda_thread ───────────────────────────────────────
    // Waits for uring_thread to finish (SPSC pop: cuda_thread is the sole
    // consumer of uring_thread's output queue), then DMA host→GPU.
    void  *rank_mid         = (char*)this->host_buffer + p.window_offset;
    void  *rank_dst         = p.rank_dst;
    void  *all_dst          = p.all_dst;
    size_t rank_size        = p.rank_size;
    size_t padded_rank_size = p.padded_rank_size;
    cudaEvent_t event       = p.event;

    auto cuda_func = [=]() {
        this->uring_thread->pop(uring_req_id);  // wait for I/O completion

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

    int cuda_req_id = this->cuda_thread->post(std::move(cuda_func));

    // ── wait_func: runs on wait_thread ───────────────────────────────────────
    auto wait_func = [=]() mutable {
        this->cuda_thread->pop(cuda_req_id);
        CUDA_CHECK(cudaEventSynchronize(event));
    };

    int completion_req_id = this->wait_thread->post(std::move(wait_func));
    return ChunkRequest{this->wait_thread.get(), completion_req_id};
}

} // namespace instanttensor
