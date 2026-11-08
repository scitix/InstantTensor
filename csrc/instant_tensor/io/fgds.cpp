#include <instant_tensor/loader.hpp>
#include <instant_tensor/dl_binding/fgds_binding.hpp>
#include <fcntl.h>
#include <sys/stat.h>

namespace instanttensor {

using namespace fgds_binding;

class FGDSBackend final : public WorkerIOBackend {
public:
    using WorkerIOBackend::WorkerIOBackend;
    void open_files() override;
    void close_files() override;
    void init_buffer() override;
    void destroy_buffer() override;
    IORequest post_read_chunk(const ChunkIOParams &p) override;
    void init_worker() override {
        CUDA_CHECK(cudaSetDevice(this->loader.device_idx));
    }

    bool context_initialized = false;
    void *device_staging_buffer = nullptr;
    void *device_staging_buffer_raw = nullptr;
    size_t device_staging_buffer_size = 0;
    void *fgds_target_addr = nullptr;
};

// ─── availability check ────────────────────────────────────────────────────

bool fgds_available() {
    return fgds_binding::init();
}

// ─── file open / close ─────────────────────────────────────────────────────

void FGDSBackend::open_files() {
    if (fgds_open(this->loader.device_idx) < 0) {
        throw std::runtime_error(
            "Failed to open fgds device for device " +
            std::to_string(this->loader.device_idx) +
            ". Ensure the fgds kernel module is loaded and /dev/fgds" +
            std::to_string(this->loader.device_idx) + " exists.");
    }
    this->context_initialized = true;
    for (auto &f : this->loader.file_info) {
        // fgds uses O_DIRECT so the kernel driver can DMA directly from storage.
        f.fd = ::open(f.filename.c_str(), O_RDONLY | O_DIRECT);
        if (f.fd < 0) {
            throw std::runtime_error("Failed to open file: " + f.filename);
        }
        struct stat st;
        if (fstat(f.fd, &st) < 0) {
            throw std::runtime_error("Failed to fstat file: " + f.filename);
        }
        f.size = st.st_size;
        debug_log("fgds open file %d done\n", f.fd);
    }
}

void FGDSBackend::close_files() {
    for (auto &f : this->loader.file_info) {
        ::close(f.fd);
    }
    if (this->context_initialized) {
        fgds_close(this->loader.device_idx);
        this->context_initialized = false;
    }
}

// ─── device buffer register / deregister ───────────────────────────────────
//
// FGDS uses O_DIRECT DMA into GPU memory.  The DMA target must be page-aligned
// (base address, offset, and transfer length all multiples of PAGE_SIZE).
// Because compute_layout aligns chunk_device_buffer_offset only to 16 bytes
// (first_tensor_alignment) for dtype alignment, we register a dedicated
// page-aligned *staging* buffer for FGDS reads and then D2D-copy to the final
// tensor location in cuda_func.

void FGDSBackend::init_buffer() {
    // DMA needs page-aligned destinations, while tensor offsets are only aligned
    // for their dtype. Read into per-window staging, then copy the logical bytes.
    this->device_staging_buffer_size = this->loader.io_depth * ROUND_UP(this->loader.rank_chunk_size, PAGE_SIZE);
    size_t alloc_size = this->device_staging_buffer_size + PAGE_SIZE - 1;
    CUDA_CHECK(cudaMalloc(&this->device_staging_buffer_raw, alloc_size));
    uintptr_t raw = reinterpret_cast<uintptr_t>(this->device_staging_buffer_raw);
    this->device_staging_buffer = reinterpret_cast<void*>(ROUND_UP(raw, (uintptr_t)PAGE_SIZE));
    void *target_addr = nullptr;
    int ret = fgds_regmem(this->loader.device_idx, this->device_staging_buffer,
                          this->device_staging_buffer_size, &target_addr);
    if (ret < 0) {
        throw std::runtime_error("fgds_regmem failed: " + std::to_string(ret));
    }
    debug_log("fgds regmem staging %p size %zu -> target %p\n",
              this->device_staging_buffer, this->device_staging_buffer_size, target_addr);
    this->fgds_target_addr = target_addr;
}

void FGDSBackend::destroy_buffer() {
    if (this->fgds_target_addr) {
        fgds_deregmem(this->loader.device_idx, this->device_staging_buffer,
                      this->device_staging_buffer_size);
        this->fgds_target_addr = nullptr;
        debug_log("fgds deregmem staging %p size %zu done\n",
                  this->device_staging_buffer, this->device_staging_buffer_size);
    }
    if (this->device_staging_buffer_raw) {
        CUDA_CHECK(cudaFree(this->device_staging_buffer_raw));
        this->device_staging_buffer_raw = nullptr;
        this->device_staging_buffer = nullptr;
    }
}

// ─── chunk read ─────────────────────────────────────────────────────────────
//
// Flow (mirrors the CUFILE backend, but DMA lands in a page-aligned staging
// buffer instead of the final device buffer):
//   1. io_thread:      submit the read task to worker_threads
//   2. worker_threads: fgds_read into device_staging_buffer (page-aligned DMA),
//                      then enqueue a D2D copy staging -> rank_dst on cuda_stream
//   3. cuda_executor:  after IO completion, record p.event on cuda_stream and
//                      optionally ncclAllGather (common launch in post_read_chunk)
//
// The D2D copy is enqueued by the worker before the common launch records the
// event on the same stream, so the event (and the NCCL stream wait with
// world_size > 1) is ordered after the copy. Hence loaded_to_device == true.

IORequest FGDSBackend::post_read_chunk(const ChunkIOParams &p) {
    chunk_id_t chunk_id = p.chunk_id;
    const ChunkIOState &state = p.chunk.io_state;
    size_t logical_size = state.rank_size;
    size_t rank_size_aligned = ROUND_UP(logical_size, this->loader.rank_alignment);
    size_t file_offset = state.rank_file_offset;
    size_t window_offset = state.window_offset;
    // Page-aligned DMA target window in the FGDS staging buffer: the staging
    // base is page-aligned and window_offset is a multiple of rank_chunk_size
    // (page-aligned), so buf_offset always satisfies DIRECT_IO alignment.
    void *staging_dst = (char *)this->device_staging_buffer + window_offset;
    void *rank_dst = p.rank_dst;
    int fd = p.file.fd;
    int io_req_id = this->loader.next_loader_task_id();

    auto io_start = [=]() {
        if (logical_size > 0) {
            fgds_fileid_t fid;
            fid.fd = fd;
            fid.deviceID = this->loader.device_idx;
            this->worker_threads->submit(io_req_id, [=]() {
                ssize_t ret = fgds_read(fid, this->device_staging_buffer,
                                        (off_t)window_offset, (ssize_t)rank_size_aligned,
                                        (off_t)file_offset);
                if (ret < 0) {
                    throw std::runtime_error("fgds_read failed for chunk " +
                        std::to_string(chunk_id) + ": " + FGDS_ERRSTR((int)ret));
                }
                if (static_cast<size_t>(ret) < logical_size) {
                    // A tail inside the last partial page at EOF is legitimate
                    // (the request is page-rounded); anything shorter is an error.
                    throw std::runtime_error(
                        "Unexpected fgds short read: chunk_id=" +
                        std::to_string(chunk_id) + ", bytes_read=" +
                        std::to_string(ret) + ", logical_size=" +
                        std::to_string(logical_size));
                }
                // Copy only the logical bytes; the alignment padding in the
                // staging window belongs to the next page read and is ignored.
                CUDA_CHECK(cudaMemcpyAsync(rank_dst, staging_dst, logical_size,
                                           cudaMemcpyDeviceToDevice, this->loader.cuda_stream));
            });
        } else {
            return IOSubmitStatus::Completed;
        }
        return IOSubmitStatus::Submitted;
    };
    this->loader.io_thread->submit(io_req_id, std::move(io_start));
    return IORequest{this->loader.io_thread.get(), io_req_id, true};
}

std::unique_ptr<IOBackend> make_fgds_backend(Loader &loader) {
    return std::make_unique<FGDSBackend>(loader);
}

} // namespace instanttensor
