#pragma once

#include <instant_tensor/types.hpp>

namespace instanttensor {

class Loader;

// One backend per Loader. Loader owns the common buffers, layout, and executors;
// the backend owns its IO context, registrations, and any helper workers.
class IOBackend {
public:
    explicit IOBackend(Loader &loader) : loader(loader) {}
    virtual ~IOBackend() = default;

    virtual void validate_config() const {}
    virtual bool needs_host_buffer() const { return false; }
    virtual void open_files() = 0;
    virtual void close_files() = 0;
    // Backend allocations/registrations, after files and common buffers are ready;
    // release registrations before freeing the common buffers.
    virtual void init_buffer() {}
    virtual void destroy_buffer() {}
    virtual void init_threads() {}
    virtual void destroy_threads() {}

    // post_read_chunk runs on the loader thread; poll and abort on the IO thread.
    // abort must finish accepted IO before publishing errors and releasing buffers.
    virtual IORequest post_read_chunk(const ChunkIOParams &params) = 0;
    virtual void poll(std::vector<IOCompletion> &completed) = 0;
    virtual void abort() = 0;

protected:
    // Advance read progress and report logical completion, retry, or error.
    IOCompletion process_read_result(chunk_id_t id, ssize_t result);
    Loader &loader;
};

// Synchronous APIs run in workers and share completion collection and draining.
class WorkerIOBackend : public IOBackend {
public:
    using IOBackend::IOBackend;
    std::unique_ptr<ThreadPoolTaskExecutor> worker_threads;

    void validate_config() const override;
    void init_threads() override;
    void destroy_threads() override;
    void poll(std::vector<IOCompletion> &completed) override;
    void abort() override;

protected:
    // CPU-only workers need no CUDA context; device backends override this.
    virtual void init_worker() {}
};

std::unique_ptr<IOBackend> make_io_backend(Backend backend, Loader &loader);
std::unique_ptr<IOBackend> make_aio_backend(Loader &loader);
std::unique_ptr<IOBackend> make_uring_backend(Loader &loader);
std::unique_ptr<IOBackend> make_mmap_backend(Loader &loader);
std::unique_ptr<IOBackend> make_cufile_backend(Loader &loader);
std::unique_ptr<IOBackend> make_fgds_backend(Loader &loader);

BackendStatus uring_status();
bool cufile_available();
bool fgds_available();

} // namespace instanttensor
