#include <instant_tensor/loader.hpp>
#include <cerrno>
#include <cstring>

namespace instanttensor {

void cleanup_cufile();
void cleanup_mmap();

void cleanup_io_backends() {
    cleanup_cufile();
    cleanup_mmap();
}

std::unique_ptr<IOBackend> make_io_backend(Backend backend, Loader &loader) {
    switch (backend) {
        case Backend::AIO:
        case Backend::AIO_BUFFERED: return make_aio_backend(loader);
        case Backend::URING:
        case Backend::URING_BUFFERED: return make_uring_backend(loader);
        case Backend::MMAP: return make_mmap_backend(loader);
        case Backend::CUFILE: return make_cufile_backend(loader);
        case Backend::FGDS: return make_fgds_backend(loader);
        default: throw std::invalid_argument("Unsupported backend: " + std::to_string(backend));
    }
}

void WorkerIOBackend::validate_config() const {
    if (this->loader.concurrency == 0) {
        throw std::invalid_argument("concurrency must be greater than zero");
    }
}

void WorkerIOBackend::init_threads() {
    this->worker_threads = std::make_unique<ThreadPoolTaskExecutor>(this->loader.concurrency,
        [this]() {
            this->init_worker();
            return std::make_unique<FunctionWorkerDriver>();
        });
}

void WorkerIOBackend::destroy_threads() {
    if (this->worker_threads) this->worker_threads->join();
}

void WorkerIOBackend::poll(std::vector<IOCompletion> &completed) {
    if (!this->worker_threads) return;
    ThreadPoolTaskExecutor::ResultItem result;
    while (this->worker_threads->try_reap_any(result)) {
        std::exception_ptr error;
        if (result.value.type() == typeid(std::exception_ptr)) {
            error = std::any_cast<std::exception_ptr>(result.value);
        }
        completed.push_back({result.request_id, error});
    }
}

void WorkerIOBackend::abort() {
    this->destroy_threads();
}

IOCompletion IOBackend::process_read_result(chunk_id_t id, ssize_t result) {
    Chunk &chunk = this->loader.chunks[id];
    ChunkIOState &state = chunk.io_state;
    IOCompletion completion{state.io_request_id, {}};
    try {
        if (result == -EAGAIN || result == -EINTR) {
            completion.retry = true;
            return completion;
        }
        std::string name = backend_to_string(this->loader.backend);
        if (result < 0) {
            throw std::runtime_error(name + " read error for chunk id: " +
                std::to_string(id) + ", error: " + strerror(-result));
        }
        size_t read_offset = ROUND_DOWN(state.bytes_completed, this->loader.rank_alignment);
        size_t covered = read_offset + static_cast<size_t>(result);
        state.bytes_completed = std::min(state.rank_size,
            std::max(state.bytes_completed, covered));
        if (state.bytes_completed >= state.rank_size) {
            return completion;
        }
        completion.retry = true;
    } catch (...) {
        completion.error = std::current_exception();
    }
    return completion;
}

} // namespace instanttensor
