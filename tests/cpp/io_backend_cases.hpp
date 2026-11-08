#pragma once

// Included after the concrete implementations in test_io_polling.cpp.
// All device operations below use the existing dynamic-library bindings.
namespace {
std::atomic<int> worker_device_sets{0};

void check_worker_device_setup() {
    auto saved_device = cuda_binding::cudaSetDevice_fn;
    cuda_binding::cudaSetDevice_fn = [](int device) {
        assert(device == 7);
        ++worker_device_sets;
        return cudaSuccess;
    };
    for (Backend kind : {Backend::AIO, Backend::MMAP, Backend::CUFILE, Backend::FGDS}) {
        Loader loader(nullptr, nullptr);
        loader.backend = kind;
        loader.device_idx = 7;
        loader.concurrency = 2;
        loader.io_backend = make_io_backend(kind, loader);
        worker_device_sets = 0;
        loader.io_backend->init_threads();
        loader.io_backend->destroy_threads();
        assert(worker_device_sets == ((kind == Backend::CUFILE || kind == Backend::FGDS) ? 2 : 0));
    }
    cuda_binding::cudaSetDevice_fn = saved_device;
}

void check_backend_factory() {
    Loader loader(nullptr, nullptr);
    for (Backend kind : {Backend::AIO, Backend::AIO_BUFFERED, Backend::URING,
                         Backend::URING_BUFFERED, Backend::MMAP, Backend::CUFILE,
                         Backend::FGDS}) {
        loader.backend = kind;
        auto backend = make_io_backend(kind, loader);
        bool worker = false;
        switch (kind) {
            case Backend::AIO:
            case Backend::AIO_BUFFERED:
                assert(dynamic_cast<AIOBackend*>(backend.get()));
                assert(backend->needs_host_buffer());
                break;
            case Backend::URING:
            case Backend::URING_BUFFERED:
                assert(dynamic_cast<UringBackend*>(backend.get()));
                assert(backend->needs_host_buffer());
                break;
            case Backend::MMAP: {
                auto* mmap_backend = dynamic_cast<MmapBackend*>(backend.get());
                assert(mmap_backend);
                const char* setting = std::getenv("INSTANTTENSOR_USE_INTERNAL_MEMORY_REGISTER");
                bool registered = setting && std::string(setting) == "1";
                assert(mmap_backend->use_internal_memory_register == registered);
                assert(backend->needs_host_buffer() == !registered);
                worker = true;
                break;
            }
            case Backend::CUFILE:
                assert(dynamic_cast<CuFileBackend*>(backend.get()));
                assert(!backend->needs_host_buffer());
                worker = true;
                break;
            case Backend::FGDS:
                assert(dynamic_cast<FGDSBackend*>(backend.get()));
                assert(!backend->needs_host_buffer());
                worker = true;
                break;
        }
        loader.concurrency = 0;
        bool rejected = false;
        try { backend->validate_config(); }
        catch (const std::invalid_argument&) { rejected = true; }
        assert(rejected == worker);
        loader.concurrency = 2;
        backend->validate_config();
    }
    bool rejected = false;
    try { make_io_backend(static_cast<Backend>(7), loader); }
    catch (const std::invalid_argument&) { rejected = true; }
    assert(rejected);
}

void check_failed_open_cleanup() {
    for (Backend kind : {Backend::AIO, Backend::AIO_BUFFERED,
                         Backend::URING, Backend::URING_BUFFERED,
                         static_cast<Backend>(7)}) {
        for (bool bad_chunk : {true, false}) {
            Loader loader(nullptr, nullptr);
            OpenArgs args({}, 0, nullptr, 0, 1, 65536,
                          bad_chunk ? 0 : 4096, 0, bad_chunk ? 4 : 0, kind, {}, {});
            bool rejected = false;
            try { loader.open(args); }
            catch (const std::invalid_argument&) { rejected = true; }
            assert(rejected);
            assert(!loader.device_buffer_raw && !loader.host_buffer_entry.ptr);
            assert(!loader.host_buffer_entry.deleter);
            assert(!loader.io_thread);
            assert(bool(loader.io_backend) == (kind != static_cast<Backend>(7)));
            // Neither an empty deleter nor uninitialized backend APIs may run.
            loader.close(CloseArgs{});
            loader.close(CloseArgs{});
            assert(!loader.host_buffer && !loader.host_buffer_entry.ptr);
        }
    }
}

class LifetimeBackend : public IOBackend {
public:
    LifetimeBackend(Loader& loader, std::weak_ptr<int> callback_token,
                    bool& destroyed, int& joins)
        : IOBackend(loader), callback_token(callback_token), destroyed(destroyed), joins(joins) {}
    ~LifetimeBackend() override {
        // Both the IO worker and its stored callback factory must be gone.
        assert(this->callback_token.expired());
        assert(this->completed);
        this->destroyed = true;
    }
    void open_files() override {}
    void close_files() override {}
    IORequest post_read_chunk(const ChunkIOParams&) override { throw std::logic_error("unused"); }
    void poll(std::vector<IOCompletion>& output) override {
        if (this->pending) {
            output.push_back({17, {}});
            this->pending = false;
            this->completed = true;
        }
    }
    void abort() override { assert(false); }
    void destroy_threads() override {
        assert(this->loader.io_thread->ready(17));
        ++this->joins;
    }
    bool pending = false;
private:
    std::weak_ptr<int> callback_token;
    bool& destroyed;
    int& joins;
    bool completed = false;
};

void check_backend_lifetime() {
    for (bool explicit_join : {false, true}) {
        bool destroyed = false;
        int joins = 0;
        {
            Loader loader(nullptr, nullptr);
            auto token = std::make_shared<int>(0);
            auto backend = std::make_unique<LifetimeBackend>(loader, token, destroyed, joins);
            auto* ptr = backend.get();
            loader.io_backend = std::move(backend);
            loader.io_thread = std::make_unique<IOExecutor>(
                [ptr, token](auto& output) { ptr->poll(output); },
                [ptr]() { ptr->abort(); });
            token.reset();
            loader.io_thread->submit(17, [ptr]() {
                ptr->pending = true;
                return IOSubmitStatus::Submitted;
            });
            if (explicit_join) loader.destroy_threads();
            assert(!destroyed);
        }
        assert(destroyed && joins == (explicit_join ? 1 : 0));
    }
}

class BufferOrderBackend : public IOBackend {
public:
    BufferOrderBackend(Loader& loader, bool& released) : IOBackend(loader), released(released) {}
    bool needs_host_buffer() const override { return true; }
    void open_files() override {}
    void close_files() override {}
    IORequest post_read_chunk(const ChunkIOParams&) override { throw std::logic_error("unused"); }
    void poll(std::vector<IOCompletion>&) override {}
    void abort() override {}
    void destroy_buffer() override {
        if (!this->released) {
            assert(this->loader.device_buffer_raw && this->loader.host_buffer_entry.ptr);
            assert(this->loader.host_buffer == this->loader.host_buffer_entry.ptr);
            this->released = true;
        }
    }
private:
    bool& released;
};

bool buffer_backend_released = false;
int device_frees = 0;
void check_buffer_release_order() {
    Loader loader(nullptr, nullptr);
    loader.io_backend = std::make_unique<BufferOrderBackend>(loader, buffer_backend_released);
    char device[64], host[64];
    loader.device_buffer_raw = device;
    loader.device_buffer = device + 16;
    loader.host_buffer = host;
    int host_frees = 0;
    loader.host_buffer_entry = {host, sizeof(host), [&](void* ptr) {
        assert(buffer_backend_released && ptr == host);
        ++host_frees;
    }};
    auto saved_free = cuda_binding::cudaFree_fn;
    cuda_binding::cudaFree_fn = [](void*) {
        assert(buffer_backend_released);
        ++device_frees;
        return cudaSuccess;
    };
    loader.destroy_buffer();
    assert(device_frees == 1);
    assert(!loader.device_buffer_raw && !loader.device_buffer);
    assert(!loader.host_buffer && !loader.host_buffer_entry.ptr);
    assert(!loader.host_buffer_entry.deleter && loader.host_buffer_entry.size == 0);
    loader.destroy_buffer();
    assert(device_frees == 1);
    if (_env_cache_buffer()) {
        assert(host_frees == 0);
        auto cached = host_buffer_cache->get(sizeof(host));
        assert(cached.ptr == host);
        cached.deleter(cached.ptr);
    }
    assert(host_frees == 1);
    cuda_binding::cudaFree_fn = saved_free;
}

struct FGDSCase {
    std::vector<char> allocation;
    void* raw = nullptr;
    void* registered = nullptr;
    void* destination = nullptr;
    ssize_t result = 0;
    size_t logical = 4097;
    int opens = 0, reads = 0, copies = 0, frees = 0, closes = 0;
    bool deregistered = false;
    std::thread::id caller = std::this_thread::get_id();
};
FGDSCase* fgds_case = nullptr;

void check_fgds_backend() {
    auto saved_malloc = cuda_binding::cudaMalloc_fn;
    auto saved_free = cuda_binding::cudaFree_fn;
    auto saved_copy = cuda_binding::cudaMemcpyAsync_fn;
    auto saved_device = cuda_binding::cudaSetDevice_fn;
    auto saved_reg = fgds_binding::fgds_regmem_fn;
    auto saved_dereg = fgds_binding::fgds_deregmem_fn;
    auto saved_read = fgds_binding::fgds_read_fn;
    auto saved_open = fgds_binding::fgds_open_fn;
    auto saved_close = fgds_binding::fgds_close_fn;
    fgds_binding::fgds_open_fn = [](int device) {
        assert(device == 7 && fgds_case->opens == 0);
        ++fgds_case->opens;
        return 0;
    };
    cuda_binding::cudaSetDevice_fn = [](int device) { assert(device == 7); return cudaSuccess; };
    cuda_binding::cudaMalloc_fn = [](void** ptr, size_t size) {
        assert(size == 16384 + PAGE_SIZE - 1);
        fgds_case->allocation.resize(size + 1);
        *ptr = fgds_case->raw = fgds_case->allocation.data() + 1;
        return cudaSuccess;
    };
    fgds_binding::fgds_regmem_fn = [](int device, const void* ptr, size_t size, void** target) {
        assert(device == 7 && size == 16384 && fgds_case->opens == 1);
        assert(reinterpret_cast<uintptr_t>(ptr) % PAGE_SIZE == 0);
        fgds_case->registered = *target = const_cast<void*>(ptr);
        return 0;
    };
    fgds_binding::fgds_deregmem_fn = [](int device, const void* ptr, size_t size) {
        assert(device == 7 && ptr == fgds_case->registered && size == 16384);
        assert(!fgds_case->deregistered && fgds_case->frees == 0);
        fgds_case->deregistered = true;
        return 0;
    };
    cuda_binding::cudaFree_fn = [](void* ptr) {
        assert(fgds_case->deregistered && ptr == fgds_case->raw);
        ++fgds_case->frees;
        return cudaSuccess;
    };
    fgds_binding::fgds_close_fn = [](int device) {
        assert(device == 7 && fgds_case->frees == 1);
        ++fgds_case->closes;
        return 0;
    };
    fgds_binding::fgds_read_fn = [](fgds_binding::fgds_fileid_t file, void* ptr,
                                   off_t offset, ssize_t size, off_t file_offset) -> ssize_t {
        assert(std::this_thread::get_id() != fgds_case->caller);
        assert(file.fd == 123 && file.deviceID == 7);
        assert(ptr == fgds_case->registered && offset == 8192);
        assert(size == 8192 && file_offset == 12288);
        ++fgds_case->reads;
        if (fgds_case->result > 0) std::memset(static_cast<char*>(ptr) + offset, 42, fgds_case->result);
        return fgds_case->result;
    };
    cuda_binding::cudaMemcpyAsync_fn = [](void* dst, const void* src, size_t size,
                                        cudaMemcpyKind kind, cudaStream_t) {
        assert(std::this_thread::get_id() != fgds_case->caller);
        assert(kind == cudaMemcpyDeviceToDevice && size == fgds_case->logical);
        assert(dst == fgds_case->destination);
        assert(src == static_cast<char*>(fgds_case->registered) + 8192);
        ++fgds_case->copies;
        std::memcpy(dst, src, size);
        return cudaSuccess;
    };
    for (ssize_t result : {8192, 4097, 4096, -5, 0}) {
        FGDSCase test;
        fgds_case = &test;
        test.result = result;
        test.logical = result == 0 ? 0 : 4097;
        std::vector<char> destination(4129, 0);
        test.destination = destination.data() + 16;
        Loader loader(nullptr, nullptr);
        loader.backend = Backend::FGDS;
        loader.device_idx = 7;
        loader.concurrency = 1;
        loader.rank_alignment = 4096;
        loader.rank_chunk_size = 8192;
        loader.io_depth = 2;
        loader.io_backend = make_io_backend(Backend::FGDS, loader);
        loader.io_backend->open_files();
        loader.io_backend->init_buffer();
        loader.io_backend->init_threads();
        loader.io_thread = std::make_unique<IOExecutor>(
            [&](auto& output) { loader.io_backend->poll(output); },
            [&]() { loader.io_backend->abort(); });
        Chunk chunk{};
        chunk.io_state.rank_size = test.logical;
        chunk.io_state.rank_file_offset = 12288;
        chunk.io_state.window_offset = 8192;
        FileInfo file{};
        file.fd = 123;
        ChunkIOParams params{5, chunk, file, 1, test.destination, destination.data(), nullptr};
        auto request = loader.io_backend->post_read_chunk(params);
        assert(request.loaded_to_device);
        std::string error;
        try { request.executor->reap(request.wait_handle); }
        catch (const std::runtime_error& e) { error = e.what(); }
        loader.destroy_threads();
        if (result == -5) assert(error.find("fgds_read failed") != std::string::npos);
        else if (result == 4096) assert(error.find("short read") != std::string::npos);
        else assert(error.empty());
        bool copied = result >= 4097;
        assert(test.reads == (test.logical ? 1 : 0));
        assert(test.copies == (copied ? 1 : 0));
        for (size_t i = 0; i < destination.size(); ++i) {
            assert(destination[i] == (copied && i >= 16 && i < 4113 ? 42 : 0));
        }
        loader.destroy_buffer();
        loader.destroy_buffer();
        loader.io_backend->close_files();
        loader.io_backend->close_files();
        assert(test.deregistered && test.frees == 1 && test.opens == 1 && test.closes == 1);
    }
    fgds_case = nullptr;
    cuda_binding::cudaMalloc_fn = saved_malloc;
    cuda_binding::cudaFree_fn = saved_free;
    cuda_binding::cudaMemcpyAsync_fn = saved_copy;
    cuda_binding::cudaSetDevice_fn = saved_device;
    fgds_binding::fgds_regmem_fn = saved_reg;
    fgds_binding::fgds_deregmem_fn = saved_dereg;
    fgds_binding::fgds_read_fn = saved_read;
    fgds_binding::fgds_open_fn = saved_open;
    fgds_binding::fgds_close_fn = saved_close;
}
} // namespace
