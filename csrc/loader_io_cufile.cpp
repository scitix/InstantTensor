#include <instant_tensor/loader.hpp>

namespace instanttensor {

class CuFileBackend final : public WorkerIOBackend {
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
};

bool cufile_available(){// best-effort check
    return cufile_binding::init();
}

void CuFileBackend::open_files() {
    cufile_context_initializer->initialize();
    for (auto &f : this->loader.file_info) {
        f.fd = ::open(f.filename.c_str(), O_RDONLY | O_DIRECT);
        if (f.fd < 0) {
            throw std::runtime_error("Failed to open file: " + f.filename);
        }
        struct stat st;
        if (fstat(f.fd, &st) < 0) { throw std::runtime_error("Failed to fstat file: " + f.filename); }
        f.size = st.st_size;
        CUfileDescr_t descr = {};
        descr.type = CU_FILE_HANDLE_TYPE_OPAQUE_FD;
        descr.handle.fd = f.fd;
        CUFILE_CHECK(cuFileHandleRegister(&f.cufile_handle, &descr));
        // Cannot close fd here since cuFileHandleRegister requires fd to be different in value (int),
        // and closing here cause the OS to reuse fd value.
    }
}

void CuFileBackend::close_files() {
    for (auto &f : this->loader.file_info) {
        cuFileHandleDeregister(f.cufile_handle);
        ::close(f.fd);
    }
}

void CuFileBackend::init_buffer() {
    CUFILE_CHECK(cuFileBufRegister(this->loader.device_buffer, this->loader.buffer_size, 0));
}

void CuFileBackend::destroy_buffer() {
    CUFILE_CHECK(cuFileBufDeregister(this->loader.device_buffer));
}

IORequest CuFileBackend::post_read_chunk(const ChunkIOParams &p) {
    chunk_id_t chunk_id = p.chunk_id;
    CUfileHandle_t handle = p.file.cufile_handle;
    size_t file_offset = p.chunk.io_state.rank_file_offset;
    size_t device_offset = p.chunk.device_buffer_offset + (file_offset - p.chunk.file_offset);
    size_t logical_size = p.chunk.io_state.rank_size;
    int io_req_id = this->loader.next_loader_task_id();
    auto io_start = [=]() {
        if (logical_size > 0) {
            this->worker_threads->submit(io_req_id, [=]() {
                size_t bytes_completed = 0;
                while (bytes_completed < logical_size) {
                    size_t remaining = logical_size - bytes_completed;
                    // cuFile supports unaligned ranges on O_DIRECT files, including retry tails.
                    ssize_t ret = cuFileRead(handle, this->loader.device_buffer, remaining,
                        file_offset + bytes_completed, device_offset + bytes_completed);
                    if (ret < 0) {
                        std::string detail = ret == -1 ? std::strerror(errno) : CUFILE_ERRSTR(-ret);
                        throw std::runtime_error("cuFileRead failed: " + detail);
                    }
                    if (ret == 0) {
                        throw std::runtime_error(
                            "Unexpected cuFile short read at EOF: chunk_id=" +
                            std::to_string(chunk_id) + ", bytes_completed=" +
                            std::to_string(bytes_completed) + ", logical_size=" +
                            std::to_string(logical_size));
                    }
                    if (static_cast<size_t>(ret) > remaining) {
                        throw std::runtime_error("cuFileRead returned too many bytes");
                    }
                    bytes_completed += static_cast<size_t>(ret);
                }
            });
        } else {
            return IOSubmitStatus::Completed;
        }
        return IOSubmitStatus::Submitted;
    };
    this->loader.io_thread->submit(io_req_id, std::move(io_start));
    return IORequest{this->loader.io_thread.get(), io_req_id, true};
}

std::unique_ptr<IOBackend> make_cufile_backend(Loader &loader) {
    return std::make_unique<CuFileBackend>(loader);
}

} // namespace instanttensor
