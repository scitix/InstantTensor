#include <instant_tensor/loader.hpp>

namespace instanttensor {

bool Loader::cufile_available(){// best-effort check
    return cufile_binding::init();
}

void Loader::open_file_cufile(FileInfo &f) {
    f.fd = ::open(f.filename.c_str(), O_RDONLY | O_DIRECT);
    if (f.fd < 0) {
        throw std::runtime_error("Failed to open file: " + f.filename);
    }
    struct stat st;
    if (fstat(f.fd, &st) < 0) { throw std::runtime_error("Failed to fstat file: " + f.filename); }
    f.size = st.st_size;

    this->need_worker_threads = true;

    cufile_context_initializer->initialize();

    CUfileDescr_t descr = {};
    descr.type = CU_FILE_HANDLE_TYPE_OPAQUE_FD;
    descr.handle.fd = f.fd;
    CUFILE_CHECK(cuFileHandleRegister(&f.cufile_handle, &descr));
    // Cannot close fd here since cuFileHandleRegister requires fd to be different in value (int),
    // and closing here cause the OS to reuse fd value.
}

void Loader::close_file_cufile(FileInfo &f) {
    cuFileHandleDeregister(f.cufile_handle);
    ::close(f.fd);
}

void Loader::register_device_buffer_cufile() {
    CUFILE_CHECK(cuFileBufRegister(this->device_buffer, this->buffer_size, 0));
}

void Loader::deregister_device_buffer_cufile() {
    CUFILE_CHECK(cuFileBufDeregister(this->device_buffer));
}

IORequest Loader::post_read_chunk_cufile(const ChunkIOParams &p) {
    chunk_id_t chunk_id = p.chunk_id;
    CUfileHandle_t handle = p.file.cufile_handle;
    size_t file_offset = p.chunk.io_state.rank_file_offset;
    size_t device_offset = p.chunk.device_buffer_offset + (file_offset - p.chunk.file_offset);
    size_t logical_size = p.chunk.io_state.rank_size;
    int io_req_id = this->next_loader_task_id();
    auto io_start = [=]() {
        if (logical_size > 0) {
            this->worker_threads->submit(io_req_id, [=]() {
                size_t bytes_completed = 0;
                while (bytes_completed < logical_size) {
                    size_t remaining = logical_size - bytes_completed;
                    // cuFile supports unaligned ranges on O_DIRECT files, including retry tails.
                    ssize_t ret = cuFileRead(handle, this->device_buffer, remaining,
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
    this->io_thread->submit(io_req_id, std::move(io_start));
    return IORequest{this->io_thread.get(), io_req_id, true};
}

} // namespace instanttensor
