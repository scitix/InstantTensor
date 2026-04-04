#include <instant_tensor/loader.hpp>
#include <instant_tensor/dlpack_utils.hpp>

namespace instanttensor {

struct RPCHandle {
    int latest_req_id;
    SPSCQueue<RPCRequest>* input_queue;
    SPSCQueue<RPCResponse>* output_queue;
};

class LoaderManager {
public:
    unordered_map<int, RPCHandle> loader_handle_to_queue;
    int loader_id_iter = 0;
    int device_idx = 0;

    int call(int remote_handle, int opcode, std::any args) {
        auto& handle = this->loader_handle_to_queue[remote_handle];
        ++handle.latest_req_id;
        int req_id = handle.latest_req_id;
        handle.input_queue->push(RPCRequest{req_id, opcode, std::move(args)});
        return req_id;
    }

    std::any get_result(int remote_handle, int req_id) {
        // NOTE: assume the requests are returned in order, otherwise we need to use a map to store the requests and responses
        auto& handle = this->loader_handle_to_queue[remote_handle];
        RPCResponse response;
        handle.output_queue->pop(response);
        while (response.id != req_id) {
            handle.output_queue->pop(response);
        }
        return response.result;
    }

    int open(const vector<string> &filenames,
            int device_idx,
            size_t process_group,
            size_t buffer_size,
            size_t chunk_size,
            size_t num_threads,
            size_t io_depth,
            const vector<pair<size_t, size_t>>& tensor_offsets)
    {
        this->device_idx = device_idx;

        int loader_handle = loader_id_iter++;
        // input_queue and output_queue are deleted by Loader
        SPSCQueue<RPCRequest> *input_queue = new SPSCQueue<RPCRequest>();
        SPSCQueue<RPCResponse> *output_queue = new SPSCQueue<RPCResponse>();
        this->loader_handle_to_queue[loader_handle] = RPCHandle{0, input_queue, output_queue};
        std::thread loader_thread(run_loader, std::unique_ptr<SPSCQueue<RPCRequest>>(input_queue), std::unique_ptr<SPSCQueue<RPCResponse>>(output_queue));
        loader_thread.detach();

        ncclComm_t nccl_group_communicator = nullptr;
        int rank = 0;
        int world_size = 1;
        if(!cuda_binding::init()) {
            print_and_throw(std::runtime_error(
                "cuda_binding: CUDA not found in process. "
                "Ensure torch (or another CUDA user) has been imported and CUDA is loaded."));
        }
        if(process_group != 0) {
            if (!nccl_binding::init()) {
                print_and_throw(std::runtime_error(
                    "nccl_binding: process_group non-zero but NCCL not found in process. "
                    "Ensure torch (or another NCCL user) has been imported and NCCL is loaded."));
            }
            nccl_group_communicator = reinterpret_cast<ncclComm_t>(process_group);
            NCCL_CHECK(ncclCommUserRank(nccl_group_communicator, &rank));
            NCCL_CHECK(ncclCommCount(nccl_group_communicator, &world_size));
        }

        int req_id = this->call(loader_handle, OPEN, std::make_any<OpenArgs>(filenames, device_idx, nccl_group_communicator, rank, world_size, buffer_size, chunk_size, num_threads, io_depth, tensor_offsets));
        this->get_result(loader_handle, req_id);
        return loader_handle;
    }

    void close(int loader_handle) {
        int req_id = this->call(loader_handle, CLOSE, std::make_any<CloseArgs>());
        this->get_result(loader_handle, req_id);
        this->loader_handle_to_queue.erase(loader_handle);
    }

    py::object get_dl_tensor(int loader_handle, int tensor_index, size_t size) {
        int req_id = this->call(loader_handle, GET_TENSOR_PTR, std::make_any<GetTensorArgs>(tensor_index));
        std::any data_ptr_any = this->get_result(loader_handle, req_id);
        void *data_ptr = std::any_cast<void*>(data_ptr_any);
        vector<int64_t> shape = {(int64_t)size};
        string dtype = "int8";
        auto ret = pack_dlpack((uint64_t)data_ptr, shape, dtype, this->device_idx);

        return ret;
    }
};


unique_ptr<LoaderManager> manager;

void init() {
    if(_env_use_cufile()) {
        cufile_context_initializer->initialize();
    }
    if(!manager) {
        manager = std::make_unique<LoaderManager>();
    }
}

int open(const vector<string> &filenames,
        int device_idx,
        size_t process_group,
        size_t buffer_size,
        size_t chunk_size,
        size_t num_threads,
        size_t io_depth,
        const vector<pair<size_t, size_t>>& tensor_offsets)
{
    if(!manager) {
        manager = std::make_unique<LoaderManager>();
    }
    return manager->open(filenames, device_idx, process_group, buffer_size, chunk_size, num_threads, io_depth, tensor_offsets);
}

void close(int loader_handle) {
    if(!manager) {
        print_and_throw(std::runtime_error("Internal error: manager is not initialized"));
    }
    manager->close(loader_handle);
}

py::object get_dl_tensor(int loader_handle, int tensor_index, size_t size) {
    if(!manager) {
        print_and_throw(std::runtime_error("Internal error: manager is not initialized"));
    }
    return manager->get_dl_tensor(loader_handle, tensor_index, size);
}

// Clean global cuda-related objects before python exit and cudaDeviceReset() is called
// to avoid cuda error when the program exits. This is registered by atexit.register() in safe_open_impl.py.
void cleanup() {
    instanttensor::cufile_context_initializer.reset();
    instanttensor::host_buffer_cache.reset();
    instanttensor::munmaper.reset();
}

} // namespace instanttensor


PYBIND11_MODULE(_C, m) {
    m.doc() = "InstantTensor C++ extension module";

    m.def("init", &instanttensor::init,
          "Initialize the cufile driver. This will be called lazily in open() if cufile is available for the specified file, "
          "but can be explicitly called to control the timing of initialization.");

    m.def("open", &instanttensor::open,
          "Open a safetensors file for loading",
          pybind11::arg("filenames"),
          pybind11::arg("device_idx"),
          pybind11::arg("process_group"),
          pybind11::arg("buffer_size"),
          pybind11::arg("chunk_size"),
          pybind11::arg("num_threads"),
          pybind11::arg("io_depth"),
          pybind11::arg("tensor_offsets"));

    m.def("close", &instanttensor::close,
          "Close the loader handle and cleanup resources",
          pybind11::arg("loader_handle"));

    m.def("get_dl_tensor", &instanttensor::get_dl_tensor,
          "Get a tensor by index from the opened file",
          pybind11::arg("loader_handle"),
          pybind11::arg("tensor_index"),
          pybind11::arg("size"));

    m.def("file_in_memory", &instanttensor::file_in_memory,
          "Check if the file is in memory",
          pybind11::arg("filenames"));

    m.def("cleanup", &instanttensor::cleanup,
          "Clean up global cuda-related objects before python exit and cudaDeviceReset() is called");
}
