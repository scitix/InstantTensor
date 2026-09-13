#include <cassert>
#include <cerrno>
#include <cstring>
#include <thread>

#include "../../csrc/loader_io_cufile.cpp"

namespace instanttensor {

// This harness exercises the backend with CPU workers and a mocked cuFile API.
Loader::Loader(unique_ptr<SPSCQueue<RPCRequest>> input,
               unique_ptr<SPSCQueue<RPCResponse>> output)
    : input_queue(std::move(input)), output_queue(std::move(output)) {}

int Loader::next_loader_task_id() { return loader_task_id++; }
int Loader::next_io_worker_task_id() { return io_worker_task_id++; }

} // namespace instanttensor

using namespace instanttensor;

struct ReadCase {
    vector<ssize_t> results;
    size_t index = 0;
    size_t completed = 0;
    std::thread::id caller;
    std::thread::id worker;
};

ssize_t mock_read(CUfileHandle_t handle, void *buffer, size_t size,
                  off_t file_offset, off_t device_offset) {
    auto &test = *static_cast<ReadCase*>(handle);
    assert(std::this_thread::get_id() != test.caller);
    if (test.index == 0) {
        test.worker = std::this_thread::get_id();
    }
    assert(test.worker == std::this_thread::get_id());
    assert(size == 10 - test.completed);
    assert(file_offset == static_cast<off_t>(4096 + test.completed));
    assert(device_offset == static_cast<off_t>(16 + test.completed));
    assert(test.index < test.results.size());
    ssize_t result = test.results[test.index++];
    if (result > 0 && static_cast<size_t>(result) <= size) {
        std::memset(static_cast<char*>(buffer) + device_offset, 42, result);
        test.completed += result;
    }
    errno = EIO;
    return result;
}

void check(vector<ssize_t> results, const string &expected_error = "", bool empty = false) {
    ReadCase test{std::move(results)};
    test.caller = std::this_thread::get_id();
    Loader loader(nullptr, nullptr);
    loader.worker_threads = std::make_unique<ThreadPoolTaskExecutor>(1);
    loader.io_thread = std::make_unique<IOExecutor>();
    char buffer[64] = {};
    loader.device_buffer = buffer;
    loader.chunks.resize(1);
    Chunk chunk{};
    chunk.file_offset = 4096;
    chunk.device_buffer_offset = 16;
    FileInfo file{};
    file.cufile_handle = &test;
    ChunkIOParams params{0, chunk, file, 4096, 0, empty ? 0U : 10U,
                         0, 0, buffer + 16, buffer, nullptr};
    auto request = loader.post_read_chunk_cufile(params);
    assert(request.loaded_to_device);
    string error;
    try {
        request.executor->reap(request.wait_handle);
    } catch (const std::runtime_error &exception) {
        error = exception.what();
    }
    if (expected_error.empty()) {
        assert(error.empty());
    } else {
        assert(error.find(expected_error) != string::npos);
    }
    loader.io_thread->join();
    loader.worker_threads->join();
    if (error.empty()) {
        assert(loader.chunks[0].extra_data.pending_worker_request_id == EXECUTOR_STOP_REQUEST_ID);
    }
    assert(loader.io_worker_task_id == (empty ? 0 : 1));
    assert(test.index == test.results.size());
    if (error.empty() && !empty) {
        for (size_t i = 16; i < 26; ++i) assert(buffer[i] == 42);
    }
}

int main() {
    cufile_binding::cuFileRead_fn = mock_read;
    check({3, 2, 5});
    check({10});
    check({3, 0}, "short read at EOF");
    check({3, -1}, "cuFileRead failed");
    check({3, -5}, "cuFile error code 5");
    check({11}, "too many bytes");
    check({}, "", true);
}
