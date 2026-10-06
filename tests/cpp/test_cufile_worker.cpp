#include <cassert>
#include <atomic>
#include <cerrno>
#include <cstring>
#include <thread>

#include "../../csrc/loader_common.cpp"
#include "../../csrc/loader_io_cufile.cpp"

// This harness exercises the backend with CPU workers and a mocked cuFile API.
using namespace instanttensor;

struct ReadCase {
    vector<ssize_t> results;
    size_t index = 0;
    size_t completed = 0;
    std::thread::id caller;
    std::thread::id worker;
    std::atomic<bool> *release = nullptr;
    std::atomic<bool> *entered = nullptr;
};

ssize_t mock_read(CUfileHandle_t handle, void *buffer, size_t size,
                  off_t file_offset, off_t device_offset) {
    auto &test = *static_cast<ReadCase*>(handle);
    assert(std::this_thread::get_id() != test.caller);
    if (test.index == 0) {
        test.worker = std::this_thread::get_id();
    }
    if (test.entered) *test.entered = true;
    while (test.release && !test.release->load()) std::this_thread::yield();
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
    loader.io_thread = std::make_unique<IOExecutor>(
        [&](auto& output) { loader.poll_worker_completions(output); },
        [&]() { loader.worker_threads->join(); });
    char buffer[64] = {};
    loader.device_buffer = buffer;
    loader.chunks.resize(1);
    Chunk chunk{};
    chunk.file_offset = 4096;
    chunk.device_buffer_offset = 16;
    chunk.io_state.rank_file_offset = 4096;
    chunk.io_state.rank_size = empty ? 0 : 10;
    FileInfo file{};
    file.cufile_handle = &test;
    ChunkIOParams params{0, chunk, file,
                         0, buffer + 16, buffer, nullptr};
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
    ThreadPoolTaskExecutor::ResultItem remaining;
    assert(!loader.worker_threads->try_reap_any(remaining));
    assert(test.index == test.results.size());
    if (error.empty() && !empty) {
        for (size_t i = 16; i < 26; ++i) assert(buffer[i] == 42);
    }
}

void check_out_of_order_error_and_drain() {
    std::atomic<bool> release{false}, entered{false};
    ReadCase first{{3, 7}}, second{{-1}};
    first.caller = second.caller = std::this_thread::get_id();
    first.release = &release;
    first.entered = &entered;
    Loader loader(nullptr, nullptr);
    loader.worker_threads = std::make_unique<ThreadPoolTaskExecutor>(2);
    loader.io_thread = std::make_unique<IOExecutor>(
        [&](auto& output) { loader.poll_worker_completions(output); },
        [&]() { loader.worker_threads->join(); });
    char buffer[64] = {};
    loader.device_buffer = buffer;
    Chunk chunk{};
    chunk.file_offset = 4096;
    chunk.device_buffer_offset = 16;
    chunk.io_state.rank_file_offset = 4096;
    chunk.io_state.rank_size = 10;
    FileInfo file{};
    file.cufile_handle = &first;
    ChunkIOParams params{0, chunk, file, 0,
                         buffer + 16, buffer, nullptr};
    auto first_request = loader.post_read_chunk_cufile(params);
    file.cufile_handle = &second;
    params.chunk_id = 1;
    auto second_request = loader.post_read_chunk_cufile(params);
    while (!entered) std::this_thread::yield();
    bool failed = false;
    try {
        second_request.executor->reap(second_request.wait_handle);
    } catch (const std::runtime_error &error) {
        failed = std::string(error.what()).find("cuFileRead failed") != std::string::npos;
    }
    assert(failed);
    std::any ignored;
    assert(!first_request.executor->try_reap(first_request.wait_handle, ignored));
    loader.io_thread->stop();
    release = true;
    loader.io_thread->join();
    first_request.executor->reap(first_request.wait_handle);
    loader.worker_threads->join();
    assert(first.completed == 10 && first.index == 2 && second.index == 1);
    for (size_t i = 16; i < 26; ++i) assert(buffer[i] == 42);
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
    check_out_of_order_error_and_drain();
}
