#include <atomic>
#include <cassert>
#include <chrono>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <unordered_set>
#include <vector>

#include <instant_tensor/cuda_executor.hpp>

using namespace instanttensor;

namespace {

std::atomic<int> event_status[4];
std::mutex thread_mutex;
std::thread::id cuda_thread_id;

cudaError_t set_device(int) {
    std::lock_guard<std::mutex> lock(thread_mutex);
    cuda_thread_id = std::this_thread::get_id();
    return cudaSuccess;
}

std::thread::id get_cuda_thread_id() {
    std::lock_guard<std::mutex> lock(thread_mutex);
    return cuda_thread_id;
}

cudaError_t query_event(cudaEvent_t event) {
    size_t index = reinterpret_cast<size_t>(event) - 1;
    return static_cast<cudaError_t>(event_status[index].load());
}

const char* error_string(cudaError_t) {
    return "test CUDA error";
}

template<typename Predicate>
void wait_until(const char* name, Predicate predicate) {
    auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(2);
    bool ready = predicate();
    while (!ready && std::chrono::steady_clock::now() < deadline) {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
        ready = predicate();
    }
    if (!ready) {
        throw std::runtime_error(std::string("timed out waiting for ") + name);
    }
}

struct MockIOExecutor {
    std::mutex mutex;
    std::unordered_set<int> started;
    std::vector<IOCompletion> completed;
    IOExecutor executor{[this](auto& output) {
        std::lock_guard<std::mutex> lock(mutex);
        output.swap(completed);
    }, [] {}};

    IORequest submit(int id, bool ready = true, std::exception_ptr error = {}) {
        executor.submit(id, [this, id, ready, error]() {
            std::lock_guard<std::mutex> lock(mutex);
            started.insert(id);
            if (ready) completed.push_back({id, error});
            return IOSubmitStatus::Submitted;
        });
        return IORequest{&executor, id, false};
    }

    void complete(int id) {
        wait_until("IO start", [&]() {
            std::lock_guard<std::mutex> lock(mutex);
            return started.count(id) != 0;
        });
        std::lock_guard<std::mutex> lock(mutex);
        completed.push_back({id, {}});
    }

    void join() { executor.join(); }
};

void test_ordered_launch_and_out_of_order_completion() {
    MockIOExecutor io;
    CUDAExecutor cuda(0);
    std::mutex launch_mutex;
    std::vector<int> launch_order;
    std::vector<std::thread::id> launch_threads;

    event_status[0] = cudaErrorNotReady;
    event_status[1] = cudaErrorNotReady;
    cuda.submit(0, CUDAOperation{
        io.submit(10, false),
        [&]() {
            std::lock_guard<std::mutex> lock(launch_mutex);
            launch_order.push_back(0);
            launch_threads.push_back(std::this_thread::get_id());
        },
        reinterpret_cast<cudaEvent_t>(1),
    });
    cuda.submit(1, CUDAOperation{
        io.submit(11, false),
        [&]() {
            std::lock_guard<std::mutex> lock(launch_mutex);
            launch_order.push_back(1);
            launch_threads.push_back(std::this_thread::get_id());
        },
        reinterpret_cast<cudaEvent_t>(2),
    });

    io.complete(11);
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
    {
        std::lock_guard<std::mutex> lock(launch_mutex);
        assert(launch_order.empty());
    }

    io.complete(10);
    wait_until("ordered launches", [&]() {
        std::lock_guard<std::mutex> lock(launch_mutex);
        return launch_order.size() == 2;
    });
    {
        std::lock_guard<std::mutex> lock(launch_mutex);
        assert((launch_order == std::vector<int>{0, 1}));
        std::thread::id worker_thread_id = get_cuda_thread_id();
        assert(launch_threads[0] == worker_thread_id);
        assert(launch_threads[1] == worker_thread_id);
        assert(launch_threads[0] != std::this_thread::get_id());
    }

    event_status[1] = 0;
    std::any result;
    wait_until("second completion", [&]() { return cuda.try_reap(1, result); });
    event_status[0] = 0;
    cuda.reap(0);

    cuda.join();
    io.join();
}

void test_not_ready() {
    MockIOExecutor io;
    CUDAExecutor cuda(0);
    event_status[2] = cudaErrorNotReady;
    cuda.submit(2, CUDAOperation{
        io.submit(12),
        [] {},
        reinterpret_cast<cudaEvent_t>(3),
    });

    std::any result;
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
    assert(!cuda.try_reap(2, result));
    event_status[2] = cudaSuccess;
    cuda.reap(2);

    cuda.join();
    io.join();
}

void test_error_propagation() {
    MockIOExecutor io;
    CUDAExecutor cuda(0);
    cuda.submit(3, CUDAOperation{
        io.submit(13, true, std::make_exception_ptr(std::runtime_error("I/O failure"))),
        [] {},
        reinterpret_cast<cudaEvent_t>(4),
    });

    bool threw = false;
    try {
        cuda.reap(3);
    }
    catch (const std::runtime_error& error) {
        threw = std::string(error.what()) == "I/O failure";
    }
    assert(threw);

    cuda.join();
    io.join();
}

void test_io_failure_blocks_later_launches() {
    MockIOExecutor io;
    CUDAExecutor cuda(0);
    std::atomic<bool> first_launched = false;
    std::atomic<bool> second_launched = false;

    event_status[0] = 0;
    cuda.submit(4, CUDAOperation{
        io.submit(14, false),
        [&]() { first_launched = true; },
        reinterpret_cast<cudaEvent_t>(1),
    });
    cuda.submit(5, CUDAOperation{
        io.submit(15, true, std::make_exception_ptr(std::runtime_error("ordered I/O failure"))),
        [&]() { second_launched = true; },
        reinterpret_cast<cudaEvent_t>(2),
    });

    std::this_thread::sleep_for(std::chrono::milliseconds(10));
    assert(!first_launched.load());
    assert(!second_launched.load());

    io.complete(14);
    cuda.reap(4);
    bool threw = false;
    try {
        cuda.reap(5);
    }
    catch (const std::runtime_error& error) {
        threw = std::string(error.what()) == "ordered I/O failure";
    }
    assert(threw);
    assert(first_launched.load());
    assert(!second_launched.load());

    cuda.join();
    io.join();
}

void test_launch_and_query_errors() {
    {
        MockIOExecutor io;
        CUDAExecutor cuda(0);
        std::atomic<bool> second_launched = false;
        cuda.submit(6, CUDAOperation{
            io.submit(16),
            [] { throw std::runtime_error("launch failure"); },
            reinterpret_cast<cudaEvent_t>(1),
        });
        cuda.submit(7, CUDAOperation{
            io.submit(17),
            [&]() { second_launched = true; },
            reinterpret_cast<cudaEvent_t>(2),
        });

        for (int request_id : {6, 7}) {
            bool threw = false;
            try {
                cuda.reap(request_id);
            }
            catch (const std::runtime_error& error) {
                threw = std::string(error.what()) == "launch failure";
            }
            assert(threw);
        }
        assert(!second_launched.load());
        cuda.join();
        io.join();
    }

    {
        MockIOExecutor io;
        CUDAExecutor cuda(0);
        event_status[3] = 999;
        cuda.submit(8, CUDAOperation{
            io.submit(18),
            [] {},
            reinterpret_cast<cudaEvent_t>(4),
        });

        bool threw = false;
        try {
            cuda.reap(8);
        }
        catch (const std::runtime_error& error) {
            threw = std::string(error.what()).find("test CUDA error") != std::string::npos;
        }
        assert(threw);
        cuda.join();
        io.join();
    }
}

void test_join_drains_pending_event() {
    MockIOExecutor io;
    CUDAExecutor cuda(0);
    std::atomic<bool> launched = false;
    event_status[0] = cudaErrorNotReady;
    cuda.submit(9, CUDAOperation{
        io.submit(19),
        [&]() { launched = true; },
        reinterpret_cast<cudaEvent_t>(1),
    });

    std::thread complete([] {
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
        event_status[0] = 0;
    });
    cuda.join();
    complete.join();
    assert(launched.load());
    io.join();
}

} // namespace

int main() {
    cuda_binding::cudaSetDevice_fn = set_device;
    cuda_binding::cudaEventQuery_fn = query_event;
    cuda_binding::cudaGetErrorString_fn = error_string;

    test_ordered_launch_and_out_of_order_completion();
    test_not_ready();
    test_error_propagation();
    test_io_failure_blocks_later_launches();
    test_launch_and_query_errors();
    test_join_drains_pending_event();
    return 0;
}
