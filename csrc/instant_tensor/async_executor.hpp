#pragma once

#include <any>
#include <atomic>
#include <functional>
#include <thread>
#include <unordered_map>
#include <utility>

#include <instant_tensor/queue.hpp>

namespace instanttensor {

template<size_t InputQueueCapacity = 1024, size_t OutputQueueCapacity = 1024>
class SPSCAsyncExecutor {
public:
    using Task      = std::function<std::any()>;

    SPSCAsyncExecutor()
      : next_id(1),
        worker(&SPSCAsyncExecutor::run, this)
    {}

    ~SPSCAsyncExecutor() {
        join();
    }

    template<typename F>
    int post(F&& f) {
        int id = next_id.fetch_add(1, std::memory_order_relaxed);

        Task t = [fn = std::forward<F>(f)]() mutable -> std::any {
            using R = std::invoke_result_t<decltype(fn)>;
            if constexpr (std::is_void_v<R>) {
                fn();               
                return std::any{};  
            }
            else {
                return std::any{ fn() };  
            }
        };

        WorkItem w{id, std::move(t)};

        while (!input_queue.try_push(w)) {
            drainOutputQueueToBuffer();
            std::this_thread::yield();
        }
        return id;
    }

    bool test(int req_id) {
        bool finished = result_buffer.find(req_id) != result_buffer.end();
        if(finished) return true;
        std::any out;
        finished = drainOutputQueueUntil(req_id, out);
        if(finished) {
            result_buffer.emplace(req_id, std::move(out));
        }
        return finished;
    }

    void wait(int req_id) {
        bool finished = result_buffer.find(req_id) != result_buffer.end();
        if(finished) return;
        while(true) {
            std::any out;
            finished = drainOutputQueueUntil(req_id, out);
            if(finished) {
                result_buffer.emplace(req_id, std::move(out));
                return;
            }
            std::this_thread::yield();
        }
    }

    bool try_pop(int req_id, std::any& out) {
        auto it = result_buffer.find(req_id);
        if (it != result_buffer.end()) {
            out = std::move(it->second);
            result_buffer.erase(it);
            return true;
        }
        return drainOutputQueueUntil(req_id, out);
    }

    template<typename R>
    void pop(int req_id, R& out) {
        std::any res;
        pop(req_id, res);
        out = std::any_cast<R>(std::move(res));
    }

    void pop(int req_id, std::any& out) {
        auto it = result_buffer.find(req_id);
        if (it != result_buffer.end()) {
            out = std::move(it->second);
            result_buffer.erase(it);
            return;
        }

        while(true) {
            bool finished = drainOutputQueueUntil(req_id, out);
            if(finished) return;
            std::this_thread::yield();
        }
    }

    void pop(int req_id) {
        std::any out;
        pop(req_id, out);
    }

    void stopAsync() {
        WorkItem w{0, Task()};  // id==0 is the sentinel
        while (!input_queue.try_push(w)) {
            drainOutputQueueToBuffer();
            std::this_thread::yield();
        }
    }

    void join() {
        if (worker.joinable()) {
            stopAsync();
            worker.join();
        }
    }

private:
    struct WorkItem {
        int   id;
        Task  task;
    };

    using WorkPair  = std::pair<int,std::any>;

    SPSCQueue<WorkItem, InputQueueCapacity>      input_queue;
    SPSCQueue<WorkPair, OutputQueueCapacity>      output_queue;
    std::unordered_map<int,std::any> result_buffer; // cache out-of-order results

    std::atomic<int>         next_id;
    std::thread              worker;

    void run() {
        WorkItem w;
        while (true) {
            while (!input_queue.try_pop(w)) {
                std::this_thread::yield();
            }
            if (!w.task) {
                break;
            }

            std::any res = w.task();

            WorkPair pr{w.id, std::move(res)};
            while (!output_queue.try_push(pr)) {
                std::this_thread::yield();
            }
        }
    }

    // Drain everything from output_queue into result_buffer
    void drainOutputQueueToBuffer() {
        WorkPair pr;
        while (output_queue.try_pop(pr)) {
            result_buffer.emplace(pr.first, std::move(pr.second));
        }
    }

    // Drain everything from output_queue into result_buffer
    // Early return if the expected result of req_id is found
    bool drainOutputQueueUntil(int req_id, std::any& out) {
        WorkPair pr;
        while (output_queue.try_pop(pr)) {
            if (pr.first == req_id) {
                out = std::move(pr.second);
                return true;
            }
            result_buffer.emplace(pr.first, std::move(pr.second)); 
        }
        return false;
    }
};

} // namespace instanttensor
