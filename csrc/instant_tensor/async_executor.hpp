#pragma once

#include <cstdio>
#include <exception>
#include <functional>
#include <memory>
#include <optional>
#include <stdexcept>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>

#include <instant_tensor/queue.hpp>

namespace instanttensor {

static constexpr int EXECUTOR_STOP_REQUEST_ID = -1;

template<typename Payload>
struct ExecutorTaskItem {
    static constexpr int STOP_REQUEST_ID = EXECUTOR_STOP_REQUEST_ID;

    int request_id = 0;
    bool needs_result = true;
    std::optional<Payload> payload;

    // Submitter side only. Builds a regular task item.
    static ExecutorTaskItem make_task(int request_id, Payload payload, bool needs_result = true) {
        return ExecutorTaskItem{request_id, needs_result, std::move(payload)};
    }

    // Lifecycle side only. Builds the stop sentinel consumed by Workers.
    static ExecutorTaskItem make_stop() {
        ExecutorTaskItem task;
        task.request_id = STOP_REQUEST_ID;
        task.needs_result = false;
        return task;
    }

    // Worker side only. Returns whether this item is the stop sentinel.
    bool is_stop() const {
        return request_id == STOP_REQUEST_ID;
    }
};

template<typename Result>
struct ExecutorResultItem {
    int request_id = 0;
    Result value;
};

// WorkerDriver is the per-worker policy object. ExecutorCore creates one
// instance per Worker thread, so each driver can keep thread-local working
// state directly in its own members.
template<typename Payload, typename Result>
class WorkerDriver {
public:
    using TaskItem = ExecutorTaskItem<Payload>;
    using ResultItem = ExecutorResultItem<Result>;

    virtual ~WorkerDriver() = default;

    // Worker side only. Returns whether this worker can accept one more task.
    virtual bool can_add_task() const = 0;

    // Worker side only. Adds a task to this worker's current working set.
    virtual void add_task(TaskItem&& task) = 0;

    // Worker side only. Advances this worker's current working set and returns
    // completed results from this call. It may return an empty list.
    virtual std::vector<ResultItem> process_tasks() = 0;

    // Worker side only. Returns whether this worker has an active working set.
    virtual bool has_pending_tasks() const = 0;
};

// ExecutorCore owns the task/result queues and worker thread lifecycle.
// It is parameterized by queue topology: SPSC/SPSC for one worker, or
// SPMC/MPSC for multiple workers. Each Worker thread receives its own driver
// instance from driver_factory.
template<
    typename Payload,
    typename Result,
    template<typename, size_t> class TaskQueue,
    template<typename, size_t> class ResultQueue,
    size_t TaskQueueCapacity,
    size_t ResultQueueCapacity>
class ExecutorCore {
public:
    using Driver = WorkerDriver<Payload, Result>;
    using TaskItem = typename Driver::TaskItem;
    using ResultItem = typename Driver::ResultItem;

    static constexpr int STOP_REQUEST_ID = TaskItem::STOP_REQUEST_ID;

    using DriverFactory = std::function<std::unique_ptr<Driver>()>;

    // Lifecycle side: stores the per-worker Driver factory.
    explicit ExecutorCore(DriverFactory driver_factory)
      : driver_factory(std::move(driver_factory))
    {}

    // Lifecycle side: joins workers before destroying queues/results.
    ~ExecutorCore() {
        join();
    }

    // Executors own threads and lock-free queues, so they are not copyable.
    ExecutorCore(const ExecutorCore&) = delete;
    // Executors own threads and lock-free queues, so they are not copyable.
    ExecutorCore& operator=(const ExecutorCore&) = delete;

    // Submitter side only. This never drains completed results; callers must
    // keep a Reaper running, or bound outstanding tasks below queue capacity.
    // Otherwise task_queue and result_queue can fill each other and deadlock.
    int submit(int request_id, Payload payload, bool needs_result = true) {
        if (request_id == STOP_REQUEST_ID) {
            throw std::invalid_argument("request_id is reserved for stop");
        }
        if (!started) {
            throw std::logic_error("executor is not started");
        }
        if (stop_requested) {
            throw std::logic_error("executor is stopped");
        }

        TaskItem task = TaskItem::make_task(request_id, std::move(payload), needs_result);
        task_queue.push(std::move(task));
        return request_id;
    }

    // Reaper side only.
    bool ready(int request_id) {
        if (completed_results.find(request_id) != completed_results.end()) {
            return true;
        }
        return cache_until_ready(request_id);
    }

    // Reaper side only.
    void wait(int request_id) {
        while(!ready(request_id)) {
            std::this_thread::yield();
        }
    }

    // Reaper side only.
    bool try_reap(int request_id, Result& result) {
        if (take_cached_result(request_id, result)) {
            return true;
        }
        return take_until_ready(request_id, result);
    }

    // Reaper side only.
    void reap(int request_id, Result& result) {
        if (take_cached_result(request_id, result)) {
            return;
        }

        while(!take_until_ready(request_id, result)) {
            std::this_thread::yield();
        }
    }

    // Reaper side only.
    void reap(int request_id) {
        auto it = completed_results.find(request_id);
        if (it != completed_results.end()) {
            completed_results.erase(it);
            return;
        }

        while(!cache_until_ready(request_id)) {
            std::this_thread::yield();
        }
        completed_results.erase(request_id);
    }

    // Lifecycle/Submitter side only. The caller must ensure outstanding work can
    // complete or be reaped; this never drains result_queue.
    void stop() {
        if (!started || stop_requested) {
            return;
        }
        stop_requested = true;

        for (size_t i = 0; i < worker_count; ++i) {
            TaskItem task = TaskItem::make_stop();
            task_queue.push(std::move(task));
        }
    }

    // Lifecycle side only. Do not call concurrently with Submitter/Reaper APIs.
    void join() {
        stop();
        for (auto& worker_thread : worker_threads) {
            if (worker_thread.joinable()) {
                worker_thread.join();
            }
        }
        worker_threads.clear();
    }

    // Lifecycle side: starts worker threads. Must be called before submit().
    void start(size_t num_workers) {
        if (started) {
            throw std::logic_error("executor is already started");
        }
        if (stop_requested) {
            throw std::logic_error("executor is stopped");
        }
        if (num_workers == 0) {
            num_workers = 1;
        }
        started = true;
        worker_count = num_workers;
        worker_threads.reserve(num_workers);
        for (size_t i = 0; i < num_workers; ++i) {
            worker_threads.emplace_back(&ExecutorCore::worker_loop, this);
        }
    }

private:
    TaskQueue<TaskItem, TaskQueueCapacity>       task_queue;
    ResultQueue<ResultItem, ResultQueueCapacity>  result_queue;
    std::unordered_map<int, Result> completed_results; // cache out-of-order results

    DriverFactory driver_factory;
    bool started = false;
    bool stop_requested = false;
    size_t worker_count = 0;
    std::vector<std::thread> worker_threads;

    // Worker side only.
    void worker_loop() {
        try {
            std::unique_ptr<Driver> driver = driver_factory();

            TaskItem task;
            while (true) {
                bool received_task = false;
                while (driver->can_add_task() && task_queue.try_pop(task)) {
                    received_task = true;
                    if (task.is_stop()) {
                        return;
                    }
                    driver->add_task(std::move(task));
                }

                if (!received_task && !driver->has_pending_tasks()) {
                    std::this_thread::yield();
                    continue;
                }

                std::vector<ResultItem> result_items = driver->process_tasks();
                for (auto& result_item : result_items) {
                    result_queue.push(std::move(result_item));
                }
            }
        }
        catch (const std::exception &e) {
            fprintf(stderr, "Executor worker thread exception: %s\n", e.what());
            throw;
        }
    }

    // Reaper side only. Moves a cached result out of completed_results.
    bool take_cached_result(int request_id, Result& result) {
        auto it = completed_results.find(request_id);
        if (it == completed_results.end()) {
            return false;
        }
        result = std::move(it->second);
        completed_results.erase(it);
        return true;
    }

    // Reaper side only. Drains result_queue into completed_results and returns
    // true once request_id has been cached.
    bool cache_until_ready(int request_id) {
        ResultItem result_item;
        while (result_queue.try_pop(result_item)) {
            int result_request_id = result_item.request_id;
            completed_results.emplace(result_request_id, std::move(result_item.value));
            if (result_request_id == request_id) {
                return true;
            }
        }
        return false;
    }

    // Reaper side only. Drains result_queue into completed_results and returns
    // true after moving request_id's result into result.
    bool take_until_ready(int request_id, Result& result) {
        ResultItem result_item;
        while (result_queue.try_pop(result_item)) {
            if (result_item.request_id == request_id) {
                result = std::move(result_item.value);
                return true;
            }
            completed_results.emplace(result_item.request_id, std::move(result_item.value));
        }
        return false;
    }
};

// SingleWorkerDriverExecutor has three roles:
// - Submitter: exactly one thread calls submit().
// - Worker: one internal thread processes payloads through a WorkerDriver.
// - Reaper: exactly one thread calls ready(), wait(), try_reap(), or reap().
// Submitter and Reaper may be the same thread only if the caller bounds the
// number of outstanding tasks so task/result queues cannot fill each other.
template<typename Payload, typename Result,
         size_t TaskQueueCapacity = 1024, size_t ResultQueueCapacity = 1024>
class SingleWorkerDriverExecutor
  : public ExecutorCore<Payload, Result, SPSCQueue, SPSCQueue,
                        TaskQueueCapacity, ResultQueueCapacity> {
public:
    using Base = ExecutorCore<Payload, Result, SPSCQueue, SPSCQueue,
                              TaskQueueCapacity, ResultQueueCapacity>;
    using DriverFactory = typename Base::DriverFactory;

    // Lifecycle side: stores the per-worker Driver factory.
    explicit SingleWorkerDriverExecutor(DriverFactory driver_factory)
      : Base(std::move(driver_factory))
    {}

    // Lifecycle side: starts the single Worker thread.
    void start() {
        Base::start(1);
    }
};

// MultiWorkerDriverExecutor has three roles:
// - Submitter: exactly one thread calls submit().
// - Workers: internal worker threads process payloads through WorkerDrivers.
// - Reaper: exactly one thread calls ready(), wait(), try_reap(), or reap().
// Submitter and Reaper may be the same thread only if the caller bounds the
// number of outstanding tasks so task/result queues cannot fill each other.
template<typename Payload, typename Result,
         size_t TaskQueueCapacity = 1024, size_t ResultQueueCapacity = 1024>
class MultiWorkerDriverExecutor
  : public ExecutorCore<Payload, Result, SPMCQueue, MPSCQueue,
                        TaskQueueCapacity, ResultQueueCapacity> {
public:
    using Base = ExecutorCore<Payload, Result, SPMCQueue, MPSCQueue,
                              TaskQueueCapacity, ResultQueueCapacity>;
    using DriverFactory = typename Base::DriverFactory;

    // Lifecycle side: stores the per-worker Driver factory.
    explicit MultiWorkerDriverExecutor(DriverFactory driver_factory)
      : Base(std::move(driver_factory))
    {}

    // Lifecycle side: starts Worker threads.
    void start(size_t num_workers = std::thread::hardware_concurrency()) {
        Base::start(num_workers);
    }
};

} // namespace instanttensor
