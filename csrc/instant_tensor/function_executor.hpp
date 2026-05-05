#pragma once

#include <any>
#include <functional>
#include <memory>
#include <optional>
#include <type_traits>
#include <utility>
#include <vector>

#include <instant_tensor/async_executor.hpp>

namespace instanttensor {

class FunctionWorkerDriver : public WorkerDriver<std::function<std::any()>, std::any> {
public:
    using TaskFn = std::function<std::any()>;
    using Base = WorkerDriver<TaskFn, std::any>;
    using TaskItem = typename Base::TaskItem;
    using ResultItem = typename Base::ResultItem;

    // Worker side only. FunctionWorkerDriver has no extra initialization.
    FunctionWorkerDriver() = default;

    // Worker side only.
    bool can_add_task() const override {
        return !pending_task.has_value();
    }

    // Worker side only.
    void add_task(TaskItem&& task) override {
        pending_task.emplace(std::move(task));
    }

    // Worker side only.
    std::vector<ResultItem> process_tasks() override {
        if (!pending_task) {
            return {};
        }

        TaskItem task = std::move(*pending_task);
        pending_task.reset();

        std::any result = (*task.payload)();
        if (!task.needs_result) {
            return {};
        }
        return {ResultItem{task.request_id, std::move(result)}};
    }

    // Worker side only.
    bool has_pending_tasks() const override {
        return pending_task.has_value();
    }

private:
    std::optional<TaskItem> pending_task;
};

// SingleWorkerFunctionExecutor is the function-task specialization. Payload is
// a callable normalized to TaskFn, and Result is std::any.
template<size_t TaskQueueCapacity = 1024, size_t ResultQueueCapacity = 1024>
class SingleWorkerFunctionExecutor
  : public SingleWorkerDriverExecutor<std::function<std::any()>, std::any,
                                      TaskQueueCapacity, ResultQueueCapacity> {
public:
    using TaskFn = std::function<std::any()>;
    using Base = SingleWorkerDriverExecutor<TaskFn, std::any,
                                            TaskQueueCapacity, ResultQueueCapacity>;

    // Lifecycle side: stores the function driver factory. Call start() before submit().
    SingleWorkerFunctionExecutor()
      : Base([]() {
            return std::make_unique<FunctionWorkerDriver>();
        })
    {}

    // Submitter side only.
    template<typename F>
    int submit(int request_id, F&& fn, bool needs_result = true) {
        return Base::submit(request_id, make_task_fn(std::forward<F>(fn)), needs_result);
    }

    using Base::reap;

    // Reaper side only.
    template<typename R>
    void reap(int request_id, R& result) {
        std::any result_any;
        Base::reap(request_id, result_any);
        result = std::any_cast<R>(std::move(result_any));
    }

private:
    // Submitter side only. Normalizes arbitrary callables to TaskFn.
    template<typename F>
    static TaskFn make_task_fn(F&& fn) {
        return [fn = std::forward<F>(fn)]() mutable -> std::any {
            using R = std::invoke_result_t<decltype(fn)>;
            if constexpr (std::is_void_v<R>) {
                fn();
                return std::any{};
            }
            else {
                return std::any{ fn() };
            }
        };
    }
};

// MultiWorkerFunctionExecutor is the function-task specialization. Payload is
// a callable normalized to TaskFn, and Result is std::any.
template<size_t TaskQueueCapacity = 1024, size_t ResultQueueCapacity = 1024>
class MultiWorkerFunctionExecutor
  : public MultiWorkerDriverExecutor<std::function<std::any()>, std::any,
                                     TaskQueueCapacity, ResultQueueCapacity> {
public:
    using TaskFn = std::function<std::any()>;
    using Base = MultiWorkerDriverExecutor<TaskFn, std::any,
                                           TaskQueueCapacity, ResultQueueCapacity>;

    // Lifecycle side: stores the function driver factory. Call start() before submit().
    MultiWorkerFunctionExecutor()
      : Base([]() {
            return std::make_unique<FunctionWorkerDriver>();
        })
    {}

    // Submitter side only.
    template<typename F>
    int submit(int request_id, F&& fn, bool needs_result = true) {
        return Base::submit(request_id, make_task_fn(std::forward<F>(fn)), needs_result);
    }

    using Base::reap;

    // Reaper side only.
    template<typename R>
    void reap(int request_id, R& result) {
        std::any result_any;
        Base::reap(request_id, result_any);
        result = std::any_cast<R>(std::move(result_any));
    }

private:
    // Submitter side only. Normalizes arbitrary callables to TaskFn.
    template<typename F>
    static TaskFn make_task_fn(F&& fn) {
        return [fn = std::forward<F>(fn)]() mutable -> std::any {
            using R = std::invoke_result_t<decltype(fn)>;
            if constexpr (std::is_void_v<R>) {
                fn();
                return std::any{};
            }
            else {
                return std::any{ fn() };
            }
        };
    }
};

} // namespace instanttensor
