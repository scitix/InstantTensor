#pragma once

#include <any>
#include <exception>
#include <functional>
#include <memory>
#include <vector>

#include <instant_tensor/async_executor.hpp>

namespace instanttensor {

class IOWorkerDriver : public WorkerDriver<IOOperation, std::any> {
public:
    using Base = WorkerDriver<IOOperation, std::any>;
    using TaskItem = typename Base::TaskItem;
    using ResultItem = typename Base::ResultItem;

    bool can_add_task() const override {
        return !stopping && active_tasks.size() < MAX_IO_DEPTH;
    }

    void request_stop() override {
        stopping = true;
    }

    void add_task(TaskItem&& task) override {
        ActiveTask active;
        active.task = std::move(task);
        try {
            if (!active.task.payload || !active.task.payload->start) {
                throw std::logic_error("IO task has no start callback");
            }
            IOOperation &operation = *active.task.payload;
            operation.start();
        }
        catch (...) {
            active.error = std::current_exception();
        }
        active_tasks.push_back(std::move(active));
    }

    std::vector<ResultItem> process_tasks() override {
        std::vector<ResultItem> completed;
        for (size_t i = 0; i < active_tasks.size();) {
            ActiveTask &active = active_tasks[i];
            bool done = active.error != nullptr;
            if (!done) {
                try {
                    done = active.task.payload->poll();
                }
                catch (...) {
                    active.error = std::current_exception();
                    done = true;
                }
            }
            if (!done) {
                ++i;
                continue;
            }

            std::any result = active.error ? std::any(active.error) : std::any{};
            if (active.task.needs_result) {
                completed.push_back(ResultItem{
                    active.task.request_id, std::move(result)});
            }
            active_tasks[i] = std::move(active_tasks.back());
            active_tasks.pop_back();
        }
        if (completed.empty() && !active_tasks.empty()) {
            std::this_thread::yield();
        }
        return completed;
    }

    bool has_pending_tasks() const override {
        return !active_tasks.empty();
    }

private:
    struct ActiveTask {
        TaskItem task;
        std::exception_ptr error;
    };
    std::vector<ActiveTask> active_tasks;
    bool stopping = false;
};

class IOExecutor : public IOExecutorBase {
public:
    using IOExecutorBase::try_reap;

    IOExecutor()
      : IOExecutorBase([]() { return std::make_unique<IOWorkerDriver>(); })
    {
        IOExecutorBase::start();
    }

    void reap(int request_id) {
        std::any result;
        IOExecutorBase::reap(request_id, result);
        if (result.type() == typeid(std::exception_ptr)) {
            std::rethrow_exception(std::any_cast<std::exception_ptr>(result));
        }
    }
};

} // namespace instanttensor
