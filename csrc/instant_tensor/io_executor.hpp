#pragma once

#include <any>
#include <chrono>
#include <deque>
#include <exception>
#include <functional>
#include <memory>
#include <vector>

#include <instant_tensor/types.hpp>

namespace instanttensor {

class IOWorkerDriver : public WorkerDriver<std::function<IOSubmitStatus()>, std::any> {
public:
    using Base = WorkerDriver<std::function<IOSubmitStatus()>, std::any>;
    using TaskItem = typename Base::TaskItem;
    using ResultItem = typename Base::ResultItem;
    using PollCompletions = std::function<void(std::vector<IOCompletion>&)>;
    using Clock = std::chrono::steady_clock;

    struct Statistics {
        bool started = false;
        Clock::time_point start{}, end{};
        double active_task_seconds = 0;

        double average_active_tasks() const {
            double seconds = std::chrono::duration<double>(end - start).count();
            return seconds > 0 ? active_task_seconds / seconds : 0;
        }
    };

    IOWorkerDriver(PollCompletions poll_completions,
                   std::function<void()> abort_io, Statistics* stats = nullptr)
        : poll_completions(std::move(poll_completions)), abort_io(std::move(abort_io)),
          stats(stats) {}

    // Worker side, after drain. Excludes tasks still waiting in the executor input queue.
    double average_active_tasks() const {
        return stats ? stats->average_active_tasks() : 0;
    }

    bool can_add_task() const override {
        return !stopping && pending_submissions.empty() &&
               active_tasks.size() < MAX_IO_DEPTH;
    }

    void request_stop() override {
        stopping = true;
    }

    void add_task(TaskItem&& task) override {
        int id = task.request_id;
        record_active_tasks();
        active_tasks.emplace(id, std::move(task));
        pending_submissions.push_back(id);
    }

    std::vector<ResultItem> process_tasks() override {
        // Previous results must leave the driver before another potentially slow submit.
        if (completed.empty() && !pending_submissions.empty()) {
            int id = pending_submissions.front();
            pending_submissions.pop_front();
            start_task(id);
        }
        if (!active_tasks.empty()) {
            std::vector<IOCompletion> events;
            try {
                poll_completions(events);
            }
            catch (...) {
                fail_tasks(std::current_exception());
                events.clear();
            }
            for (auto& event : events) {
                if (event.retry) {
                    pending_submissions.push_back(event.request_id);
                } else {
                    finish_task(event.request_id, event.error);
                }
            }
        }
        if (completed.empty() && has_pending_tasks()) {
            std::this_thread::yield();
        }
        std::vector<ResultItem> result;
        result.swap(completed);
        return result;
    }

    bool has_pending_tasks() const override {
        return !active_tasks.empty() || !completed.empty();
    }

private:
    std::unordered_map<int, TaskItem> active_tasks;
    std::deque<int> pending_submissions;
    std::vector<ResultItem> completed;
    PollCompletions poll_completions;
    std::function<void()> abort_io;
    std::exception_ptr failure;
    bool stopping = false;
    Statistics* stats;

    // Integrate the old count immediately before changing active_tasks.
    void record_active_tasks() {
        if (!stats) return;
        auto now = Clock::now();
        if (!stats->started) {
            stats->start = now;
            stats->started = true;
        } else {
            stats->active_task_seconds +=
                std::chrono::duration<double>(now - stats->end).count() * active_tasks.size();
        }
        stats->end = now;
    }

    void finish_task(int id, std::exception_ptr error = {}) {
        auto it = active_tasks.find(id);
        if (it == active_tasks.end()) {
            throw std::logic_error("Unknown IO completion request ID");
        }
        if (it->second.needs_result) {
            completed.push_back({id, error ? std::any(error) : std::any{}});
        }
        record_active_tasks();
        active_tasks.erase(it);
    }

    void start_task(int id) {
        if (failure) {
            finish_task(id, failure);
            return;
        }
        try {
            auto& task = active_tasks.at(id);
            if (!task.payload || !*task.payload) {
                throw std::logic_error("IO task has no start callback");
            }
            switch ((*task.payload)()) {
                case IOSubmitStatus::Submitted:
                    break;
                case IOSubmitStatus::Retry:
                    // Retry an unaccepted submission before admitting new tasks.
                    pending_submissions.push_front(id);
                    break;
                case IOSubmitStatus::Completed:
                    finish_task(id);
                    break;
            }
        } catch (...) {
            fail_tasks(std::current_exception());
        }
    }

    void fail_tasks(std::exception_ptr error) {
        failure = error;
        // Quiesce both kernel IO and read workers before releasing their windows.
        abort_io();
        pending_submissions.clear();
        while (!active_tasks.empty()) {
            finish_task(active_tasks.begin()->first, error);
        }
    }
};

class IOExecutor : public IOExecutorBase {
public:
    // Optional statistics storage must outlive this executor.
    IOExecutor(IOWorkerDriver::PollCompletions poll_completions,
               std::function<void()> abort_io, IOWorkerDriver::Statistics* stats = nullptr)
      : IOExecutorBase([poll_completions, abort_io, stats]() {
            return std::make_unique<IOWorkerDriver>(
                poll_completions, abort_io, stats);
        })
    {
        IOExecutorBase::start();
    }

    bool try_reap(int request_id, std::any& result) {
        if (!IOExecutorBase::try_reap(request_id, result)) {
            return false;
        }
        rethrow_if_error(result);
        return true;
    }

    void reap(int request_id) {
        std::any result;
        IOExecutorBase::reap(request_id, result);
        rethrow_if_error(result);
    }

private:
    static void rethrow_if_error(const std::any& result) {
        if (result.type() == typeid(std::exception_ptr)) {
            std::rethrow_exception(std::any_cast<std::exception_ptr>(result));
        }
    }
};

} // namespace instanttensor
