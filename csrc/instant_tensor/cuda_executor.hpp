#pragma once

#include <any>
#include <array>
#include <chrono>
#include <deque>
#include <exception>
#include <functional>
#include <memory>
#include <vector>

#include <instant_tensor/types.hpp>
#include <instant_tensor/io_executor.hpp>

namespace instanttensor {

struct CUDAOperation {
    IORequest io_request;
    std::function<void()> launch;
    cudaEvent_t completion_event;
};

class CUDAWorkerDriver : public WorkerDriver<CUDAOperation, std::any> {
public:
    using Base = WorkerDriver<CUDAOperation, std::any>;
    using TaskItem = typename Base::TaskItem;
    using ResultItem = typename Base::ResultItem;
    using Clock = std::chrono::steady_clock;

    struct Statistics {
        bool started = false;
        Clock::time_point start{}, end{};
        std::array<Clock::duration, 2> stage_times{};

        std::array<double, 2> average_stage_counts() const {
            double seconds = std::chrono::duration<double>(end - start).count();
            if (seconds <= 0) return {};
            return {
                std::chrono::duration<double>(stage_times[0]).count() / seconds,
                std::chrono::duration<double>(stage_times[1]).count() / seconds,
            };
        }
    };

    explicit CUDAWorkerDriver(int device_idx, Statistics* stats = nullptr)
        : stats(stats) {
        try {
            CUDA_CHECK(cudaSetDevice(device_idx));
        }
        catch (...) {
            launch_failure = std::current_exception();
        }
    }

    // Worker side, after drain. Order: IO_READY, GPU_PENDING.
    std::array<double, 2> average_stage_counts() const {
        return stats ? stats->average_stage_counts() : std::array<double, 2>{};
    }

    bool can_add_task() const override {
        return active_tasks.size() < MAX_IO_DEPTH;
    }

    void add_task(TaskItem&& task) override {
        active_tasks.push_back(ActiveTask{std::move(task)});
        if (stats && !stats->started) {
            stats->start = Clock::now();
            stats->started = true;
        }
    }

    std::vector<ResultItem> process_tasks() override {
        std::vector<ResultItem> completed;
        bool launch_blocked = false;

        for (auto it = active_tasks.begin(); it != active_tasks.end();) {
            ActiveTask& active = *it;

            if (active.stage == Stage::IO_PENDING) {
                try {
                    std::any ignored;
                    CUDAOperation& operation = *active.task.payload;
                    if (operation.io_request.executor->try_reap(
                            operation.io_request.wait_handle, ignored)) {
                        set_stage(active, Stage::IO_READY);
                    }
                }
                catch (...) {
                    active.error = std::current_exception();
                    set_stage(active, Stage::IO_READY);
                }
            }

            if (active.stage == Stage::IO_PENDING && !launch_failure) {
                launch_blocked = true;
            }
            else if (active.stage == Stage::IO_READY && !launch_blocked) {
                if (active.error && !launch_failure) {
                    launch_failure = active.error;
                }
                if (launch_failure) {
                    active.error = launch_failure;
                    set_stage(active, Stage::FAILED);
                }
                else {
                    try {
                        active.task.payload->launch();
                        set_stage(active, Stage::GPU_PENDING);
                    }
                    catch (...) {
                        active.error = std::current_exception();
                        set_stage(active, Stage::FAILED);
                        launch_failure = active.error;
                    }
                }
            }

            if (active.stage == Stage::GPU_PENDING) {
                try {
                    cudaError_t error = cudaEventQuery(
                        active.task.payload->completion_event);
                    if (error == cudaSuccess) {
                        set_stage(active, Stage::COMPLETE);
                    }
                    else if (!cudaErrorIsNotReady(error)) {
                        CUDA_CHECK(error);
                    }
                }
                catch (...) {
                    active.error = std::current_exception();
                    set_stage(active, Stage::FAILED);
                }
            }

            if (active.stage != Stage::COMPLETE &&
                active.stage != Stage::FAILED) {
                ++it;
                continue;
            }
            if (active.task.needs_result) {
                std::any result = active.error
                    ? std::any(active.error)
                    : std::any{};
                completed.push_back(ResultItem{
                    active.task.request_id, std::move(result)});
            }
            it = active_tasks.erase(it);
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
    enum class Stage {
        IO_PENDING,
        IO_READY,
        GPU_PENDING,
        COMPLETE,
        FAILED,
    };

    struct ActiveTask {
        TaskItem task;
        Stage stage = Stage::IO_PENDING;
        std::exception_ptr error;
        Clock::time_point stage_started{};
    };

    // Sum per-task residence times, equivalent to integrating each stage's task count.
    void set_stage(ActiveTask& active, Stage stage) {
        if (stats) {
            auto now = Clock::now();
            if (active.stage == Stage::IO_READY) {
                stats->stage_times[0] += now - active.stage_started;
            } else if (active.stage == Stage::GPU_PENDING) {
                stats->stage_times[1] += now - active.stage_started;
            }
            active.stage_started = now;
            stats->end = now;
        }
        active.stage = stage;
    }

    Statistics* stats;
    std::exception_ptr launch_failure;
    std::deque<ActiveTask> active_tasks;
};

using CUDAExecutorBase = SingleWorkerDriverExecutor<CUDAOperation, std::any,
                                                    MAX_IO_DEPTH, MAX_IO_DEPTH>;

class CUDAExecutor : public CUDAExecutorBase {
public:
    // Optional statistics storage must outlive this executor.
    explicit CUDAExecutor(int device_idx, CUDAWorkerDriver::Statistics* stats = nullptr)
      : CUDAExecutorBase([device_idx, stats]() {
            return std::make_unique<CUDAWorkerDriver>(device_idx, stats);
        })
    {
        CUDAExecutorBase::start();
    }

    bool try_reap(int request_id, std::any& result) {
        if (!CUDAExecutorBase::try_reap(request_id, result)) {
            return false;
        }
        rethrow_if_error(result);
        return true;
    }

    void reap(int request_id) {
        std::any result;
        CUDAExecutorBase::reap(request_id, result);
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
