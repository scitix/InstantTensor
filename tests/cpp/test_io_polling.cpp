#include <cassert>
#include <atomic>
#include <cmath>
#include <deque>
#include <mutex>
#include <instant_tensor/loader.hpp>

using namespace instanttensor;

namespace {
struct Read {
    chunk_id_t id;
    size_t offset;
    size_t size;
    void* buffer;
    unsigned flags;
};
std::deque<std::pair<chunk_id_t, int>> events;
std::deque<int> read_results;
std::deque<int> submit_results;
std::vector<Read> reads;
std::function<void(const Read&)> check_submission;
std::mutex mock_mutex;
std::atomic<bool> block_submit{false}, submit_entered{false};
std::atomic<int> destroys{0};
int poll_error = 0;
int polls = 0;
int sqes = 0;
io_uring_sqe sqe{};
io_uring_cqe cqe{};
io_uring_cqe batch_cqes[MAX_IO_DEPTH];
int batch_peeks = 0, batch_advances = 0;
bool empty_batch_once = false;

int submit(Read read) {
    if (check_submission) check_submission(read);
    int result = 1;
    {
        std::lock_guard<std::mutex> lock(mock_mutex);
        reads.push_back(read);
        if (!submit_results.empty()) {
            result = submit_results.front();
            submit_results.pop_front();
        }
        if (result == 1) {
            assert(!read_results.empty());
            events.emplace_back(read.id, read_results.front());
            read_results.pop_front();
        }
    }
    submit_entered = true;
    while (block_submit) std::this_thread::yield();
    return result;
}

int mock_io_submit(io_context_t, long count, iocb** blocks) {
    assert(count == 1);
    auto& cb = *blocks[0];
    return submit({static_cast<chunk_id_t>(reinterpret_cast<uintptr_t>(cb.data)),
                   static_cast<size_t>(cb.u.c.offset), static_cast<size_t>(cb.u.c.nbytes),
                   cb.u.c.buf, 0});
}
int mock_io_getevents(io_context_t, long min, long max, io_event* output, timespec* timeout) {
    assert(min == 0 && timeout->tv_sec == 0 && timeout->tv_nsec == 0);
    ++polls;
    if (poll_error) return std::exchange(poll_error, 0);
    std::lock_guard<std::mutex> lock(mock_mutex);
    int count = 0;
    while (!events.empty() && count < max) {
        output[count] = {};
        output[count].data = reinterpret_cast<void*>(static_cast<uintptr_t>(events.front().first));
        output[count].res = events.front().second;
        events.pop_front();
        ++count;
    }
    return count;
}
int mock_io_setup(unsigned, io_context_t*) { return 0; }
int mock_io_destroy(io_context_t) { ++destroys; events.clear(); return 0; }
io_uring_sqe* mock_get_sqe(io_uring*) { ++sqes; sqe = {}; return &sqe; }
int mock_uring_submit(io_uring*) {
    return submit({static_cast<chunk_id_t>(sqe.user_data), sqe.off, sqe.len,
                   reinterpret_cast<void*>(sqe.addr), sqe.flags});
}
int mock_peek(io_uring*, io_uring_cqe** output) {
    ++polls;
    if (poll_error) return std::exchange(poll_error, 0);
    if (events.empty()) return -EAGAIN;
    cqe = {};
    cqe.user_data = events.front().first;
    cqe.res = events.front().second;
    *output = &cqe;
    return 0;
}
void mock_seen(io_uring*, io_uring_cqe*) { assert(!events.empty()); events.pop_front(); }
unsigned mock_peek_batch(io_uring*, io_uring_cqe** output, unsigned count) {
    ++polls;
    ++batch_peeks;
    // The batch API returns a count, not a negative errno.
    if (poll_error || std::exchange(empty_batch_once, false)) return 0;
    count = std::min(count, static_cast<unsigned>(events.size()));
    for (unsigned i = 0; i < count; ++i) {
        batch_cqes[i] = {};
        batch_cqes[i].user_data = events[i].first;
        batch_cqes[i].res = events[i].second;
        output[i] = &batch_cqes[i];
    }
    return count;
}
void mock_cq_advance(io_uring*, unsigned count) {
    ++batch_advances;
    assert(count <= events.size());
    for (unsigned i = 0; i < count; ++i) {
        events.pop_front();
        batch_cqes[i].user_data = static_cast<uint64_t>(-1);
    }
}
void mock_exit(io_uring*) { assert(events.empty()); ++destroys; }
int mock_unregister(io_uring*) { assert(events.empty()); return 0; }
}

// Interpose only system interfaces; scheduling, read accounting, and drain are production code.
#define io_submit mock_io_submit
#define io_getevents mock_io_getevents
#define io_setup mock_io_setup
#define io_destroy mock_io_destroy
#define io_uring_get_sqe mock_get_sqe
#define io_uring_submit mock_uring_submit
#define io_uring_peek_cqe mock_peek
#define io_uring_peek_batch_cqe mock_peek_batch
#define io_uring_cq_advance mock_cq_advance
#define io_uring_wait_cqe mock_peek
#define io_uring_cqe_seen mock_seen
#define io_uring_queue_exit mock_exit
#define io_uring_unregister_buffers mock_unregister
#define io_uring_unregister_files mock_unregister
#include "../../csrc/loader_common.cpp"
#include "../../csrc/loader_io_aio.cpp"
#include "../../csrc/loader_io_uring.cpp"
#include "../../csrc/loader_io_inmem.cpp"

namespace {
void reset() {
    events.clear(); read_results.clear(); submit_results.clear(); reads.clear();
    check_submission = {};
    poll_error = polls = sqes = 0;
    batch_peeks = batch_advances = 0;
    empty_batch_once = false;
    destroys = 0; block_submit = false; submit_entered = false;
}

void check_publishing() {
    std::atomic<bool> entered{false}, release{false};
    std::vector<IOCompletion> ready;
    IOExecutor executor([&](auto& output) { output.swap(ready); }, [] {});
    // Hold the first start until all 512 tasks and the stop sentinel are queued.
    // The next start itself checks the result queue, so this is not a timing benchmark.
    for (int id = 0; id < 512; ++id) {
        auto start = [&, id]() {
            if (id == 0) {
                entered = true;
                while (!release) std::this_thread::yield();
            } else {
                assert(executor.ready(id - 1));
            }
            if (id % 2) return IOSubmitStatus::Completed;
            ready.push_back({id, {}});
            return IOSubmitStatus::Submitted;
        };
        executor.submit(id, std::move(start));
    }
    while (!entered) std::this_thread::yield();
    executor.stop();
    release = true;
    executor.join();
    for (int id = 0; id < 512; ++id) executor.reap(id);
}

void check_driver() {
    std::vector<IOCompletion> events;
    int polls = 0, starts = 0, aborts = 0;
    bool fail_poll = false;
    IOWorkerDriver driver([&](auto& output) {
        ++polls;
        if (fail_poll) throw std::runtime_error("poll failed");
        output.swap(events);
    }, [&]() { ++aborts; });
    for (int id = 0; id < 512; ++id) {
        driver.add_task(IOWorkerDriver::TaskItem::make_task(id,
            [&]() { ++starts; return IOSubmitStatus::Submitted; }));
        assert(starts == id); // Admission only queues the first submission.
        assert(!driver.can_add_task());
        assert(driver.process_tasks().empty());
        assert(starts == id + 1); // One submit followed by one poll per round.
    }
    for (int i = 0; i < 100; ++i) assert(driver.process_tasks().empty());
    assert(polls == 612); // Independent of the 512 active tasks.
    events = {{511, {}}, {2, {}, true}, {3, std::make_exception_ptr(std::runtime_error("read"))}};
    auto result = driver.process_tasks();
    assert(starts == 512 && result.size() == 2);
    assert(result[0].request_id == 511 && result[1].request_id == 3);
    assert(result[1].value.type() == typeid(std::exception_ptr));
    assert(!driver.can_add_task());
    driver.process_tasks();
    assert(starts == 513);
    driver.request_stop();
    fail_poll = true;
    result = driver.process_tasks();
    assert(aborts == 1 && result.size() == 510 && !driver.has_pending_tasks());
    for (const auto& item : result) assert(item.value.type() == typeid(std::exception_ptr));
}

void check_completed_status() {
    std::vector<IOCompletion> ready;
    int polls = 0;
    IOWorkerDriver driver([&](auto& output) {
        ++polls;
        output.swap(ready);
    }, [] { assert(false); });
    auto completed_task = [] { return IOSubmitStatus::Completed; };
    driver.add_task(IOWorkerDriver::TaskItem::make_task(0, completed_task));
    auto result = driver.process_tasks();
    assert(result.size() == 1 && result[0].request_id == 0 && !result[0].value.has_value());
    assert(polls == 0 && !driver.has_pending_tasks());
    assert(driver.process_tasks().empty()); // No duplicate completion.

    driver.add_task(IOWorkerDriver::TaskItem::make_task(1, completed_task, false));
    assert(driver.process_tasks().empty());
    assert(polls == 0 && !driver.has_pending_tasks());

    driver.add_task(IOWorkerDriver::TaskItem::make_task(2,
        [] { return IOSubmitStatus::Submitted; }));
    assert(driver.process_tasks().empty() && polls == 1);
    ready.push_back({2, {}});
    driver.add_task(IOWorkerDriver::TaskItem::make_task(3, completed_task));
    result = driver.process_tasks();
    assert(result.size() == 2 && result[0].request_id == 3 && result[1].request_id == 2);
    assert(polls == 2 && !driver.has_pending_tasks());
}

void check_io_statistics(bool enabled) {
    IOWorkerDriver::Statistics stats;
    {
        IOWorkerDriver driver([](auto&) {}, [] { assert(false); }, enabled ? &stats : nullptr);
        assert(driver.average_active_tasks() == 0);
        int attempts = 0;
        driver.add_task(IOWorkerDriver::TaskItem::make_task(0, [&] {
            return ++attempts < 3 ? IOSubmitStatus::Retry : IOSubmitStatus::Completed;
        }));
        assert(driver.process_tasks().empty());
        assert(driver.process_tasks().empty());
        assert(driver.process_tasks().size() == 1);
        assert(!driver.has_pending_tasks());
        double average = driver.average_active_tasks();
        assert(std::abs(average - (enabled ? 1.0 : 0.0)) < 1e-9);
        for (int i = 0; i < 100; ++i) assert(driver.process_tasks().empty());
        assert(driver.average_active_tasks() == average);
    }
    // Statistics survive the worker driver and are read only after it has drained.
    assert(std::abs(stats.average_active_tasks() - (enabled ? 1.0 : 0.0)) < 1e-9);
}

void check_io_statistics_on_abort() {
    IOWorkerDriver::Statistics stats;
    int aborts = 0;
    IOWorkerDriver driver([](auto&) {}, [&] { ++aborts; }, &stats);
    driver.add_task(IOWorkerDriver::TaskItem::make_task(0,
        [] { return IOSubmitStatus::Submitted; }));
    assert(driver.process_tasks().empty());
    driver.add_task(IOWorkerDriver::TaskItem::make_task(1, []() -> IOSubmitStatus {
        throw std::runtime_error("submit failed");
    }));
    assert(driver.process_tasks().size() == 2 && aborts == 1);
    assert(!driver.has_pending_tasks());
    double average = driver.average_active_tasks();
    assert(std::isfinite(average) && average >= 1 && average <= 2);
}

struct Fixture {
    Loader loader{nullptr, nullptr};
    std::vector<char> buffer = std::vector<char>(65536);
    FILE* file = tmpfile();
    bool aio;
    explicit Fixture(bool aio) : aio(aio) {
        reset();
        assert(file && ftruncate(fileno(file), 65536) == 0);
        loader.backend = aio ? Backend::AIO : Backend::URING;
        loader.io_depth = 4; loader.rank_chunk_size = 16384;
        loader.rank_alignment = loader.world_chunk_alignment = 4096;
        loader.world_size = 1; loader.rank = 0;
        loader.host_buffer = buffer.data();
        loader.file_info.resize(1);
        loader.file_info[0].fd = fileno(file);
        loader.chunks.resize(4);
        loader.uring_register_file = loader.uring_register_buffer = false;
        if (aio) {
            loader.initialize_aio_context();
            loader.last_page_reader_thread = std::make_unique<SingleThreadTaskExecutor>();
        } else {
            loader.uring_context_initialized = true;
        }
    }
    ~Fixture() {
        loader.destroy_threads();
        fclose(file);
    }
    void start_executor() {
        loader.io_thread = std::make_unique<IOExecutor>(
            [&](auto& output) { poll(output); }, [&]() { loader.abort_io(); });
    }
    void poll(std::vector<IOCompletion>& output) {
        if (aio) loader.poll_aio(output); else loader.poll_uring(output);
    }
    IORequest post(int id, size_t size) {
        Chunk& chunk = loader.chunks[id];
        chunk.size = size;
        chunk.file_offset = id * 16384;
        size_t padded_rank = ROUND_UP(size, loader.world_chunk_alignment) / loader.world_size;
        size_t rank_offset = padded_rank * loader.rank;
        size_t window_idx = id % loader.io_depth;
        chunk.io_state.rank_size = rank_logical_size(size, rank_offset, padded_rank);
        chunk.io_state.rank_file_offset = chunk.file_offset + rank_offset;
        chunk.io_state.window_offset = window_idx * loader.rank_chunk_size;
        ChunkIOParams params{ id, chunk, loader.file_info[0],
            window_idx,
            nullptr, nullptr, nullptr };
        return aio ? loader.post_read_chunk_aio(params) : loader.post_read_chunk_uring(params);
    }
};

void check_read(bool aio, std::deque<int> results, size_t logical,
                const std::string& expected_error = "", std::deque<int> submits = {}) {
    Fixture f(aio);
    size_t expected_reads = results.size();
    read_results = std::move(results);
    submit_results = std::move(submits);
    f.start_executor();
    auto request = f.post(0, logical);
    f.loader.io_thread->stop(); // Retries and last-page work must still drain after stop.
    std::string error;
    try { request.executor->reap(request.wait_handle); }
    catch (const std::runtime_error& e) { error = e.what(); }
    f.loader.io_thread->join();
    assert(expected_error.empty() ? error.empty() : error.find(expected_error) != std::string::npos);
    assert(read_results.empty() && submit_results.empty() && events.empty());
    if (!aio) assert(sqes == static_cast<int>(expected_reads));
    assert(f.loader.chunks[0].size == logical && f.loader.chunks[0].file_offset == 0);
    if (expected_error.empty()) assert(f.loader.chunks[0].io_state.bytes_completed == logical);
    for (const auto& read : reads) {
        assert(read.offset % 4096 == 0 && read.size % 4096 == 0);
        assert(read.buffer == f.buffer.data() + read.offset);
        if (!aio) assert(bool(read.flags & IOSQE_ASYNC) == (logical % 4096 != 0));
    }
}

void check_fixed_read_range(bool aio, int id) {
    Fixture f(aio);
    f.loader.chunks.resize(id + 1);
    assert(ftruncate(fileno(f.file), 131072) == 0);
    f.loader.world_size = 2;
    f.loader.world_chunk_alignment = 8192;
    f.loader.rank = 1;
    read_results = {1000, 3000, 100, 4096, 1};
    std::vector<size_t> progress{0, 1000, 3000, 3000, 4096};
    size_t chunk_file_offset = id * 16384;
    size_t rank_file_offset = chunk_file_offset + 8192;
    size_t attempt = 0;
    check_submission = [&](const Read& read) {
        const auto& chunk = f.loader.chunks[id];
        const auto& state = chunk.io_state;
        assert(chunk.size == 12289 && chunk.file_offset == chunk_file_offset);
        assert(state.rank_size == 4097 && state.rank_file_offset == rank_file_offset);
        assert(state.window_offset == 16384);
        assert(attempt < progress.size() && state.bytes_completed == progress[attempt]);
        size_t read_offset = attempt == 4 ? 4096 : 0;
        assert(read.id == id && read.offset == rank_file_offset + read_offset);
        assert(read.buffer == f.buffer.data() + 16384 + read_offset);
        assert(read.size == (attempt == 4 ? 4096 : 8192));
        ++attempt;
    };
    f.start_executor();
    auto request = f.post(id, 12289);
    f.loader.io_thread->stop();
    request.executor->reap(request.wait_handle);
    f.loader.io_thread->join();
    const auto& state = f.loader.chunks[id].io_state;
    assert(attempt == progress.size() && state.bytes_completed == 4097);
    assert(f.loader.chunks[id].size == 12289 && f.loader.chunks[id].file_offset == chunk_file_offset);
    assert(read_results.empty() && events.empty());
    check_submission = {};
}

void check_tail_completion_before_helper(bool retry) {
    Fixture f(true);
    auto& state = f.loader.chunks[0].io_state;
    f.loader.chunks[0].size = 100;
    state.io_request_id = 17;
    state.rank_size = 100;
    state.rank_file_offset = 0;
    state.bytes_completed = 0;
    read_results = retry ? std::deque<int>{40, 100} : std::deque<int>{100};
    block_submit = true;
    assert(f.loader.schedule_read_aio(0) == IOSubmitStatus::Submitted);
    while (!submit_entered) std::this_thread::yield();
    std::vector<IOCompletion> completed;
    f.poll(completed);
    assert(completed.size() == 1 && completed[0].request_id == 17);
    assert(!completed[0].error && completed[0].retry == retry);
    assert(state.bytes_completed == (retry ? 40 : 100));
    assert(f.loader.aio_last_page_submissions.size() == 1);
    assert(f.loader.chunks[0].size == 100 && f.loader.chunks[0].file_offset == 0);
    if (retry) {
        // Queue the retry before the previous submit helper returns.
        assert(f.loader.schedule_read_aio(0) == IOSubmitStatus::Submitted);
        assert(f.loader.aio_last_page_submissions.size() == 2);
    }
    completed.clear();
    f.poll(completed);
    assert(completed.empty());
    block_submit = false;
    if (retry) {
        while (completed.empty()) f.poll(completed);
        assert(completed.size() == 1 && completed[0].request_id == 17);
        assert(!completed[0].error && !completed[0].retry);
    }
    // Shutdown must reap acknowledgements even after all logical IO completed.
    f.loader.destroy_threads();
    assert(f.loader.aio_last_page_submissions.empty());
    SingleThreadTaskExecutor::ResultItem ignored;
    assert(!f.loader.last_page_reader_thread->try_reap_any(ignored));
    completed.clear();
    f.poll(completed);
    assert(completed.empty());
    assert(state.bytes_completed == 100);
}

void check_tail_published_before_helper() {
    Fixture f(true);
    read_results = {100};
    block_submit = true;
    f.start_executor();
    auto request = f.post(0, 100);
    while (!submit_entered) std::this_thread::yield();
    request.executor->reap(request.wait_handle);
    // The IO driver can finish while the successful submit helper is still returning.
    f.loader.io_thread->join();
    assert(block_submit && submit_entered);
    assert(f.loader.aio_last_page_submissions.size() == 1);
    block_submit = false;
    f.loader.destroy_threads();
    assert(f.loader.aio_last_page_submissions.empty());
    SingleThreadTaskExecutor::ResultItem ignored;
    assert(!f.loader.last_page_reader_thread->try_reap_any(ignored));
}

void check_error_routing(bool aio) {
    Fixture f(aio);
    for (int id = 0; id < 4; ++id) {
        auto& chunk = f.loader.chunks[id];
        chunk.size = 8192;
        chunk.io_state.io_request_id = 100 + id;
        chunk.io_state.rank_size = 8192;
        chunk.io_state.rank_file_offset = chunk.file_offset;
        chunk.io_state.bytes_completed = 0;
    }
    events = {{2, -EIO}, {3, 8192}, {0, -EINTR}, {1, 4096}};
    f.loader.uring_reads_pending = 4;
    std::vector<IOCompletion> output;
    f.poll(output);
    assert(output.size() == 4);
    assert(output[0].request_id == 102 && output[0].error);
    assert(output[1].request_id == 103 && !output[1].error && !output[1].retry);
    assert(output[2].request_id == 100 && output[2].retry);
    assert(output[3].request_id == 101 && output[3].retry);
    assert(reads.empty()); // No resubmissions hidden in the completion batch.
}

void check_queue_failure(bool aio, bool submit_failure, bool tail) {
    Fixture f(aio);
    size_t size = tail ? 17 : 8192;
    if (submit_failure) {
        submit_results = aio ? std::deque<int>{-EIO}
                             : std::deque<int>{0, -EINTR, -EAGAIN, -EIO};
    } else {
        read_results = {static_cast<int>(size)};
        poll_error = -EIO;
    }
    f.start_executor();
    auto first = f.post(0, size);
    auto second = f.post(1, 0);
    f.loader.io_thread->stop();
    int failures = 0;
    for (auto request : {first, second}) {
        try { request.executor->reap(request.wait_handle); }
        catch (const std::runtime_error&) { ++failures; }
    }
    f.loader.io_thread->join();
    // A failed last-page helper submit is local; a queue failure poisons later tasks.
    bool local = aio && tail && submit_failure;
    assert(failures == (local ? 1 : 2));
    assert(destroys == (local ? 0 : 1));
    assert(events.empty() && read_results.empty() && submit_results.empty());
    assert(f.loader.uring_reads_pending == 0);
    if (!local) {
        f.loader.abort_io();
        assert(destroys == 1); // Close after failure cannot tear down twice.
    }
}

void check_empty_rank(bool aio) {
    Fixture f(aio);
    f.loader.world_size = 2;
    f.loader.world_chunk_alignment = 8192;
    f.loader.rank = 1;
    f.start_executor();
    auto request = f.post(0, 17);
    request.executor->reap(request.wait_handle);
    f.loader.io_thread->join();
    const auto& state = f.loader.chunks[0].io_state;
    assert(state.rank_size == 0 && state.rank_file_offset == 4096);
    assert(reads.empty() && sqes == 0);
}

void check_retry_priority() {
    std::vector<IOCompletion> ready;
    int attempt = 0;
    std::vector<int> order;
    IOWorkerDriver driver([&](auto& output) { output.swap(ready); }, [] {});
    driver.add_task(IOWorkerDriver::TaskItem::make_task(0,
        [&]() { order.push_back(0); return IOSubmitStatus::Submitted; }));
    assert(order.empty());
    driver.process_tasks();
    driver.add_task(IOWorkerDriver::TaskItem::make_task(1,
        [&]() {
            order.push_back(1);
            if (attempt++ < 2) return IOSubmitStatus::Retry;
            ready.push_back({1, {}});
            return IOSubmitStatus::Submitted;
        }));
    ready.push_back({0, {}, true});
    assert((order == std::vector<int>{0}));
    driver.process_tasks();
    assert((order == std::vector<int>{0, 1}));
    assert(!driver.can_add_task());
    driver.process_tasks();
    assert((order == std::vector<int>{0, 1, 1}));
    assert(!driver.can_add_task());
    auto result = driver.process_tasks();
    assert(result.size() == 1 && result[0].request_id == 1);
    assert((order == std::vector<int>{0, 1, 1, 1}));
    assert(!driver.can_add_task());
    driver.process_tasks();
    assert(order.back() == 0);
    ready.push_back({0, {}});
    assert(driver.process_tasks().size() == 1);
}

void check_uring_submit_spin() {
    Fixture f(false);
    f.loader.chunks[0].size = 8192;
    auto& state = f.loader.chunks[0].io_state;
    state.io_request_id = 17;
    state.rank_size = 8192;
    state.rank_file_offset = state.bytes_completed = 0;
    read_results = {8192};
    submit_results = {0, -EINTR, -EAGAIN, 1};
    assert(f.loader.submit_read_uring(0) == IOSubmitStatus::Submitted);
    assert(submit_results.empty() && reads.size() == 4);
    assert(sqes == 1 && polls == 0 && f.loader.uring_reads_pending == 1);
    std::vector<IOCompletion> completed;
    f.poll(completed);
    assert(completed.size() == 1 && completed[0].request_id == 17);
    assert(!completed[0].error && !completed[0].retry);
    assert(state.bytes_completed == 8192 && f.loader.uring_reads_pending == 0);
}

void check_uring_batch_poll() {
    Fixture f(false);
    f.loader.chunks.resize(5);
    for (int id = 0; id < 5; ++id) {
        auto& state = f.loader.chunks[id].io_state;
        state.io_request_id = 100 + id;
        state.rank_size = 8192;
        state.bytes_completed = 0;
    }
    events = {{4, 8192}, {1, -EIO}, {3, 4096}, {0, -EAGAIN}, {2, 8192}};
    f.loader.uring_reads_pending = events.size();
    std::vector<IOCompletion> output;
    f.poll(output);
    assert(polls == 1 && batch_peeks == 1 && batch_advances == 1);
    assert(output.size() == 4 && events.size() == 1 && f.loader.uring_reads_pending == 1);
    assert(output[0].request_id == 104 && !output[0].error && !output[0].retry);
    assert(output[1].request_id == 101 && output[1].error);
    assert(output[2].request_id == 103 && output[2].retry);
    assert(output[3].request_id == 100 && output[3].retry);
    assert(reads.empty()); // Short reads only enqueue retries, never submit within the batch.

    // A completion can become visible between batch peek and the fallback peek.
    empty_batch_once = true;
    output.clear();
    f.poll(output);
    assert(polls == 3 && batch_peeks == 2 && batch_advances == 2);
    assert(output.size() == 1 && output[0].request_id == 102);
    assert(!output[0].error && !output[0].retry);
    assert(events.empty() && f.loader.uring_reads_pending == 0);
    output.clear();
    f.poll(output);
    assert(output.empty() && batch_peeks == 3 && batch_advances == 2);
}

cudaError_t mock_copy(void* dst, const void* src, size_t size,
                      cudaMemcpyKind kind, cudaStream_t) {
    assert(kind == cudaMemcpyHostToDevice);
    std::memcpy(dst, src, size);
    return cudaSuccess;
}

void check_mmap(bool registered, bool empty) {
    Loader loader(nullptr, nullptr);
    loader.backend = Backend::MMAP;
    loader.use_internal_memory_register = registered;
    if (!registered) loader.worker_threads = std::make_unique<ThreadPoolTaskExecutor>(2);
    loader.io_thread = std::make_unique<IOExecutor>(
        [&](auto& output) { loader.poll_worker_completions(output); },
        [&]() { loader.abort_io(); });
    loader.chunks.resize(1);
    char source[128], staging[128] = {}, device[128] = {};
    std::memset(source, 42, sizeof(source));
    loader.host_buffer = staging;
    loader.chunks[0].file_offset = 16;
    loader.chunks[0].io_state.rank_file_offset = 24;
    loader.chunks[0].io_state.rank_size = empty ? 0 : 17;
    loader.chunks[0].io_state.window_offset = 64;
    FileInfo file{};
    file.mapped_memory = source;
    ChunkIOParams params{0, loader.chunks[0], file,
                         0, device + 32, device, nullptr};
    auto request = loader.post_read_chunk_inmem(params);
    assert(request.loaded_to_device == registered);
    loader.io_thread->join();
    if (loader.worker_threads) loader.worker_threads->join();
    request.executor->reap(request.wait_handle);
    auto* result = registered ? device + 32 : staging + 64;
    for (int i = 0; i < 17; ++i) assert(result[i] == (empty ? 0 : 42));
}

void check_close_reply_lifetime() {
    auto output = std::make_shared<SPSCQueue<RPCResponse>>();
    {
        Loader loader(nullptr, output);
        loader.output_queue->push(RPCResponse{17, {}});
    }
    RPCResponse response;
    assert(output->try_pop(response) && response.id == 17);
}

void check_chunk_completion_handles() {
    auto saved_set_device = cuda_binding::cudaSetDevice_fn;
    auto saved_query_event = cuda_binding::cudaEventQuery_fn;
    cuda_binding::cudaSetDevice_fn = [](int) { return cudaSuccess; };
    cuda_binding::cudaEventQuery_fn = [](cudaEvent_t event) {
        return reinterpret_cast<std::atomic<cudaError_t>*>(event)->load();
    };
    Loader loader(nullptr, nullptr);
    loader.chunks.resize(2);
    loader.chunk_reading = 1;
    std::atomic<cudaError_t> status[2] = {cudaErrorNotReady, cudaSuccess};
    std::vector<IOCompletion> ready;
    IOExecutor io([&](auto& output) { output.swap(ready); }, [] {});
    CUDAExecutor cuda(0);
    for (int i = 0; i < 2; ++i) {
        auto& state = loader.chunks[i].io_state;
        state.io_request_id = 100 + i;
        state.bytes_completed = 4096 + i;
        io.submit(state.io_request_id, [&, i]() {
            ready.push_back({100 + i, {}});
            return IOSubmitStatus::Submitted;
        });
        int cuda_request_id = 200 + i;
        cuda.submit(cuda_request_id, CUDAOperation{
            IORequest{&io, state.io_request_id, false}, [] {},
            reinterpret_cast<cudaEvent_t>(&status[i])});
        state.cuda_executor = &cuda;
        state.cuda_request_id = cuda_request_id;
    }
    while (!cuda.ready(201)) std::this_thread::yield();
    loader.poll_read_chunk();
    assert(loader.chunk_read == -1);
    status[0] = cudaSuccess;
    loader.wait_read_chunk(1);
    assert(loader.chunk_read == 1);
    for (int i = 0; i < 2; ++i) {
        assert(loader.chunks[i].io_state.io_request_id == 100 + i);
        assert(loader.chunks[i].io_state.bytes_completed == 4096 + i);
    }
    cuda.join();
    io.join();
    cuda_binding::cudaSetDevice_fn = saved_set_device;
    cuda_binding::cudaEventQuery_fn = saved_query_event;
}

void check_worker_abort_drains(bool fail_poll) {
    Loader loader(nullptr, nullptr);
    loader.backend = Backend::MMAP;
    loader.worker_threads = std::make_unique<ThreadPoolTaskExecutor>(1);
    std::atomic<bool> entered{false}, release{false}, finished{false};
    std::atomic<bool> abort_entered{false}, inject_error{false};
    loader.io_thread = std::make_unique<IOExecutor>(
        [&](auto& output) {
            if (fail_poll && inject_error) throw std::runtime_error("worker poll failure");
            loader.poll_worker_completions(output);
        }, [&]() {
            abort_entered = true;
            loader.abort_io();
            assert(finished);
        });
    loader.io_thread->submit(17, [&]() {
        loader.worker_threads->submit(17, [&]() {
            entered = true;
            while (!release) std::this_thread::yield();
            finished = true;
        });
        return IOSubmitStatus::Submitted;
    });
    while (!entered) std::this_thread::yield();
    loader.io_thread->submit(29, [&]() {
        inject_error = true;
        if (!fail_poll) throw std::runtime_error("worker submit failure");
        return IOSubmitStatus::Submitted;
    });
    while (!abort_entered) std::this_thread::yield();
    std::any ignored;
    assert(!loader.io_thread->try_reap(17, ignored));
    assert(!loader.io_thread->try_reap(29, ignored));
    loader.io_thread->stop();
    release = true;
    loader.io_thread->join();
    for (int id : {17, 29}) {
        bool failed = false;
        try { loader.io_thread->reap(id); }
        catch (const std::runtime_error&) { failed = true; }
        assert(failed);
    }
    assert(finished);
}

void check_worker_poll_batch() {
    Loader loader(nullptr, nullptr);
    loader.worker_threads = std::make_unique<ThreadPoolTaskExecutor>(2);
    std::atomic<bool> release{false}, entered{false};
    loader.worker_threads->submit(17, [&]() {
        entered = true;
        while (!release) std::this_thread::yield();
    });
    while (!entered) std::this_thread::yield();
    loader.worker_threads->submit(29, []() { throw std::runtime_error("read failure"); });
    std::vector<IOCompletion> completed;
    while (completed.empty()) loader.poll_worker_completions(completed);
    assert(completed.size() == 1);
    assert(completed.back().request_id == 29 && completed.back().error);
    completed.clear();
    for (int i = 0; i < 100; ++i) {
        loader.poll_worker_completions(completed);
        assert(completed.empty());
    }
    release = true;
    loader.worker_threads->join();
    loader.poll_worker_completions(completed);
    assert(completed.size() == 1 && completed[0].request_id == 17 && !completed[0].error);
}
}

int main() {
    check_publishing();
    check_driver();
    check_completed_status();
    check_io_statistics(false);
    check_io_statistics(true);
    check_io_statistics_on_abort();
    check_close_reply_lifetime();
    check_chunk_completion_handles();
    check_worker_poll_batch();
    check_worker_abort_drains(false);
    check_worker_abort_drains(true);
    check_retry_priority();
    check_uring_submit_spin();
    check_uring_batch_poll();
    for (bool aio : {true, false}) {
        check_read(aio, {8192}, 8192);
        check_read(aio, {4096, 4096}, 8192);
        check_read(aio, {5000, 4096}, 8192);
        check_read(aio, {5000, 100, 0, -EAGAIN, 4096}, 8192);
        check_read(aio, {0, 8192}, 8192);
        check_read(aio, {-EINTR, -EAGAIN, 8192}, 8192);
        check_read(aio, {8192}, 8192, "", {0, -EINTR, -EAGAIN, 1});
        check_read(aio, {17}, 17, "", {0, -EINTR, -EAGAIN, 1});
        check_read(aio, {-EIO}, 8192, "read error");
        check_read(aio, {-EIO}, 17, "read error");
        check_read(aio, {4096, 17}, 4113);
        check_read(aio, {4096, 0, 17}, 4113);
        check_read(aio, {}, 0);
        check_error_routing(aio);
        check_empty_rank(aio);
        check_fixed_read_range(aio, 1);
        check_fixed_read_range(aio, 5);
        for (bool tail : {false, true}) {
            check_queue_failure(aio, false, tail);
            check_queue_failure(aio, true, tail);
        }
    }
    check_tail_completion_before_helper(false);
    check_tail_completion_before_helper(true);
    check_tail_published_before_helper();
    cuda_binding::cudaMemcpyAsync_fn = mock_copy;
    for (bool registered : {false, true}) {
        check_mmap(registered, false);
        check_mmap(registered, true);
    }
}
