#include <instant_tensor/loader.hpp>

namespace instanttensor {

Loader::Loader(unique_ptr<SPSCQueue<RPCRequest>> input_queue, unique_ptr<SPSCQueue<RPCResponse>> output_queue) {
    this->input_queue = std::move(input_queue);
    this->output_queue = std::move(output_queue);
    this->use_internal_memory_register = _env_use_internal_memory_register();
    this->use_cufile = _env_use_cufile();
}

void Loader::open_file() {
    for(size_t i = 0; i < this->file_info.size(); i++) {
        FileInfo& f = this->file_info[i];
        f.in_memory = instanttensor::file_in_memory(f.filename);

        if (f.in_memory) {
            this->open_file_inmem(f);
        } else if (this->use_cufile) {
            this->open_file_cufile(f);
        } else {
            this->open_file_aio(f);
        }
    }
    if(this->need_aio) {
        this->open_file_aio_context();
    }
}

void Loader::close_file() {
    for(size_t i = 0; i < this->file_info.size(); i++) {
        FileInfo& f = this->file_info[i];
        if (f.in_memory) {
            this->close_file_inmem(f);
        } else if (this->use_cufile) {
            this->close_file_cufile(f);
        } else {
            this->close_file_aio(f);
        }
    }
    if(this->need_aio) {
        this->close_file_aio_context();
    }
}

void Loader::init_buffer() {
    this->thread_alignment = PAGE_SIZE;
    this->rank_alignment = this->thread_alignment * this->num_threads;
    this->world_chunk_alignment = this->rank_alignment * this->world_size;
    if(this->thread_chunk_size % this->thread_alignment != 0) {
        size_t new_chunk_size = ROUND_UP(this->thread_chunk_size, this->thread_alignment);
        if(_env_debug()) {
            fprintf(stderr, "Enlarge thread_chunk_size from %zu to %zu to align to %zu\n", this->thread_chunk_size, new_chunk_size, this->thread_alignment);
        }
        this->thread_chunk_size = new_chunk_size;
    }
    this->rank_chunk_size = this->thread_chunk_size * this->num_threads;
    this->world_chunk_size = this->rank_chunk_size * this->world_size;

    size_t inflight_device_buffer_size = this->io_depth * this->world_chunk_size;
    if (this->buffer_size < inflight_device_buffer_size) this->buffer_size = inflight_device_buffer_size;

    // At most first_tensor_alignment bytes are padded both before and after the chunk
    // At most thread_alignment bytes are padded before a chunk if the previous chunk's size < world_chunk_size
    // At most world_chunk_alignment bytes are padded after a chunk if its size < world_chunk_size
    // For three tensors, these paddings exist at most 3 times
    size_t padded_size = 3 * (this->first_tensor_alignment + this->thread_alignment + this->world_chunk_alignment); // can be any value that >= 0
    this->buffer_size += padded_size;
    CUDA_CHECK(cudaMalloc(&this->device_buffer, this->buffer_size));

    if (this->need_cufile) {
        CUFILE_CHECK(cuFileBufRegister(this->device_buffer, this->buffer_size, 0));
    }
    if (this->need_host_buffer) {
        size_t inflight_host_buffer_size = this->io_depth * this->rank_chunk_size;
        size_t host_buffer_size = inflight_host_buffer_size;
        this->host_buffer_entry = host_buffer_cache->get(host_buffer_size);
        if (this->host_buffer_entry.ptr == NULL) {
            // aligned_alloc + cudaHostRegister is faster than cudaHostAlloc
            this->host_buffer_entry.ptr = aligned_alloc(this->thread_alignment, host_buffer_size);
            if (this->host_buffer_entry.ptr == NULL) {
                throw std::runtime_error("Failed to allocate host buffer: " + std::string(strerror(errno)));
            }

            // NOTE: cudaHostRegisterReadOnly is not used since the host buffer is writable for the CPU
            CUDA_CHECK(cudaHostRegister(this->host_buffer_entry.ptr, host_buffer_size, this->cuda_host_register_flags));

            this->host_buffer_entry.size = host_buffer_size;
            this->host_buffer_entry.deleter = [=](void *ptr) {
                CUDA_CHECK(cudaHostUnregister(ptr));
                free(ptr);
            };
        }
        this->host_buffer = (char*)this->host_buffer_entry.ptr;
    }
}

void Loader::destroy_buffer() {
    if (this->need_cufile) {
        CUFILE_CHECK(cuFileBufDeregister(this->device_buffer));
    }
    CUDA_CHECK(cudaFree(this->device_buffer));
    if (this->need_host_buffer) {
        if (_env_cache_buffer()) {
            host_buffer_cache->put(std::move(this->host_buffer_entry));
        }
        else {
            this->host_buffer_entry.deleter(this->host_buffer_entry.ptr);
            this->host_buffer_entry.ptr = nullptr;
        }
    }
}

void Loader::init_threads() {
    auto set_device_func = [=]() { CUDA_CHECK(cudaSetDevice(this->device_idx)); };

    if(this->need_worker_threads) {
        while(this->worker_threads.size() < this->num_threads) {
            this->worker_threads.emplace_back(std::make_unique<AsyncExecutor>());
        }
    }
    if(this->need_cuda_thread) {
        if(!this->cuda_thread) {
            this->cuda_thread = std::make_unique<AsyncExecutor>();
        }
    }

    if(!this->cuda_stream) {
        CUDA_CHECK(cudaStreamCreateWithFlags(&this->cuda_stream, cudaStreamNonBlocking));
    }
    if(!this->nccl_stream) {
        CUDA_CHECK(cudaStreamCreateWithFlags(&this->nccl_stream, cudaStreamNonBlocking));
    }
    if(!this->wait_thread) {
        this->wait_thread = std::make_unique<AsyncExecutor>();
    }

    for(auto &thread : this->worker_threads) {
        thread->post(set_device_func);
    }
    if(this->cuda_thread) {
        this->cuda_thread->post(set_device_func);
    }
    if(this->wait_thread) {
        this->wait_thread->post(set_device_func);
    }
    this->cuda_events.resize(this->io_depth);
    for(size_t i = 0; i < this->io_depth; i++) {
        CUDA_CHECK(cudaEventCreateWithFlags(&this->cuda_events[i], cudaEventDisableTiming));
    }

    if(_env_debug()) {
        this->io_depth_sample_thread = std::thread([=]() {
            while(!this->stop) {
                std::this_thread::sleep_for(std::chrono::milliseconds(10));
                this->io_depth_sum += this->chunk_reading - this->chunk_read;
                this->io_depth_sample ++;
            }
        });
    }
}

void Loader::destroy_threads() {
    for (auto& thread : this->worker_threads) {
        thread->join();
    }
    if (this->cuda_thread) {
        this->cuda_thread->join();
    }
    if (this->wait_thread) {
        this->wait_thread->join();
    }
    if (this->cuda_stream) {
        CUDA_CHECK(cudaStreamDestroy(this->cuda_stream));
    }
    if (this->nccl_stream) {
        CUDA_CHECK(cudaStreamDestroy(this->nccl_stream));
    }
    if(this->io_depth_sample_thread.joinable()) {
        this->io_depth_sample_thread.join();
    }
}

/*
 * Compute: 1) addresses of the tensors in the buffer, 2) each chunk's range, 3) prefetch chunk for each tensor
 */
void Loader::compute_layout(const vector<pair<size_t, size_t>>& tensor_offsets) {// list[(file_index, tensor_offset)]
    // NOTE: One chunk is a group of contiguous tensors, may have left and right non-tensor paddings.
    //       It may correspond to at most three types of storage: file, host buffer, device buffer.
    //       The size and offset of the file storage and the host buffer of a chunk is aligned to thread_alignment (== PAGE_SIZE)
    //       since libaio with O_DIRECT requres page-aligned I/O. (cuFile does not require this)
    //       The size of the device buffer of a chunk is aligned to world_chunk_alignment for ncclAllGather, and the offset is set to let tensors aligned (see below).
    // NOTE: We set the device buffer offset of the first chunk of a file properly
    //       to let the device buffer offset of the first tensor be aligned to first_tensor_alignment.
    //       Since for any dtype (FP32/FP16/...), first_tensor_alignment % sizeof(dtype) == 0, the first tensor is always dtype-aligned.
    //       Consecutive tensors in current and consecutive chunks are then also dtype-aligned since the tensors
    //       are listed in the order of decreasing dtype size (e.g., FP32->FP16/BF16->FP8/INT8) in safetensors files.
    size_t chunk_file_index = 0;
    size_t chunk_file_offset = 0; // The chunk offset from the beginning of the file, aligned to thread_alignment (typically == PAGE_SIZE)
    size_t chunk_device_buffer_offset = 0;
    size_t current_chunk_size = 0;

    // Points to the current tensor
    size_t tensor_id = 0;
    // Points to the earliest tensor that is still in the device buffer
    // in_buffer_tensor_id increases with tensor_id and satisfies in_buffer_tensor_id <= tensor_id
    size_t in_buffer_tensor_id = 0;
    // Points to the tensor with the smallest device memory address
    // left_most_tensor_id increases with tensor_id and satisfies in_buffer_tensor_id <= left_most_tensor_id <= tensor_id
    size_t left_most_tensor_id = 0;

    size_t num_files = tensor_offsets[tensor_offsets.size() - 1].first + 1;
    size_t num_tensors = tensor_offsets.size() - num_files; // for each file, num_offsets = num_tensors + 1
    this->tensors.resize(num_tensors);

    auto tensor_device_buffer_offset = [&](size_t tensor_file_offset) -> size_t {
        return tensor_file_offset - chunk_file_offset + chunk_device_buffer_offset;
        // assert(tensor_file_offset - chunk_file_offset == current_chunk_size);
    };

    auto finish_chunk = [&]() {
        if(current_chunk_size > 0) {
            // ROUND_UP/DOWN make real effect only at the right most chunk
            this->chunks.push_back(Chunk{current_chunk_size, chunk_file_index, chunk_file_offset, chunk_device_buffer_offset, {}, {}});

            if(chunk_file_offset % this->thread_alignment != 0) {
                throw std::runtime_error("Internal error: Chunk alignment error.");
            }

            // may reread the last file page
            chunk_file_offset += ROUND_DOWN(current_chunk_size, this->thread_alignment);
            chunk_device_buffer_offset += ROUND_UP(current_chunk_size, this->world_chunk_alignment);
            // equals to "current_chunk_size -= ROUND_DOWN(current_chunk_size, this->thread_alignment);""
            current_chunk_size %= this->thread_alignment;


            chunk_id_t prev_chunk_id = (chunk_id_t)this->chunks.size() - 2;
            while(in_buffer_tensor_id < left_most_tensor_id
                && this->tensors[in_buffer_tensor_id].device_buffer_offset < chunk_device_buffer_offset) {
                this->tensors[in_buffer_tensor_id].prefetch_chunk_id = prev_chunk_id;
                in_buffer_tensor_id++;
            }
        }
    };
    auto reset_chunk_buffer_offset = [&](size_t new_left_most_tensor_id) {
        if(current_chunk_size >= this->thread_alignment) {
            print_and_throw(std::runtime_error("Internal error: Unchunked page detected when resetting chunk buffer offset."));
        }
        chunk_id_t latest_chunk_id = (chunk_id_t)this->chunks.size() - 1;
        while(in_buffer_tensor_id < left_most_tensor_id) {
            this->tensors[in_buffer_tensor_id].prefetch_chunk_id = latest_chunk_id;
            in_buffer_tensor_id++;
        }
        chunk_device_buffer_offset = this->first_tensor_alignment - current_chunk_size % this->first_tensor_alignment; // make the first tensor address aligned to first_tensor_alignment
        left_most_tensor_id = new_left_most_tensor_id;
    };
    auto reset_chunk_file = [&](size_t file_index, size_t file_offset) {
        if(current_chunk_size >= this->thread_alignment) {
            print_and_throw(std::runtime_error("Internal error: Unchunked page detected when resetting chunk file offset."));
        }
        chunk_file_index = file_index;
        chunk_file_offset = ROUND_DOWN(file_offset, this->thread_alignment);
        current_chunk_size = file_offset % this->thread_alignment;
        chunk_device_buffer_offset = ROUND_UP(chunk_device_buffer_offset, this->first_tensor_alignment) + this->first_tensor_alignment - current_chunk_size % this->first_tensor_alignment; // make the first tensor address aligned to first_tensor_alignment
    };


    reset_chunk_file(tensor_offsets[0].first, tensor_offsets[0].second);

    for (size_t i = 0; i < tensor_offsets.size() - 1; i++) {
        size_t file_index = tensor_offsets[i].first;
        size_t tensor_file_offset = tensor_offsets[i].second;
        size_t next_file_index = tensor_offsets[i + 1].first;
        size_t next_tensor_offset = tensor_offsets[i + 1].second;
        if (file_index != next_file_index) {
            finish_chunk();
            reset_chunk_file(next_file_index, next_tensor_offset);
            continue;
        }
        if(current_chunk_size != tensor_file_offset - chunk_file_offset) {
            throw std::runtime_error("Internal error: Unchunked file size mismatch.");
        }
        size_t tensor_size = next_tensor_offset - tensor_file_offset;
        if (tensor_size > this->buffer_size) {
            print_and_throw(std::runtime_error("Internal error: Buffer size is smaller than a single tensor size. This should be forbidden by the python frontend."));
        }
        if (chunk_device_buffer_offset + ROUND_UP(current_chunk_size + tensor_size, this->world_chunk_alignment) > this->buffer_size) {
            finish_chunk();
            // place the tensor at the beginning of the buffer
            reset_chunk_buffer_offset(tensor_id);
        }
        size_t _tensor_device_buffer_offset = tensor_device_buffer_offset(tensor_file_offset);
        size_t tensor_size_left = tensor_size;
        while(current_chunk_size + tensor_size_left > this->world_chunk_size) {
            size_t size_add = this->world_chunk_size - current_chunk_size;
            current_chunk_size += size_add;
            tensor_size_left -= size_add;
            finish_chunk();
        }
        current_chunk_size += tensor_size_left;
        chunk_id_t current_chunk_id = (chunk_id_t)this->chunks.size();
        this->tensors[tensor_id] = TensorMetadate{tensor_size, file_index, tensor_file_offset, _tensor_device_buffer_offset, current_chunk_id, 0};

        tensor_id ++;
    }
    finish_chunk();
    // set prefetch_chunk_id of the remaining second-to-last layer tensors to the last chunk
    reset_chunk_buffer_offset(tensor_id);
    // set prefetch_chunk_id of all last layer tensors to the last chunk
    reset_chunk_buffer_offset(tensor_id);
}

void Loader::open(OpenArgs args) {
    this->file_info.resize(args.filenames.size());
    for(size_t i = 0; i < args.filenames.size(); i++) {
        this->file_info[i].filename = args.filenames[i];
    }
    this->device_idx = args.device_idx;
    this->group_communicator = args.group_communicator;
    this->rank = args.rank;
    this->world_size = args.world_size;
    this->buffer_size = args.buffer_size;
    this->thread_chunk_size = args.chunk_size;
    this->num_threads = args.num_threads;
    this->io_depth = args.io_depth;

    if (this->world_size > 1 && this->group_communicator == NULL) {
        print_and_throw(std::runtime_error("Internal error: A communicatior should be provided if world_size > 1"));
    }
    if (this->world_size == 1 && this->group_communicator != NULL) {
        print_and_throw(std::runtime_error("Internal error: A communicatior should not be provided if world_size == 1"));
    }

    auto t0 = std::chrono::high_resolution_clock::now();
    cudaSetDevice(this->device_idx);
    auto t1 = std::chrono::high_resolution_clock::now();

    // this->init_comm();
    auto t2 = std::chrono::high_resolution_clock::now();
    this->open_file();// NOTE: this should be called before init_buffer/init_threads/
    auto t3 = std::chrono::high_resolution_clock::now();
    this->init_buffer();
    auto t4 = std::chrono::high_resolution_clock::now();
    this->init_threads();
    auto t5 = std::chrono::high_resolution_clock::now();
    this->compute_layout(args.tensor_offsets);
    auto t6 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> d1 = t1 - t0;
    std::chrono::duration<double> d2 = t2 - t1;
    std::chrono::duration<double> d3 = t3 - t2;
    std::chrono::duration<double> d4 = t4 - t3;
    std::chrono::duration<double> d5 = t5 - t4;
    std::chrono::duration<double> d6 = t6 - t5;
    if(_env_debug()) {
        fprintf(stderr, "Config: rank=%d/%d, num_threads=%zu, device_buffer_size=%zu, host_buffer_size=%zu, chunk_size=%zu, io_depth=%zu, device=%d, communicator=%p\n",
            this->rank, this->world_size, this->num_threads, this->buffer_size, this->host_buffer_entry.size, this->thread_chunk_size, this->io_depth, this->device_idx, (void*)(this->group_communicator));
        fprintf(stderr, "Open time: device=%f, comm=%f, file=%f, buffer=%f, threads=%f, layout=%f\n", d1.count(), d2.count(), d3.count(), d4.count(), d5.count(), d6.count());
    }
}

void Loader::close(CloseArgs args) {
    this->stop = true;
    auto t0 = std::chrono::high_resolution_clock::now();
    this->destroy_threads();
    auto t1 = std::chrono::high_resolution_clock::now();
    this->destroy_buffer();
    auto t2 = std::chrono::high_resolution_clock::now();
    this->close_file();
    auto t3 = std::chrono::high_resolution_clock::now();
    // this->destroy_comm();
    auto t4 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> d1 = t1 - t0;
    std::chrono::duration<double> d2 = t2 - t1;
    std::chrono::duration<double> d3 = t3 - t2;
    std::chrono::duration<double> d4 = t4 - t3;
    if(_env_debug()) {
        fprintf(stderr, "Close time: threads=%f, buffer=%f, file=%f, comm=%f\n", d1.count(), d2.count(), d3.count(), d4.count());
        fprintf(stderr, "Average io_depth = %.2lf\n", 1.0 * this->io_depth_sum / this->io_depth_sample);
    }
}

void Loader::post_read_chunk() {
    this->chunk_reading.store(this->chunk_reading.load(std::memory_order_relaxed) + 1, std::memory_order_relaxed);
    chunk_id_t chunk_id = this->chunk_reading.load(std::memory_order_relaxed);
    Chunk &chunk = this->chunks[chunk_id];

    // only the right most chunks of files and of buffers may not be aligned and need padding
    size_t padded_world_chunk_size = ROUND_UP(chunk.size, this->world_chunk_alignment);
    size_t padded_rank_size = padded_world_chunk_size / this->world_size;
    size_t padded_thread_size = padded_rank_size / this->num_threads;
    size_t rank_offset = padded_rank_size * this->rank;
    size_t rank_size = std::min((size_t)std::max((ssize_t)(chunk.size - rank_offset), (ssize_t)0), padded_rank_size);
    size_t window_idx = chunk_id % this->io_depth;
    size_t window_offset = window_idx * this->rank_chunk_size;
    void *rank_dst = (char*)this->device_buffer + chunk.device_buffer_offset + rank_offset;
    void *all_dst = (char*)this->device_buffer + chunk.device_buffer_offset;

    cudaEvent_t event = this->cuda_events[window_idx];// NOTE: cudaEvent_t is a pointer type

    chunk_id_t wait_chunk_id = max(chunk_id - (chunk_id_t)MAX_PREFETCH_CHUNKS, chunk_id - (chunk_id_t)this->io_depth);
    if(wait_chunk_id >= 0) {
        // wait for the AsyncExecutor to be available to post tasks
        // and wait for existing usage of host_buffer
        this->wait_read_chunk(wait_chunk_id);
    }

    if(_env_debug()) {
        if (this->rank == 0 && this->prev_file_index != chunk.file_index) {
            this->prev_file_index = chunk.file_index;
            fprintf(stderr, "file_index = %zu\n", chunk.file_index);
        }
    }

    FileInfo &f = this->file_info[chunk.file_index];
    ChunkIOParams params{chunk_id, chunk, f, padded_world_chunk_size, padded_rank_size, padded_thread_size,
                         rank_offset, rank_size, window_idx, window_offset, rank_dst, all_dst, event};

    ChunkRequest result;
    if (f.in_memory) {
        result = this->post_read_chunk_inmem(params);
    } else if (this->use_cufile) {
        result = this->post_read_chunk_cufile(params);
    } else {
        result = this->post_read_chunk_aio(params);
    }

    this->chunks[chunk_id].request = result;
}

void Loader::poll_read_chunk() {
    chunk_id_t next_chunck_id = this->chunk_read.load(std::memory_order_relaxed) + 1;
    while(next_chunck_id <= this->chunk_reading.load(std::memory_order_relaxed)
        && this->chunks[next_chunck_id].request.executor->test(this->chunks[next_chunck_id].request.wait_handle)) {
        next_chunck_id++;
    }
    this->chunk_read.store(next_chunck_id - 1, std::memory_order_relaxed);
}

void Loader::wait_read_chunk(chunk_id_t chunk_id) {
    chunk_id_t next_chunck_id = this->chunk_read.load(std::memory_order_relaxed) + 1;
    if(chunk_id > this->chunk_reading.load(std::memory_order_relaxed)) {
        print_and_throw(std::runtime_error("Internal error: chunk_id out of range."));
    }
    while(next_chunck_id <= chunk_id) {
        this->chunks[next_chunck_id].request.executor->wait(this->chunks[next_chunck_id].request.wait_handle);
        next_chunck_id++;
    }
    this->chunk_read.store(next_chunck_id - 1, std::memory_order_relaxed);
}

void Loader::step() {
    this->post_read_chunk();
    this->poll_read_chunk();
}

bool Loader::can_step() {
    if(this->tensors.size() == 0) { // not opened
        return false;
    }
    if(this->current_tensor_index >= this->tensors.size()) {
        print_and_throw(std::runtime_error("Internal error: current_tensor_index (" + std::to_string(this->current_tensor_index) + ") out of range (" + std::to_string(this->tensors.size()) + ")."));
    }
    TensorMetadate& tensor = this->tensors[this->current_tensor_index];
    chunk_id_t prefetch_chunk_id = tensor.prefetch_chunk_id;

    if(this->chunk_reading.load(std::memory_order_relaxed) > prefetch_chunk_id) {
        print_and_throw(std::runtime_error("Internal error: chunk_reading is greater than prefetch_chunk_id."));
    }
    return this->chunk_reading.load(std::memory_order_relaxed) < prefetch_chunk_id;
}

void Loader::try_step() {
    if(this->can_step()) {
        this->step();
    }
    else {
        this->poll_read_chunk();
    }
}

void* Loader::get_tensor_ptr(GetTensorArgs args) {
    auto index = args.tensor_index;

    if (index >= this->tensors.size()) {
        print_and_throw(std::runtime_error("Internal error: index out of range."));
    }

    if (index < this->current_tensor_index) {
        print_and_throw(std::runtime_error("Should call get_tensor_ptr(key) with keys in order of keys()."));
    }
    if (index > this->current_tensor_index + 1) {
        fprintf(stderr, "WARNING: index jumped from %zu to %zu\n", this->current_tensor_index, index);
    }
    this->current_tensor_index = index;

    TensorMetadate& tensor = this->tensors[this->current_tensor_index];
    chunk_id_t last_chunk_id = tensor.last_chunk_id;

    while (this->chunk_read.load(std::memory_order_relaxed) < last_chunk_id) {
        if(this->can_step()) {
            this->step();
        }
        else {
            this->wait_read_chunk(last_chunk_id);
        }
    }

    return (char*)this->device_buffer + this->tensors[index].device_buffer_offset;
}

std::any Loader::dispatch(const RPCRequest &m) {
    switch (m.op) {
        case OPEN:
            this->open(std::any_cast<OpenArgs>(m.args));
            return std::any();
        case CLOSE:
            this->close(std::any_cast<CloseArgs>(m.args));
            return std::any();
        case GET_TENSOR_PTR:
            void* ret = this->get_tensor_ptr(std::any_cast<GetTensorArgs>(m.args));
            return std::any(ret);
    }
    print_and_throw(std::runtime_error("Invalid operation"));
}

void Loader::run() {
    while (!this->stop.load(std::memory_order_relaxed)) {
        if (!this->can_step() || !this->input_queue->empty()) {
            RPCRequest m;
            this->input_queue->pop(m);
            std::any ret = this->dispatch(m);
            this->output_queue->push(RPCResponse{m.id, std::move(ret)});
        }
        this->try_step();
    }
}

void run_loader(unique_ptr<SPSCQueue<RPCRequest>> input_queue, unique_ptr<SPSCQueue<RPCResponse>> output_queue) {
    Loader loader(std::move(input_queue), std::move(output_queue));
    loader.run();
}

} // namespace instanttensor
