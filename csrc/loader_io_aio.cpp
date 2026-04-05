#include <instant_tensor/loader.hpp>

namespace instanttensor {

void Loader::open_file_aio(FileInfo &f) {
    f.fd = ::open(f.filename.c_str(), O_RDONLY | O_DIRECT);
    if (f.fd < 0) {
        throw std::runtime_error("Failed to open file: " + f.filename);
    }
    struct stat st;
    if (fstat(f.fd, &st) < 0) { throw std::runtime_error("Failed to fstat file: " + f.filename); }
    f.size = st.st_size;

    this->need_aio = true;
    this->need_host_buffer = true;
    this->need_cuda_thread = true;
}

void Loader::open_file_aio_context() {
    int ret = io_setup(this->io_depth * this->num_threads, &this->aio_ctx);
    if(ret < 0){
        print_and_throw(std::runtime_error("Failed to setup aio: " + std::string(strerror(-ret))));
    }
    this->aio_iocbs.resize(this->io_depth * this->num_threads);
    this->aio_iocb_ptrs.resize(this->io_depth * this->num_threads);
    for(size_t i = 0; i < this->io_depth * this->num_threads; i++) {
        this->aio_iocb_ptrs[i] = &this->aio_iocbs[i];
    }
    this->aio_events.resize(this->io_depth * this->num_threads);
}

void Loader::close_file_aio(FileInfo &f) {
    ::close(f.fd);
}

void Loader::close_file_aio_context() {
    int ret = io_destroy(this->aio_ctx);
    if(ret < 0){
        print_and_throw(std::runtime_error("Failed to destroy aio: " + std::string(strerror(-ret))));
    }
}

ChunkRequest Loader::post_read_chunk_aio(const ChunkIOParams &p) {
    chunk_id_t chunk_id = p.chunk_id;
    size_t submit_cnt = 0;
    size_t window_idx = p.window_idx;

    bool unaligned_last_page = false;

    for(size_t i = 0; i < this->num_threads; i++) {
        size_t thread_offset = p.padded_thread_size * i;
        // thread_size <= thread_size_aligned <= padded_thread_size
        size_t thread_size = std::min((size_t)std::max((ssize_t)(p.chunk.size - p.rank_offset - thread_offset), (ssize_t)0), p.padded_thread_size);
        size_t thread_size_aligned = ROUND_UP(thread_size, this->thread_alignment);
        if(thread_size != thread_size_aligned) unaligned_last_page = true;
        if(thread_size == 0) continue;
        struct iocb *iocb = this->aio_iocb_ptrs[window_idx * this->num_threads + i];
        size_t file_offset = p.chunk.file_offset + p.rank_offset + thread_offset;
        // NOTE: aio needs the read size aligned to PAGE_SIZE
        io_prep_pread(iocb, p.file.fd, (char*)this->host_buffer + p.window_offset + thread_offset, thread_size_aligned, file_offset);
        iocb->data = (void*)chunk_id;
        submit_cnt ++;
    }
    this->chunks[chunk_id].extra_data.aio_unfinished_cnt = submit_cnt;

    // NOTE: This will block at the last page of the file if the file is not page aligned.
    //       So we put the last page into another thread
    auto aio_func = [=]() {
        int ret = io_submit(this->aio_ctx, submit_cnt, this->aio_iocb_ptrs.data() + window_idx * this->num_threads);
        if(ret < 0){
            print_and_throw(std::runtime_error("Failed to submit aio: " + std::string(strerror(-ret))));
        }
    };

    int aio_req_id = 0;
    if(unaligned_last_page) {
        aio_req_id = this->aio_fallback_thread->post(std::move(aio_func));
    }
    else {
        aio_func();
    }
    

    void *rank_mid = (char*)this->host_buffer + p.window_offset;
    void *rank_dst = p.rank_dst;
    void *all_dst = p.all_dst;
    size_t rank_size = p.rank_size;
    size_t padded_rank_size = p.padded_rank_size;
    cudaEvent_t event = p.event;
    auto cuda_func = [=]() {
        if(unaligned_last_page) {
            this->aio_fallback_thread->pop(aio_req_id);
        }
        // disk to host
        size_t &unfinished_cnt = this->chunks[chunk_id].extra_data.aio_unfinished_cnt;
        while(unfinished_cnt > 0) {
            int got = io_getevents(this->aio_ctx, unfinished_cnt, unfinished_cnt, this->aio_events.data(), NULL);
            // got may < min_nr and >= 0 if interrupted
            if(got < 0){
                print_and_throw(std::runtime_error("Failed to get aio events: " + std::string(strerror(-got))));
            }
            for(int i = 0; i < got; i++) {
                // We do not check the expected return value of this->aio_events[i].res here.
                // We believe checking only for errors (this->aio_events[i].res < 0) is enough.
                if(this->aio_events[i].res < 0) {
                    print_and_throw(std::runtime_error("Failed to get aio events: " + std::string(strerror(-this->aio_events[i].res))));
                }
                // NOTE: event_chunk_id can be different from chunk_id
                chunk_id_t event_chunk_id = (chunk_id_t)this->aio_events[i].data;
                this->chunks[event_chunk_id].extra_data.aio_unfinished_cnt --;
            }
        }

        // host to device
        CUDA_CHECK(cudaMemcpyAsync(rank_dst, rank_mid, rank_size, cudaMemcpyHostToDevice, this->cuda_stream));// 240GB/s for 8 GPUs
        CUDA_CHECK(cudaEventRecord(event, this->cuda_stream));
        if(this->world_size > 1) {
            CUDA_CHECK(cudaStreamWaitEvent(this->nccl_stream, event));
            NCCL_CHECK(ncclAllGather(rank_dst, all_dst, padded_rank_size, ncclInt8, this->group_communicator, this->nccl_stream));// 320GB/s for 8 GPUs
            CUDA_CHECK(cudaEventRecord(event, this->nccl_stream));
        }
    };

    int cuda_req_id = this->cuda_thread->post(std::move(cuda_func));

    auto wait_func = [=]() mutable {
        this->cuda_thread->pop(cuda_req_id);
        CUDA_CHECK(cudaEventSynchronize(event));
    };

    int completion_req_id = this->wait_thread->post(std::move(wait_func));
    AsyncExecutor *completion_thread = this->wait_thread.get();

    return ChunkRequest{completion_thread, completion_req_id};
}

} // namespace instanttensor
