import time
import json
import torch # must before instanttensor._C
import torch.distributed as dist
import instanttensor._C
from typing import Union, List
import threading
import io

import atexit
import os 

atexit.register(instanttensor._C.cleanup)


def safetensors_dtype_to_torch_dtype(dtype: str):
    # "bool" => Dtype::BOOL,
    # "int8" => Dtype::I8,
    # "uint8" => Dtype::U8,
    # "int16" => Dtype::I16,
    # "uint16" => Dtype::U16,
    # "int32" => Dtype::I32,
    # "uint32" => Dtype::U32,
    # "int64" => Dtype::I64,
    # "uint64" => Dtype::U64,
    # "float16" => Dtype::F16,
    # "float32" => Dtype::F32,
    # "float64" => Dtype::F64,
    # "bfloat16" => Dtype::BF16,
    # "float8_e4m3fn" => Dtype::F8_E4M3,
    # "float8_e5m2" => Dtype::F8_E5M2,
    # "float8_e8m0fnu" => Dtype::F8_E8M0,
    # "float4_e2m1fn_x2" => Dtype::F4,
    if dtype == "BOOL":
        return torch.bool
    elif dtype == "I8":
        return torch.int8
    elif dtype == "U8":
        return torch.uint8
    elif dtype == "I16":
        return torch.int16
    elif dtype == "U16":
        return torch.uint16
    elif dtype == "I32":
        return torch.int32
    elif dtype == "U32":
        return torch.uint32
    elif dtype == "I64":
        return torch.int64
    elif dtype == "U64":
        return torch.uint64
    elif dtype == "F16":
        return torch.float16
    elif dtype == "F32":
        return torch.float32
    elif dtype == "F64":
        return torch.float64
    elif dtype == "BF16":
        return torch.bfloat16
    elif dtype == "F8_E4M3":
        return torch.float8_e4m3fn
    elif dtype == "F8_E5M2":
        return torch.float8_e5m2
    elif dtype == "F8_E8M0":
        return torch.float8_e8m0fnu
    elif dtype == "F4":
        return torch.float4_e2m1fn_x2
    else:
        raise ValueError(f"Safetensors does not support dtype: {dtype}")

def read_safetensors_metadata(filename: str):
    with open(filename, "rb") as f:
        metadata_size = int.from_bytes(f.read(8), "little")
        metadata_str = f.read(metadata_size).decode("utf-8")
        tensor_metadata = json.loads(metadata_str)
        file_metadata = tensor_metadata.pop("__metadata__", None)
        return file_metadata, tensor_metadata, 8 + metadata_size

# This will be called lazily in safe_open(), but can be explicitly called to control the timing of initialization
def init(): 
    instanttensor._C.init() 

def file_in_memory(filename: str):
    return instanttensor._C.file_in_memory(filename)

group_communicator_cache = {}

class safe_open:
    """
    Opens a safetensors lazily and returns tensors as asked
    To get the best performance:
    1. pass multiple files through a filename list instead of calling safe_open multiple times.
    2. set the buffer_size as large as possible.
    3. set the chunk_size large enough to saturate the loading bandwidth, 
       but not too large to 
    Setting the 

    Args:
        filename (`str`, or `List[str]`):
            The filename(s) to open

        framework (`str`):
            The framework you want you tensors in. 
            Supported values: `pt`, `tf`, `flax`, `numpy`.
            Currently only `pt` is supported.

        device (`str`, or `torch.device`):
            The device on which you want the tensors. Must be a CUDA device.

        process_group (`Any`, or `None`, defaults to `None`):
            Process group from torch.distributed, or `None` for single-process usage.

        buffer_size (`int`, defaults to `1024*1024*1024`):
            The buffer size in bytes on the device. 
            This may be further enlarged or shrunk to imporve the performance or avoid memory waste.

        chunk_size (`int`, defaults to None):
            Size of each file IO in bytes.
            Leave it as None to use the automatically determined value.
        
        concurrency (`int`, defaults to None):
            Number of concurrent IOs for the loading.
            Leave it as None to use the automatically determined value.

        load_now (`bool`, defaults to `True`):
            Whether to load the tensors immediately.
            For testing and debugging purposes.
            If True, it starts to load the tensors immediately.
            If False, it only reads the file metadata, and the tensors will be loaded when the context manager is entered.
            
    """
    def __init__(self, filename: Union[str, List[str]], framework: str, 
            device: Union[str, torch.device], process_group=None, 
            buffer_size=1*1024*1024*1024, chunk_size=None, concurrency=None, load_now=True): 
        self.init_time = time.perf_counter()

        if isinstance(filename, str):
            filename = [filename]

        filename.sort()
        
        all_file_in_memory = all(file_in_memory(file) for file in filename)

        self.world_size = 1 if process_group is None else dist.get_world_size(process_group)
        self.rank = 0 if process_group is None else dist.get_rank(process_group)

        if all_file_in_memory:
            if chunk_size is None:
                chunk_size = 2*1024*1024
            if concurrency is None:
                concurrency = max(min(32, os.cpu_count()) // self.world_size, 1)
            io_depth = 3 # memcpy + cudaMemcpyAsync + ncclAllGather
        else:
            if os.environ.get("INSTANTTENSOR_USE_CUFILE", "0") == "1":
                if chunk_size is None:
                    chunk_size = 4*1024*1024
                if concurrency is None:
                    concurrency = max(64 // self.world_size, 1) # 64 # OK if os.cpu_count() < 64 since threads are blocked by cuFileRead
                io_depth = 2 # cuFileRead + ncclAllGather
            else: # aio
                if chunk_size is None:
                    chunk_size = 4*1024*1024
                if concurrency is None:
                    concurrency = max(32 // self.world_size, 1)
                io_depth = 3 # aio read + cudaMemcpyAsync + ncclAllGather

        device = torch.device(device)
        assert device.type == "cuda", "InstantTensor only supports CUDA devices for now"
        assert framework == "pt", "InstantTensor only supports pytorch for now"

        self.filename = filename
        self.framework = framework
        self.device = device
        self.device_idx = device.index
        self.process_group = process_group
        self.buffer_size = buffer_size
        self.chunk_size = chunk_size
        self.concurrency = concurrency
        self.io_depth = io_depth
        self.loader_handle = None
        self.expect_index = 0

        self.ordered_key_values = []
        self.tensor_offsets = []
        
        self.sync_time = time.perf_counter()
        if self.process_group is not None:
            # Warm up nccl to ensure ncclComm_t is initialized
            # Even set async_op=True, the first call may still block to initialize ncclComm_t
            # Most of the time is spent on NCCL initialization rather than on the all_reduce itself.
            dist.all_reduce(torch.zeros(1, device=self.device), group=self.process_group) 
            # sync_work.wait()
            # print("ncclComm_t:", self.process_group._get_backend(self.device)._comm_ptr())

        self.meta_read_time = time.perf_counter()

        # t0 = time.perf_counter()
        meta_read_results = self.read_metadata()
        # print(f"Time: read_safetensors_metadata = {time.perf_counter() - t0:.2f}s")

        for f_idx, f in enumerate(self.filename):
            file_metadata, tensor_metadata, tensor_offset = meta_read_results[f_idx]
            assert file_metadata is None or file_metadata.get("format", "pt") == "pt", "InstantTensor only supports pytorch format for now"
            # A typical entry: "model.layers.20.post_attention_layernorm.weight":{"dtype":"BF16","shape":[2880],"data_offsets":[0,5760]}
            ordered_key_values = sorted(tensor_metadata.items(), key=lambda kv: kv[1]["data_offsets"][0])
            assert all(ordered_key_values[i][1]["data_offsets"][1] == ordered_key_values[i+1][1]["data_offsets"][0] for i in range(len(ordered_key_values) - 1))
            
            self.tensor_offsets.extend([(f_idx, v["data_offsets"][0] + tensor_offset) for k, v in ordered_key_values] + [(f_idx, ordered_key_values[-1][1]["data_offsets"][1] + tensor_offset)])
            self.ordered_key_values.extend(ordered_key_values)
        

        self.key_to_index = {k: i for i, (k, v) in enumerate(self.ordered_key_values)}

        # adjust buffer size    
        tensor_sizes = [v["data_offsets"][1] - v["data_offsets"][0] for k, v in self.ordered_key_values]
        self.total_tensor_size = sum(tensor_sizes)

        # make sure any two contiguous tensors will not be overlapped with each other in the buffer
        min_buffer_size = max(tensor_sizes[i]+2*tensor_sizes[i+1] for i in range(len(tensor_sizes) - 1))
        
        if self.buffer_size < min_buffer_size:
            print(f"Warning: Enlarge buffer size from {self.buffer_size} to {min_buffer_size} for better performance")
            self.buffer_size = min_buffer_size

        max_buffer_size = self.total_tensor_size
        if self.buffer_size > max_buffer_size:
            print(f"Warning: Shrink buffer size from {self.buffer_size} to {max_buffer_size} to avoid memory waste")
            self.buffer_size = max_buffer_size

        if load_now:
            self._open()

    def read_metadata(self):
        meta_read_threads = []
        meta_read_results = [None] * len(self.filename)

        # print(f"world_size = {self.world_size}, rank = {self.rank}")

        # meta_read_start = len(self.filename) // self.world_size * self.rank + min(self.rank, len(self.filename) % self.world_size)
        # meta_read_cnt = len(self.filename) // self.world_size + int(self.rank < len(self.filename) % self.world_size)
        # meta_read_end = meta_read_start + meta_read_cnt
        meta_read_start = 0 
        meta_read_end = len(self.filename)

        # print(f"meta_read = {meta_read_start}-{meta_read_end}")
        for f_idx, f in list(enumerate(self.filename))[meta_read_start:meta_read_end]:
            def read_safetensors_metadata_wrapper(f, result_idx):
                meta_read_results[result_idx] = read_safetensors_metadata(f)
            
            t = threading.Thread(target=read_safetensors_metadata_wrapper, args=(f, f_idx))
            t.start()
            meta_read_threads.append(t)

        for t in meta_read_threads:
            t.join()

        
        # if self.world_size > 1:
        #     tmp = [None for _ in range(self.world_size)]
        #     # import pickle
        #     # print(len(pickle.dumps(meta_read_results[meta_read_start:meta_read_end])))
        #     t0 = time.perf_counter()
        #     dist.all_gather_object(tmp, meta_read_results[meta_read_start:meta_read_end], self.process_group)
        #     t1 = time.perf_counter()
        #     meta_read_results = [item for sublist in tmp for item in sublist]
        #     print(f"Time: all_gather = {t1 - t0:.2f}s")

        return meta_read_results

    def _get_group_communicator(self):
        group_constructor = None
        group_communicator = None
        group_rank = 0
        group_world_size = 1
        if self.process_group is not None:
            group_rank = dist.get_rank(self.process_group)
            # global_rank = dist.get_global_rank(self.process_group, group_rank)
            group_world_size = dist.get_world_size(self.process_group)

            # cache the nccl communicator since ncclCommInitRank and ncclCommDestroy are very slow
            if group_world_size > 1:
                if self.process_group in group_communicator_cache:
                    group_communicator = group_communicator_cache[self.process_group]
                else:
                    if group_rank == 0:
                        group_constructor = instanttensor._C.create_group_constructor()
                        group_constructor_list = [group_constructor]
                    else:
                        group_constructor_list = [None]
                    src_global_rank = dist.get_global_rank(self.process_group, 0)
                    dist.broadcast_object_list(group_constructor_list, src=src_global_rank, group=self.process_group)
                    group_constructor = group_constructor_list[0]
                    with torch.cuda.device(self.device_idx): # Device should be set before calling ncclCommInitRank
                        group_communicator = instanttensor._C.create_group_communicator(group_constructor, group_world_size, group_rank)
                    group_communicator_cache[self.process_group] = group_communicator
        return group_communicator, group_rank, group_world_size

    def _open(self):
        # group_communicator, group_rank, group_world_size = self._get_group_communicator()
        # # print(group_world_size, group_communicator)

        # if dist.get_rank(self.process_group) == 0:
        #     import pdb; pdb.set_trace()
        # else:
        #     while True:
        #         time.sleep(1)
        self.open_time = time.perf_counter()
        self.loader_handle = instanttensor._C.open(
            self.filename, self.device_idx, self.process_group, self.buffer_size, 
            self.chunk_size, self.concurrency, self.io_depth, self.tensor_offsets)

    def __enter__(self):
        """
        Start the context manager
        """
        if self.loader_handle is None:
            self._open()
        self.enter_time = time.perf_counter()
        return self

    def __exit__(self, _exc_type, _exc_value, _traceback):
        """
        Exits the context manager
        """
        stream = torch.cuda.current_stream()
        stream.synchronize() # make sure all the data transfer is done
        self.exit_time = time.perf_counter()
        instanttensor._C.close(self.loader_handle)
        self.close_time = time.perf_counter()
        total_time = self.close_time - self.init_time
        init_time = self.sync_time - self.init_time
        sync_time = self.meta_read_time - self.sync_time
        meta_read_time = self.open_time - self.meta_read_time
        open_time = self.enter_time - self.open_time
        load_time = self.exit_time - self.enter_time
        close_time = self.close_time - self.exit_time
        print(f"Time: total={total_time:.2f}s, init={init_time:.2f}s, sync={sync_time:.2f}s, meta_read={meta_read_time:.2f}s, open={open_time:.2f}s, load={load_time:.2f}s, close={close_time:.2f}s")
        print(f"Throughput: total={self.total_tensor_size * 1e-9 / total_time:.2f}GB/s, load={self.total_tensor_size * 1e-9 / load_time:.2f}GB/s")

    # for experiment
    def get_shape(self, name):
        """
        Returns the size of the tensor
        """
        tensor_index = self.key_to_index[name]
        return self.ordered_key_values[tensor_index][1]["shape"]

    # for experiment
    def get_dtype(self, name):
        """
        Returns the dtype of the tensor
        """
        tensor_index = self.key_to_index[name]
        return safetensors_dtype_to_torch_dtype(self.ordered_key_values[tensor_index][1]["dtype"])

    # def get_slice(self, name):
    #     """
    #     Returns a full slice view object

    #     Args:
    #         name (`str`):
    #             The name of the tensor you want

    #     Returns:
    #         (`PySafeSlice`):
    #             A dummy object you can slice into to get a real tensor
    #     Example:
    #     ```python
    #     from safetensors import safe_open

    #     with safe_open("model.safetensors", framework="pt", device=0) as f:
    #         tensor_part = f.get_slice("embedding")[:, ::8]

    #     ```
    #     """
    #     pass

    def get_tensor(self, name):
        """
        Returns a full tensor

        Args:
            name (`str`):
                The name of the tensor you want

        Returns:
            (`Tensor`):
                The tensor in the framework you opened the file for.

        Example:
        ```python
        from safetensors import safe_open

        with safe_open("model.safetensors", framework="pt", device=0) as f:
            tensor = f.get_tensor("embedding")

        ```
        """
        # Since we gradually free old tensors in the buffer, we need to synchronize here to ensure the data transfer is finished on GPU.
        # TODO: Do we really need this?
        # NOTE: This synchronization seems to make the performance a little bit better, but the reason is not clear.
        stream = torch.cuda.current_stream()
        stream.synchronize()
        tensor_index = self.key_to_index[name]
        assert tensor_index == self.expect_index, f"get_tensor() should be called in the order of tensor names returned by keys()" 
        self.expect_index += 1
        shape = self.ordered_key_values[tensor_index][1]["shape"]
        dtype = safetensors_dtype_to_torch_dtype(self.ordered_key_values[tensor_index][1]["dtype"])
        tensor = instanttensor._C.get_tensor(self.loader_handle, tensor_index, shape, dtype)
        if tensor.data_ptr() % tensor.element_size() != 0:
            raise ValueError(f"Tensor {name} address {tensor.data_ptr():#x} is not aligned to dtype {dtype} size {tensor.element_size()}B")
        return tensor

    def keys(self):
        """
        Returns the names of the tensors in the file.

        Returns:
            (`List[str]`):
                The name of the tensors contained in that file
        """
        return self.offset_keys()

    def metadata(self):
        """
        Return the special non tensor information in the header

        Returns:
            (`Dict[str, str]`):
                The freeform metadata.
        """
        return self.file_metadata

    def offset_keys(self):
        """
        Returns the names of the tensors in the file, ordered by offset.

        Returns:
            (`List[str]`):
                The name of the tensors contained in that file
        """
        return [key for key, _ in self.ordered_key_values]