import os 
import time
import json
import torch # must before instanttensor._C
import torch.distributed as dist
import instanttensor._C
from typing import Union, Generator
import threading
import atexit
from collections import defaultdict


try:
    atexit.register(instanttensor._C.cleanup)
except AttributeError:
    # _C module is mocked (e.g., during Sphinx documentation build)
    print("instanttensor._C is mocked, skipping cleanup registration")

_env_debug = None
_env_use_cufile = None

def env_debug():
    global _env_debug
    if _env_debug is None:
        _env_debug = os.environ.get("INSTANTTENSOR_DEBUG", "0") == "1"
    return _env_debug

def env_use_cufile():
    global _env_use_cufile
    if _env_use_cufile is None:
        _env_use_cufile = os.environ.get("INSTANTTENSOR_USE_CUFILE", "0") == "1"
    return _env_use_cufile

def env_chunk_size():
    ret = os.environ.get("INSTANTTENSOR_CHUNK_SIZE")
    return int(ret) if ret is not None else None

def env_concurrency():
    ret = os.environ.get("INSTANTTENSOR_CONCURRENCY")
    return int(ret) if ret is not None else None

def env_io_depth():
    ret = os.environ.get("INSTANTTENSOR_IO_DEPTH")
    return int(ret) if ret is not None else None

def env_max_free_mem_usage():
    ret = os.environ.get("INSTANTTENSOR_MAX_FREE_MEM_USAGE")
    return float(ret) if ret is not None else None

def env_buffer_size():
    ret = os.environ.get("INSTANTTENSOR_BUFFER_SIZE")
    return int(ret) if ret is not None else None

# reference: https://github.com/run-ai/runai-model-streamer/blob/0.15.6/py/runai_model_streamer/runai_model_streamer/safetensors_streamer/safetensors_pytorch.py
def get_safetensors_dtype_map() -> dict:
    safetensors_to_torch_dtype = {
        "F64": torch.float64,
        "F32": torch.float32,
        "F16": torch.float16,
        "BF16": torch.bfloat16,
        "I64": torch.int64,
        "I32": torch.int32,
        "I16": torch.int16,
        "I8":  torch.int8,
        "U8":  torch.uint8,
        "BOOL": torch.bool,
        "C64": torch.complex64,
    }

    # Add unsigned types if available (PyTorch >= 2.3.0)
    for st_name, torch_name in [("U64", "uint64"), ("U32", "uint32"), ("U16", "uint16")]:
        if hasattr(torch, torch_name):
            safetensors_to_torch_dtype[st_name] = getattr(torch, torch_name)

    # Experimental types with their PyTorch attribute names
    # Note: If a type is listed here but not available in the current PyTorch version,
    # it won't be added to the type map. If a file contains such a dtype (e.g., "F4"),
    # get_torch_dtype() will raise a clear ValueError: "Unsupported dtype 'F4'".
    # This is correct forward-compatible behavior - fail fast with a clear error message.
    _EXPERIMENTAL_ALIASES = {
        "F8_E4M3": ["float8_e4m3fn", "float8_e4m3fnuz"],
        "F8_E5M2": ["float8_e5m2", "float8_e5m2fnuz"],
        "F8_E8M0": ["float8_e8m0fnu", "float8_e8m0fnuz"],
        "F4":      ["float4_e2m1fn_x2"],  # Not yet in PyTorch (as of 2.5.1)
        # FP6 is not supported by PyTorch yet
    }

    for st_type, torch_aliases in _EXPERIMENTAL_ALIASES.items():
        for alias in torch_aliases:
            if hasattr(torch, alias):
                safetensors_to_torch_dtype[st_type] = getattr(torch, alias)
                break

    return safetensors_to_torch_dtype

safetensors_to_torch_dtype = get_safetensors_dtype_map()



def read_safetensors_metadata(filename: str) -> tuple:
    """Read the safetensors metadata from a file.
    
    This function reads the header metadata from a safetensors file, which
    contains information about the tensors stored in the file.
    
    Args:
        filename: Path to the safetensors file to read.
    
    Returns:
        A tuple containing:
            - file_metadata (``dict`` or ``None``): File-level metadata if present
            - tensor_metadata (``dict``): Dictionary mapping tensor names to their
              metadata (shape, dtype, offsets, etc.)
            - header_size (``int``): Size of the metadata header in bytes (including the 8 bytes of metadata size)
    
    Raises:
        FileNotFoundError: If the specified file does not exist.
        ValueError: If the file format is invalid.
    
    Example:
        >>> file_meta, tensor_meta, header_size = read_safetensors_metadata("model.safetensors")
        >>> print(f"Found {len(tensor_meta)} tensors")
        >>> print(f"Header size: {header_size} bytes")
    """
    with open(filename, "rb") as f:
        metadata_size = int.from_bytes(f.read(8), "little")
        metadata_str = f.read(metadata_size).decode("utf-8")
        tensor_metadata = json.loads(metadata_str)
        file_metadata = tensor_metadata.pop("__metadata__", None)
        return file_metadata, tensor_metadata, 8 + metadata_size

def init():
    """Initialize the InstantTensor library.
    
    This function initializes the underlying C++ backend of InstantTensor.
    It is an optional function and will be called lazily when ``safe_open()`` is first used, but can be
    explicitly called to control the timing of initialization.
    
    Example:
        >>> import instanttensor
        >>> instanttensor.init()  # Explicit initialization
    """
    instanttensor._C.init() 

def file_in_memory(filename: str) -> bool:
    """Check if a file is located in an in-memory filesystem.
    
    This helper function determines whether a file is stored in a tmpfs or
    ramfs filesystem, which affects the I/O strategy used by InstantTensor.
    
    Args:
        filename: Path to the file to check.
    
    Returns:
        ``True`` if the file is in an in-memory filesystem (tmpfs/ramfs),
        ``False`` otherwise.
    
    Example:
        >>> if file_in_memory("model.safetensors"):
        ...     print("File is in memory, using optimized path")
        ... else:
        ...     print("File is on disk, using standard I/O")
    """
    return instanttensor._C.file_in_memory(filename)

def get_tensor_size(shape: list[int], dtype: torch.dtype) -> int:
    ret = torch.tensor([], dtype=dtype).element_size()
    for s in shape:
        ret *= s
    return ret

def recommended_buffer_size_for_tensors(tensor_sizes: list[int], overlap_factor: float = 0.9) -> int:
    """
    Compute the recommended buffer size for the given tensor sizes.

    Args:
        tensor_sizes: The sizes of the tensors.
        overlap_factor: How much tensor loading (in size) can be overlapped 
            with user processing if the user processes tensor at the same speed
            as we load.
    
    Returns:
        The recommended buffer size.
    """
    if len(tensor_sizes) == 0:
        return 4096
    
    max_tensor_size = max(tensor_sizes)
    overlapped_size_of_buffer_size = defaultdict(int)
    overlapped_size_of_buffer_size[tensor_sizes[0]] = tensor_sizes[0]

    for i in range(len(tensor_sizes)-1):
        tensor_size = tensor_sizes[i+1]
        expected_buffer_size = tensor_sizes[i] + 2 * tensor_sizes[i+1]
        overlapped_size_of_buffer_size[expected_buffer_size] += tensor_size
    
    buffer_sizes = sorted(overlapped_size_of_buffer_size.keys())
    total_tensor_size = sum(overlapped_size_of_buffer_size.values())
    total_overlapped_size = 0
    for buffer_size in buffer_sizes:
        total_overlapped_size += overlapped_size_of_buffer_size[buffer_size]
        if total_overlapped_size >= total_tensor_size * overlap_factor:
            return max(buffer_size, max_tensor_size)
    
    assert False, "Should not reach here"



group_communicator_cache = {}

class safe_open:
    """Context manager for lazily loading safetensors files with high performance.
    
    This class provides an ultra-fast, distributed safetensors loader that
    maximizes I/O throughput when moving model weights from safetensors files
    to GPU memory. It supports multiple I/O backends including GPUDirect
    Storage, legacy storage, and memory-based storage.
    
    Args:
        filename: The filename(s) to open. Can be a single file path (``str``) or
            a list of file paths (``list[str]``) for multi-file loading. When
            multiple files are provided, they are automatically sorted. Providing
            all files in a single list has better performance than calling 
            ``safe_open`` multiple times.
        framework: The framework you want tensors in. Currently only ``"pt"``
            (PyTorch) is supported.
        device: The device on which you want the tensors. Must be a CUDA device.
            Can be specified as an ``int`` (device ID), ``str`` (e.g., ``"cuda:0"``), or
            ``torch.device`` object.
        process_group: Process group from ``torch.distributed`` for distributed
            loading, or ``None`` for single-process usage. When provided, InstantTensor
            uses NCCL to coordinate loading across processes for higher throughput.
        buffer_size: The size of the GPU buffer used for tensors in bytes.
            If ``None`` (default), automatically determined based on tensor sizes
            for optimal performance. Larger values improve throughput but use
            more GPU memory.
        chunk_size: The size of each file I/O operation in bytes. If ``None``
            (default), automatically determined based on storage type.
            Increasing this value can improve throughput, but values that are
            too large may conversely reduce throughput.
        concurrency: The number of concurrent I/O operations. If ``None`` (default),
            automatically determined based on storage type and system capabilities.
            Increasing this value can improve throughput, but values that are
            too large may conversely reduce throughput.
        io_depth: The number of queued I/O operations per thread. If ``None`` (default),
            automatically determined based on storage type and system capabilities.
        max_free_mem_usage: Max ratio of idle memory used. If ``None`` (default), 0.5 is used.
        load_now: Whether to load tensors immediately. If ``True`` (default), starts
            loading immediately. If ``False``, only reads file metadata initially;
            tensors will be loaded when the context manager is entered. Useful
            for testing and debugging.
    
    Returns:
        A context manager that yields a file-like object with tensor access
        methods.
    
    Example:
        Basic single-file usage:
        
        >>> from instanttensor import safe_open
        >>> tensors = {}
        >>> with safe_open("model.safetensors", framework="pt", device=0) as f:
        ...     for name, tensor in f.tensors():
        ...         tensors[name] = tensor.clone()
        
        Multi-file loading (recommended for better performance):
        
        >>> files = ["model-00001-of-00002.safetensors",
        ...          "model-00002-of-00002.safetensors"]
        >>> with safe_open(files, framework="pt", device=0) as f:
        ...     for name, tensor in f.tensors():
        ...         tensors[name] = tensor.clone()
        
        Distributed loading:
        
        >>> import torch
        >>> import torch.distributed as dist
        >>> dist.init_process_group(backend="nccl")
        >>> process_group = dist.GroupMember.WORLD
        >>> with safe_open(files, framework="pt",
        ...                device=torch.cuda.current_device(),
        ...                process_group=process_group) as f:
        ...     for name, tensor in f.tensors():
        ...         tensors[name] = tensor.clone()
    """
    def __init__(self, filename: Union[str, list[str]], framework: str, 
            device: Union[int, str, torch.device], process_group=None, *,
            buffer_size=None, chunk_size=None, concurrency=None, io_depth=None, 
            max_free_mem_usage=None, load_now=True):
        """Initialize the safe_open context manager.
        
        See class docstring for detailed parameter descriptions.
        """ 
        self.init_time = time.perf_counter()

        if isinstance(filename, str):
            filename = [filename]

        filename.sort()

        device = torch.device(device)
        assert device.type == "cuda", "InstantTensor only supports CUDA devices for now"
        assert framework == "pt", "InstantTensor only supports pytorch for now"

        self.world_size = 1 if process_group is None else dist.get_world_size(process_group)
        self.rank = 0 if process_group is None else dist.get_rank(process_group)

        self.filename = filename
        self.framework = framework
        self.device = device
        self.device_idx = device.index
        self.process_group = process_group
        self.loader_handle = None
        self.distributed_metadata_read = False

        self.ordered_tensor_metadatas = []
        self.tensor_offsets = []
        self.iterated = False
        self.tmp_generator = None

        self._determine_io_params(chunk_size, concurrency, io_depth, max_free_mem_usage)

        self.meta_read_time = time.perf_counter()

        meta_read_results = self._read_metadata()

        self.file_metadata = None
        for f_idx, f in enumerate(self.filename):
            file_metadata, tensor_metadata, tensor_offset = meta_read_results[f_idx]
            if file_metadata is not None:
                self.file_metadata = file_metadata
            assert file_metadata is None or file_metadata.get("format", "pt") == "pt", "InstantTensor only supports pytorch format for now"
            # A typical entry: "model.layers.20.post_attention_layernorm.weight":{"dtype":"BF16","shape":[2880],"data_offsets":[0,5760]}
            ordered_tensor_metadatas = sorted(tensor_metadata.items(), key=lambda kv: kv[1]["data_offsets"][0])
            assert all(ordered_tensor_metadatas[i][1]["data_offsets"][1] == ordered_tensor_metadatas[i+1][1]["data_offsets"][0] for i in range(len(ordered_tensor_metadatas) - 1))
            
            self.tensor_offsets.extend([(f_idx, v["data_offsets"][0] + tensor_offset) for k, v in ordered_tensor_metadatas] + [(f_idx, ordered_tensor_metadatas[-1][1]["data_offsets"][1] + tensor_offset)])
            self.ordered_tensor_metadatas.extend(ordered_tensor_metadatas)
        

        self.tensor_name_to_index = {k: i for i, (k, v) in enumerate(self.ordered_tensor_metadatas)}

        # adjust buffer size    
        self.tensor_sizes = [v["data_offsets"][1] - v["data_offsets"][0] for k, v in self.ordered_tensor_metadatas]
        self.total_tensor_size = sum(self.tensor_sizes)

        self._determine_buffer_size(buffer_size)

        if load_now:
            self._open()

    def _determine_io_params(self, chunk_size, concurrency, io_depth, max_free_mem_usage):
        if chunk_size is None:
            chunk_size = env_chunk_size()
        if concurrency is None:
            concurrency = env_concurrency()
        if io_depth is None:
            io_depth = env_io_depth()
        if max_free_mem_usage is None:
            max_free_mem_usage = env_max_free_mem_usage()

        all_file_in_memory = all(file_in_memory(file) for file in self.filename)

        if all_file_in_memory:
            if chunk_size is None:
                chunk_size = 2*1024*1024
            if concurrency is None:
                concurrency = max(min(32, os.cpu_count()) // self.world_size, 1)
            if io_depth is None:
                io_depth = 3 # memcpy + cudaMemcpyAsync + ncclAllGather
        else:
            if env_use_cufile():
                if chunk_size is None:
                    chunk_size = 8*1024*1024
                if concurrency is None:
                    # Since these are all IO-intensive threads, using more threads than CPU cores is acceptable
                    concurrency = max(32 // self.world_size, 1) 
                if io_depth is None:
                    io_depth = 16 # cuFileRead + ncclAllGather # why this has effect?
            else: # aio
                if chunk_size is None:
                    chunk_size = 8*1024*1024
                if concurrency is None:
                    concurrency = 1 # max(1 // self.world_size, 1)
                if io_depth is None:
                    io_depth = max(512 // self.world_size, 3) # aio read + cudaMemcpyAsync + ncclAllGather
        
        if max_free_mem_usage is None:
            max_free_mem_usage = 0.5
        
        free_bytes, total_bytes = torch.cuda.mem_get_info()
        avail_bytes = int(free_bytes * max_free_mem_usage)

        self.sync_time = time.perf_counter()
        if self.process_group is not None:
            # Warm up nccl to ensure ncclComm_t is initialized
            # Even set async_op=True, the first call may still block to initialize ncclComm_t
            # Most of the time is spent on NCCL initialization rather than on the all_reduce itself.
            avail_bytes_tensor = torch.tensor([avail_bytes], device=self.device)
            dist.all_reduce(avail_bytes_tensor, op=torch.distributed.ReduceOp.MIN, group=self.process_group) 
            avail_bytes = avail_bytes_tensor.item()
            # print("ncclComm_t:", self.process_group._get_backend(self.device)._comm_ptr())


        if chunk_size * concurrency * io_depth * self.world_size > avail_bytes:
            shrinked_io_depth = max(avail_bytes // (chunk_size * concurrency * self.world_size), 3)
            if shrinked_io_depth != io_depth:
                print(f"Warning: Shrink io_depth from {io_depth} to {shrinked_io_depth} due to memory limit")
                io_depth = shrinked_io_depth
        
        if chunk_size * concurrency * io_depth * self.world_size > avail_bytes:
            shrinked_concurrency = max(avail_bytes // (chunk_size * io_depth * self.world_size), 1)
            if shrinked_concurrency != concurrency:
                print(f"Warning: Shrink concurrency from {concurrency} to {shrinked_concurrency} due to memory limit")
                concurrency = shrinked_concurrency

        assert chunk_size * concurrency * io_depth * self.world_size <= avail_bytes, "Device memory is not enough"

        self.chunk_size = chunk_size
        self.concurrency = concurrency
        self.io_depth = io_depth

    def _determine_buffer_size(self, buffer_size):
        if buffer_size is None:
            buffer_size = env_buffer_size()
        
        if buffer_size is None:
            # make sure any two contiguous tensors will not be overlapped with each other in the buffer
            buffer_size_for_tensors = recommended_buffer_size_for_tensors(self.tensor_sizes)
            buffer_size_for_io = self.chunk_size * self.concurrency * self.io_depth * self.world_size
            self.buffer_size = max(buffer_size_for_tensors, buffer_size_for_io)
        else:
            self.buffer_size = buffer_size
            min_buffer_size = max(self.tensor_sizes)
            if self.buffer_size < min_buffer_size:
                print(f"Warning: Enlarge buffer size from {self.buffer_size} to {min_buffer_size} to match the largest tensor size.")
                self.buffer_size = min_buffer_size

            max_buffer_size = self.total_tensor_size
            if self.buffer_size > max_buffer_size:
                print(f"Warning: Shrink buffer size from {self.buffer_size} to {max_buffer_size} to avoid memory waste")
                self.buffer_size = max_buffer_size

    def _read_metadata(self):
        meta_read_threads = []
        meta_read_results = [None] * len(self.filename)

        if self.distributed_metadata_read: # slower due to all_gather
            print(f"world_size = {self.world_size}, rank = {self.rank}")
            meta_read_start = len(self.filename) // self.world_size * self.rank + min(self.rank, len(self.filename) % self.world_size)
            meta_read_cnt = len(self.filename) // self.world_size + int(self.rank < len(self.filename) % self.world_size)
            meta_read_end = meta_read_start + meta_read_cnt
            print(f"meta_read = {meta_read_start}-{meta_read_end}")
        else:
            meta_read_start = 0 
            meta_read_end = len(self.filename)
        
        for f_idx, f in list(enumerate(self.filename))[meta_read_start:meta_read_end]:
            def read_safetensors_metadata_wrapper(f, result_idx):
                meta_read_results[result_idx] = read_safetensors_metadata(f)
            
            t = threading.Thread(target=read_safetensors_metadata_wrapper, args=(f, f_idx))
            t.start()
            meta_read_threads.append(t)

        for t in meta_read_threads:
            t.join()

        
        if self.distributed_metadata_read and self.world_size > 1:
            tmp = [None for _ in range(self.world_size)]
            # import pickle
            # print(len(pickle.dumps(meta_read_results[meta_read_start:meta_read_end])))
            t0 = time.perf_counter()
            dist.all_gather_object(tmp, meta_read_results[meta_read_start:meta_read_end], self.process_group)
            t1 = time.perf_counter()
            meta_read_results = [item for sublist in tmp for item in sublist]
            print(f"Time: all_gather = {t1 - t0:.2f}s")

        return meta_read_results

    def _open(self):
        self.open_time = time.perf_counter()
        nccl_communicator = self.process_group._get_backend(self.device)._comm_ptr() if self.process_group is not None else 0
        self.loader_handle = instanttensor._C.open(
            self.filename, self.device_idx, nccl_communicator, self.buffer_size, 
            self.chunk_size, self.concurrency, self.io_depth, self.tensor_offsets)

    def __enter__(self) -> 'safe_open':
        if self.loader_handle is None:
            self._open()
        self.enter_time = time.perf_counter()
        return self

    def __exit__(self, _exc_type, _exc_value, _traceback) -> None:
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
        if env_debug():
            print(f"Time: total={total_time:.2f}s, init={init_time:.2f}s, sync={sync_time:.2f}s, meta_read={meta_read_time:.2f}s, open={open_time:.2f}s, load={load_time:.2f}s, close={close_time:.2f}s")
            print(f"Throughput: total={self.total_tensor_size * 1e-9 / total_time:.2f}GB/s, load={self.total_tensor_size * 1e-9 / load_time:.2f}GB/s")

    def tensors(self) -> Generator[tuple[str, torch.Tensor], None, None]:
        """Iterate over all tensors in the safetensors file(s).
        
        This method returns an iterator that yields (name, tensor) pairs for
        all tensors from the safetensors file(s) that are loaded on the
        specified GPU device.

        Note:
            This method synchronizes CUDA streams to ensure data transfer
            completion.
        
        Yields:
            tuple: A (name, tensor) pair where:
                - name (``str``): The name/key of the tensor
                - tensor (``torch.Tensor``): The tensor data on the specified device
        
        Example:
            >>> with safe_open("model.safetensors", framework="pt", device=0) as f:
            ...     for name, tensor in f.tensors():
            ...         print(f"{name}: {tensor.shape}, {tensor.dtype}")
            ...         # Important: copy if storing for later use
            ...         stored_tensor = tensor.clone()

        Warning:
            The tensors returned by ``tensors()`` point to an internal buffer that
            is reused during iteration. You must copy each tensor (e.g., using
            ``.clone()`` or ``.copy_()``) if you need to use it after the current
            iteration completes. Otherwise, the tensor data may be overwritten by
            subsequent iterations, leading to incorrect results.
        """
        if self.iterated:
            raise RuntimeError("tensors() can only be called once")
        self.iterated = True
        for tensor_index, (name, metadata) in enumerate(self.ordered_tensor_metadatas):
            stream = torch.cuda.current_stream()
            stream.synchronize()
            shape = metadata["shape"]
            safetensors_dtype = metadata["dtype"]
            torch_dtype = safetensors_to_torch_dtype.get(safetensors_dtype, None)
            if torch_dtype is None:
                raise ValueError(f"Unsupported safetensors dtype: {safetensors_dtype}")
            
            tensor_size = get_tensor_size(shape, torch_dtype)
            dl_tensor = instanttensor._C.get_dl_tensor(self.loader_handle, tensor_index, tensor_size) # always returns int8 tensor
            tensor_int8 = torch.from_dlpack(dl_tensor)
            tensor = tensor_int8.view(torch_dtype).reshape(shape)
            
            if tensor.data_ptr() % tensor.element_size() != 0:
                raise ValueError(f"Tensor {name} address {tensor.data_ptr():#x} is not aligned to dtype {torch_dtype} size {tensor.element_size()}B")
            yield name, tensor

    def get_tensor(self, name: str) -> torch.Tensor:
        """Safetensors-compatible API: get a specific tensor by name from the safetensors file(s).
        
        This method retrieves a single tensor by its name. Random access is not supported;
        tensors must be retrieved sequentially in the order returned by keys().
        
        Note:
            It is recommended to use ``tensors()`` directly instead of this method.
        
        Args:
            name: The name/key of the tensor to retrieve. Must match the next
                tensor name in the sequence returned by keys().
        
        Returns:
            The tensor as a ``torch.Tensor`` on the specified device. The tensor
            points to an internal buffer and should be copied (using ``.clone()`` 
            or ``.copy_()``) if used outside the current iteration.
        
        Raises:
            ValueError: If the requested tensor name does not match the expected
                name (i.e., tensors are not retrieved in the order returned by
                ``keys()``).
        
        Example:
            Compatible usage with safetensors API:
            
            >>> from instanttensor import safe_open
            >>> 
            >>> with safe_open("model.safetensors", framework="pt", device=0) as f:
            ...     # Must get tensors in the exact order returned by keys()
            ...     for key in f.keys():
            ...         tensor = f.get_tensor(key)
            ...         # Important: copy the tensor if storing for later use
            ...         stored_tensor = tensor.clone()
            ...         print(f"{key}: {stored_tensor.shape}")
        
        Warning:
            Tensors must be retrieved in the exact order returned by ``keys()``.
            Calling ``get_tensor()`` with a name that doesn't match the next expected
            tensor will raise a ``ValueError``.
        """
        if self.tmp_generator is None:
            self.tmp_generator = self.tensors()
        expect_name, tensor = next(self.tmp_generator)
        if name != expect_name:
            raise ValueError(f"get_tensor() should be called in the order of tensor names returned by keys()")
        return tensor

    def keys(self) -> list[str]:
        """Safetensors-compatible API: get the names of all tensors in the safetensors file(s).
        
        This is an alias for ``offset_keys()`` that returns tensor names in the
        order they appear in the file (by offset).
        
        Returns:
            A list of tensor names (keys) in the order they appear in the file.
        
        Example:
            >>> with safe_open("model.safetensors", framework="pt", device=0) as f:
            ...     tensor_names = f.keys()
            ...     print(f"Found {len(tensor_names)} tensors")
            ...     for name in tensor_names:
            ...         tensor = f.get_tensor(name)
        """
        return self.offset_keys()

    def metadata(self) -> dict:
        """Safetensors-compatible API: get the file-level metadata from the safetensors file(s).
        
        This method returns the special non-tensor information stored in the
        safetensors file header (under the ``"__metadata__"`` key).
        
        Returns:
            A dictionary containing file-level metadata, or ``None`` if no
            metadata is present in the file.
        
        Example:
            >>> with safe_open("model.safetensors", framework="pt", device=0) as f:
            ...     meta = f.metadata()
            ...     if meta:
            ...         print(f"File format: {meta.get('format', 'pt')}")
        """
        return dict(self.file_metadata)

    def offset_keys(self) -> list[str]:
        """Safetensors-compatible API: get the names of all tensors, ordered by their offset in the file.
        
        This method returns tensor names in the order they appear in the
        safetensors file(s), sorted by their data offset. This is the order
        in which tensors should be retrieved using ``get_tensor()`` for optimal
        performance.
        
        Returns:
            A list of tensor names (keys) ordered by their data offset in
            the file(s).
        
        Example:
            >>> with safe_open("model.safetensors", framework="pt", device=0) as f:
            ...     # Get keys in offset order
            ...     keys = f.offset_keys()
            ...     for key in keys:
            ...         tensor = f.get_tensor(key)  # Must be in this order
        """
        return [key for key, _ in self.ordered_tensor_metadatas]

    def get_tensor_metadata(self, name: str) -> tuple[torch.dtype, torch.Size]:
        """Get the metadata (dtype and shape) of a specific tensor by name from the safetensors file(s).
        
        This method provides compatibility with the safetensors library API.
        It retrieves the metadata of a single tensor by its name.
        
        Args:
            name: The name/key of the tensor to retrieve metadata for.
        
        Returns:
            A tuple containing the dtype and shape of the tensor.

        Example:
            >>> with safe_open("model.safetensors", framework="pt", device=0) as f:
            ...     tensor_names = f.keys()
            ...     for name in tensor_names:
            ...         dtype, shape = f.get_tensor_metadata(name)
            ...         print(f"Tensor {name} has dtype {dtype} and shape {shape}")
        """
        tensor_metadata = self.ordered_tensor_metadatas[self.tensor_name_to_index[name]][1]
        torch_dtype = safetensors_to_torch_dtype.get(tensor_metadata["dtype"], None)
        if torch_dtype is None:
            raise ValueError(f"Unsupported safetensors dtype: {tensor_metadata['dtype']}")
        return torch_dtype, torch.Size(tensor_metadata["shape"])
