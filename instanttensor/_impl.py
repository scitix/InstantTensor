import os 
import time
import json
import torch # must before instanttensor._C
import torch.distributed as dist
import instanttensor._C
from typing import Union, Generator
import threading
import atexit


try:
    atexit.register(instanttensor._C.cleanup)
except AttributeError:
    # _C module is mocked (e.g., during Sphinx documentation build)
    print("instanttensor._C is mocked, skipping cleanup registration")

_instanttensor_debug = None
_instanttensor_use_cufile = None

def instanttensor_debug():
    global _instanttensor_debug
    if _instanttensor_debug is None:
        _instanttensor_debug = os.environ.get("INSTANTTENSOR_DEBUG", "0") == "1"
    return _instanttensor_debug

def instanttensor_use_cufile():
    global _instanttensor_use_cufile
    if _instanttensor_use_cufile is None:
        _instanttensor_use_cufile = os.environ.get("INSTANTTENSOR_USE_CUFILE", "0") == "1"
    return _instanttensor_use_cufile

def safetensors_dtype_to_torch_dtype(dtype: str) -> torch.dtype:
    """Convert a safetensors dtype string to the corresponding PyTorch dtype.
    
    Args:
        dtype: The safetensors dtype string. Supported values include:
            - "BOOL" -> ``torch.bool``
            - "I8", "I16", "I32", "I64" -> ``torch.int8``, ``torch.int16``, ``torch.int32``, ``torch.int64``
            - "U8", "U16", "U32", "U64" -> ``torch.uint8``, ``torch.uint16``, ``torch.uint32``, ``torch.uint64``
            - "F16", "F32", "F64" -> ``torch.float16``, ``torch.float32``, ``torch.float64``
            - "BF16" -> ``torch.bfloat16``
            - "F8_E4M3", "F8_E5M2" -> ``torch.float8`` variants
            - "F8_E8M0" -> ``torch.float8_e8m0fnu``
            - "F4" -> ``torch.float4_e2m1fn_x2``
    
    Returns:
        The corresponding PyTorch dtype object (e.g., ``torch.float32``).
    
    Raises:
        ValueError: If the dtype string is not supported by safetensors.
    
    Example:
        >>> dtype = safetensors_dtype_to_torch_dtype("F32")
        >>> print(dtype)
        torch.float32
    """
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
            device: Union[int, str, torch.device], process_group=None, 
            buffer_size=None, chunk_size=None, concurrency=None, load_now=True):
        """Initialize the safe_open context manager.
        
        See class docstring for detailed parameter descriptions.
        """ 
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
            if instanttensor_use_cufile():
                if chunk_size is None:
                    chunk_size = 4*1024*1024
                if concurrency is None:
                    # Since these are all IO-intensive threads, using more threads than CPU cores is acceptable
                    concurrency = max(64 // self.world_size, 1) 
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
        self.distributed_metadata_read = False

        self.ordered_tensor_metadatas = []
        self.tensor_offsets = []
        self.iterated = False
        self.tmp_generator = None
        
        self.sync_time = time.perf_counter()
        if self.process_group is not None:
            # Warm up nccl to ensure ncclComm_t is initialized
            # Even set async_op=True, the first call may still block to initialize ncclComm_t
            # Most of the time is spent on NCCL initialization rather than on the all_reduce itself.
            dist.all_reduce(torch.zeros(1, device=self.device), group=self.process_group) 
            # print("ncclComm_t:", self.process_group._get_backend(self.device)._comm_ptr())

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
        tensor_sizes = [v["data_offsets"][1] - v["data_offsets"][0] for k, v in self.ordered_tensor_metadatas]
        self.total_tensor_size = sum(tensor_sizes)

        if self.buffer_size is None:
            # make sure any two contiguous tensors will not be overlapped with each other in the buffer
            if len(tensor_sizes) >= 2:
                recommended_buffer_size = max(tensor_sizes[i]+2*tensor_sizes[i+1] for i in range(len(tensor_sizes) - 1))
            elif len(tensor_sizes) == 1:
                recommended_buffer_size = tensor_sizes[0]
            else:
                recommended_buffer_size = 4096 # avoid setting it to 0 to avoid potential issues
            self.buffer_size = recommended_buffer_size
        else:
            min_buffer_size = max(tensor_sizes)
            if self.buffer_size < min_buffer_size:
                print(f"Warning: Enlarge buffer size from {self.buffer_size} to {min_buffer_size} to match the largest tensor size.")
                self.buffer_size = min_buffer_size

            max_buffer_size = self.total_tensor_size
            if self.buffer_size > max_buffer_size:
                print(f"Warning: Shrink buffer size from {self.buffer_size} to {max_buffer_size} to avoid memory waste")
                self.buffer_size = max_buffer_size

        if load_now:
            self._open()

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
        self.loader_handle = instanttensor._C.open(
            self.filename, self.device_idx, self.process_group, self.buffer_size, 
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
        if instanttensor_debug():
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
            dtype = safetensors_dtype_to_torch_dtype(metadata["dtype"])
            tensor = instanttensor._C.get_tensor(self.loader_handle, tensor_index, shape, dtype)
            if tensor.data_ptr() % tensor.element_size() != 0:
                raise ValueError(f"Tensor {name} address {tensor.data_ptr():#x} is not aligned to dtype {dtype} size {tensor.element_size()}B")
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
        return safetensors_dtype_to_torch_dtype(tensor_metadata["dtype"]), torch.Size(tensor_metadata["shape"])
