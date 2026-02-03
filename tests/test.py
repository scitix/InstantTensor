import os
import sys
from typing import Any
import torch
import torch.distributed as dist
import time
from tqdm import tqdm
from functools import partial
import argparse
import numpy as np
import ctypes
# import zlib
import instantsafetensors
from instantsafetensors import safe_open as instant_safe_open

from safetensors import safe_open as safetensors_safe_open

# import pdb
# pdb.set_trace()

def parse_args():
    parser = argparse.ArgumentParser(
        description='Example: parse backend, files list, and optional tp/pp integer parameters.'
    )
    # first positional argument: backend
    parser.add_argument(
        'backend',
        type=str,
        choices=['instantsafetensors', 'safetensors'],
        help='Backend type (choices: instantsafetensors, safetensors)'
    )
    # all remaining positional arguments go into files list
    parser.add_argument(
        'files',
        nargs='*',            # zero or more filenames; use '+' if you require at least one
        type=str,
        help='List of files to process'
    )
    # optional integer argument --tp
    parser.add_argument(
        '--tp',
        type=int,
        default=1,
        help='Optional tp value (integer). Note that world_size / (tp * pp) is the data parallelism size.'
    )
    # optional integer argument --pp
    parser.add_argument(
        '--pp',
        type=int,
        default=1,
        help='Optional pp value (integer). Note that world_size / (tp * pp) is the data parallelism size.'
    )
    parser.add_argument(
        '--load-group-size',
        type=int,
        default=None,
        help='Optional load group size value (integer). If not set, the default value is the same as the world size.'
    )
    parser.add_argument(
        '--checksum',
        action='store_true',
        help='Compute checksum of the tensors'
    )
    parser.add_argument(
        '--rank0-only',
        action='store_true',
        help='Only run safe_open() on rank0. For debug purpose. May blocks the script.'
    )
    # parser.add_argument(
    #     '--grouped',
    #     action='store_true',
    #     help='Group the files into a single list'
    # )
    parser.add_argument(
        '--local-rank',
        type=int,
        default=None,
        help='Local rank (ignored). A placeholder to keep compatible with torch.distributed.launch'
    )
    return parser.parse_args()

args = parse_args()
backend = args.backend
files = args.files
tp = args.tp
pp = args.pp
load_group_size = args.load_group_size
compute_checksum = args.checksum
rank0_only = args.rank0_only
grouped = True if backend == 'instantsafetensors' else False

if os.environ.get("NCCL_IB_GID_INDEX") != "3":
    print("Setting NCCL_IB_GID_INDEX to 3")
    os.environ["NCCL_IB_GID_INDEX"] = "3"

safe_open_dict = {
    'instantsafetensors': instant_safe_open,
    'safetensors': safetensors_safe_open,
}

device_dict = {
    'instantsafetensors': 'cuda',
    'safetensors': 'cpu', # NOTE: safetensors performs best when the intermediate tensors are on CPU
}

def collective_print(*args, **kwargs):
    if dist.is_initialized():
        for rank in range(dist.get_world_size()):
            dist.barrier()
            if rank == dist.get_rank():
                print(f"[Rank{rank}]", *args, **kwargs)
    else:
        print("[Rank0]", *args, **kwargs)

def allocate_tensors(global_rank, files):
    tp_rank = global_rank % tp
    pp_rank = global_rank // tp % pp
    dp_rank = global_rank // (tp * pp)
    tensors = {}
    f = instant_safe_open(files, framework='pt', device='cuda:0', load_now=False) # only reads metadata, device is ignored
    tensor_names = f.keys()
    num_tensors = len(tensor_names)
    pp_tensor_names = tensor_names[(num_tensors+pp-1)//pp*pp_rank : (num_tensors+pp-1)//pp*(pp_rank+1)] # the right side may overflow, but it is ok
    for key in pp_tensor_names:
        shape = f.get_shape(key)
        dtype = f.get_dtype(key)


        tp_shard_l = (shape[0]+tp-1) // tp * tp_rank
        tp_shard_r = (shape[0]+tp-1) // tp * (tp_rank+1)
        tp_shard_r = min(tp_shard_r, max(shape[0], tp_shard_l))
        shape = (tp_shard_r - tp_shard_l, *shape[1:])
        

        tensor = torch.empty(shape, dtype=dtype, device="cuda")
        tensors[key] = tensor
    return tensors, tensor_names

_PyMemView_FromMem = ctypes.pythonapi.PyMemoryView_FromMemory
_PyMemView_FromMem.argtypes = (ctypes.c_void_p, ctypes.c_ssize_t, ctypes.c_int)
_PyMemView_FromMem.restype  = ctypes.py_object

def tensor_checksum(x: torch.Tensor) -> str:
    device = x.device
    x = x.detach().contiguous().cpu()

    ptr   = x.data_ptr()
    nbyte = x.numel() * x.element_size()
    mv = _PyMemView_FromMem(ctypes.c_void_p(ptr), nbyte, 0x200)
    b = np.ndarray(buffer=mv, dtype=np.uint8, shape=(nbyte,))

    rem = b.shape[0] % 8
    if rem:
        b = np.concatenate([b, np.zeros(8-rem, dtype=np.uint8)])
    u64 = b.view(np.uint64)
    h = np.bitwise_xor.reduce(u64)
    return int(h)
    # t = torch.from_numpy(u64).to(device=device, dtype=torch.uint64)
    # s = torch.sum(t, dtype=torch.int64) # NOTE: sum of uint64 is not implemented, use int64 instead
    # return (s.item() + (1<<64)) % (1<<64)

def distributed_load():
    global load_group_size
    safe_open = safe_open_dict[backend]
    buffer_device = device_dict[backend]

    
    if os.environ.get('RANK') is None:
        global_world_size = 1
        global_rank = 0
        if load_group_size is None:
            load_group_size = 1
        load_group_id = 0
        load_group_rank = 0
        local_rank = 0
        collective_print("[TEST] Skip init_process_group() since RANK is not set")
    else:
        local_rank = int(os.environ.get('LOCAL_RANK', 0))
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend='nccl')
        # warm up nccl
        dist.all_reduce(torch.zeros(1, device='cuda'))
        global_world_size = dist.get_world_size()
        global_rank = dist.get_rank()
        if load_group_size is None:
            load_group_size = global_world_size
        assert global_world_size % load_group_size == 0
        load_group_id = global_rank // load_group_size
        load_group_rank = global_rank % load_group_size
    
    collective_print(f"[TEST] Rank {load_group_rank} in group {load_group_id} of size {load_group_size}")
    
    if buffer_device == 'cuda':
        buffer_device = f'cuda:{local_rank}'
    
    if backend == 'instantsafetensors':
        if any(not instantsafetensors.file_in_memory(file) for file in files):
            instantsafetensors.init()
        # NOTE: process_group can be the world group or a sub-group, 
        #       each group loads tensors independently. 
        #       Typically, we use the world group to maximize concurrency and avoid redundant loading.
        process_group = None
        if load_group_size > 1:
            if load_group_size == global_world_size:
                process_group = dist.GroupMember.WORLD
            else:
                for gid in range(global_world_size // load_group_size):
                    new_process_group = dist.new_group(ranks=[gid * load_group_size + i for i in range(load_group_size)])
                    if gid == load_group_id:
                        process_group = new_process_group
                dist.all_reduce(torch.zeros(1, device='cuda'), group=process_group)
        safe_open = partial(safe_open, process_group=process_group) 
    
    tp_rank = global_rank % tp
    pp_rank = global_rank // tp % pp
    dp_rank = global_rank // (tp * pp)

    tensors, full_ordered_keys = allocate_tensors(global_rank, files)
    total_size = sum(tensor.numel() * tensor.element_size() for tensor in tensors.values())
    
    pbar = tqdm(total=len(full_ordered_keys), position=local_rank, desc=f"{load_group_id}-{load_group_rank}") # add a header in format that shows which group it is

    grouped_files = [files] if grouped else files

    if rank0_only and load_group_rank != 0 and backend == 'instantsafetensors':
        for file in grouped_files:
            safe_open(file, framework='pt', device=buffer_device, open_now=False) # run allreduce inside
    
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    t_open = 0
    t_close = 0
    t_keys = 0
    t_first = 0

    if not rank0_only or load_group_rank == 0:
        for file in grouped_files:
            t_open_record = time.perf_counter()
            with safe_open(file, framework='pt', device=buffer_device) as f:
                t_open_once = time.perf_counter() - t_open_record
                # print(f"[TEST] Time taken to open: {t_open_once:.3f} seconds")
                t_open += t_open_once

                t_keys_record = time.perf_counter()
                keys = f.keys()
                t_keys += time.perf_counter() - t_keys_record

                for key_idx, key in enumerate(keys):
                    t_get_record = time.perf_counter()
                    loaded_tensor = f.get_tensor(key)
                    pbar.update(1)
                    t_get = time.perf_counter() - t_get_record
                    # if key_idx < 20:
                    #     print(f"[TEST] Time taken to get tensor {key}: {t_get:.6f} seconds")
                    if key_idx == 0:
                        t_first += t_get

                    gpu_tensor = tensors.get(key)
                    if gpu_tensor is None: # skip tensors on other pp ranks
                        continue 
                    
                    shape = loaded_tensor.shape
                    tp_shard_l = (shape[0]+tp-1) // tp * tp_rank
                    tp_shard_r = (shape[0]+tp-1) // tp * (tp_rank+1)
                    tp_shard_r = min(tp_shard_r, max(shape[0], tp_shard_l))
                    shape = (tp_shard_r - tp_shard_l, *shape[1:])
                    loaded_tensor = loaded_tensor[tp_shard_l:tp_shard_r]
                    gpu_tensor.copy_(loaded_tensor)
                t_close_record = time.perf_counter()
            t_close += time.perf_counter() - t_close_record
    
    torch.cuda.synchronize()
    t1 = time.perf_counter()
    pbar.close()

    t = t1 - t0
    size_gb = total_size / 1000**3
    size_gib = total_size / 1024**3

    collective_print(f"[TEST] Time taken to load: open: {t_open:.2f}s, keys: {t_keys:.2f}s, first tensor: {t_first:.2f}s, close: {t_close:.2f}s")
    collective_print(f"[TEST] Summary: time: {t:.2f}s, tensors: {len(tensors)}, size: {size_gb:.2f}GB/{size_gib:.2f}GiB, throughput: {size_gb / t:.2f}GBps/{size_gib / t:.2f}GiBps")

    size_tensor = torch.tensor([total_size], device='cuda')
    if global_world_size > 1:
        dist.all_reduce(size_tensor)
    t_world = time.perf_counter() - t0
    if global_rank == 0:
        print(f"[TEST] Total throughput: {size_tensor.item() / t_world / 1000**3:.2f}GBps/{size_tensor.item() / t_world / 1024**3:.2f}GiBps")

    # print(ordered_keys[0], tensors[ordered_keys[0]])
    # print(ordered_keys[-1], tensors[ordered_keys[-1]])

    if compute_checksum:
        collective_print("[TEST] Computing checksum")
        checksum = 0
        for key in tqdm(full_ordered_keys):
            if key in tensors: # some keys are not in tensors in PP mode
                checksum ^= tensor_checksum(tensors[key])
        collective_print(f"[TEST] Final checksum: {checksum:016x}")

distributed_load()


