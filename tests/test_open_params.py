from datetime import timedelta
import os
from pathlib import Path
import subprocess
import sys
import unittest
from unittest import mock

import torch
import torch.distributed as dist

import instanttensor._internal.loader as impl


def make_loader(process_group, device):
    loader = impl.safe_open.__new__(impl.safe_open)
    loader.process_group = process_group
    loader.rank = dist.get_rank(process_group) if process_group is not None else 0
    loader.world_size = dist.get_world_size(process_group) if process_group is not None else 1
    loader.device = device
    loader.device_idx = device.index
    loader.filename = ['unused.safetensors']
    loader.loader_handle = None
    loader.buffer_size = 1 << 26
    loader.chunk_size = 1 << 23
    loader.concurrency = 0
    loader.io_depth = 2
    loader.backend = impl.Backend.URING
    loader.tensor_offsets = [(0, 128), (0, (1 << 32) + 128), (1, 128), (1, 256)]
    loader.tensor_keep = [True, False]
    return loader


class OpenParamsTest(unittest.TestCase):
    def test_local_open_skips_collective(self):
        loader = make_loader(None, torch.device('cpu'))
        with mock.patch.object(dist, 'all_gather_into_tensor') as gather, \
             mock.patch.object(impl._C, 'open', return_value=123) as native_open:
            loader._check_open_params()
            loader._open()
        gather.assert_not_called()
        native_open.assert_called_once()
        self.assertEqual(loader.loader_handle, 123)

    def test_constructor_rejects_before_native_open_even_when_deferred(self):
        metadata = {'weight': {'dtype': 'F32', 'shape': [1], 'data_offsets': [0, 4]}}
        for load_now in (False, True):
            with self.subTest(load_now=load_now), \
                 mock.patch.object(impl.safe_open, '_determine_io_params'), \
                 mock.patch.object(impl.safe_open, '_read_metadata', return_value=[(None, metadata, 128)]), \
                 mock.patch.object(impl.safe_open, '_finalize_buffer_size'), \
                 mock.patch.object(impl.safe_open, '_check_open_params', side_effect=ValueError('mismatch')) as check, \
                 mock.patch.object(impl._C, 'open') as native_open:
                with self.assertRaisesRegex(ValueError, 'mismatch'):
                    impl.safe_open('unused.safetensors', 'pt', 'cuda:0', load_now=load_now)
                check.assert_called_once()
                native_open.assert_not_called()

    def test_distributed_cpu_validation(self):
        self.run_distributed('gloo')

    @unittest.skipUnless(torch.cuda.device_count() >= 2, 'two GPUs required')
    def test_distributed_cuda_validation(self):
        self.run_distributed('nccl')

    def run_distributed(self, backend):
        root = Path(__file__).resolve().parents[1]
        result = subprocess.run([
            sys.executable, '-m', 'torch.distributed.run', '--standalone', '--nproc-per-node=2',
            __file__, '--distributed', backend,
        ], cwd=root, env=dict(os.environ, PYTHONPATH=str(root)),
            capture_output=True, text=True, timeout=90)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


def exercise_distributed(backend):
    rank = int(os.environ['LOCAL_RANK'])
    device = torch.device('cuda', rank) if backend == 'nccl' else torch.device('cpu')
    if backend == 'nccl':
        torch.cuda.set_device(device)
    dist.init_process_group(backend, timeout=timedelta(seconds=20))
    try:
        loader = make_loader(dist.group.WORLD, device)
        loader._check_open_params()
        loader.tensor_offsets, loader.tensor_keep = [], []
        loader._check_open_params()

        changes = [
            ('buffer_size', 1 << 27), ('chunk_size', 1 << 22),
            ('concurrency', 1), ('io_depth', 3), ('backend', impl.Backend.AIO),
            # Only the upper 32 bits differ, so truncating offsets would miss this.
            ('tensor_offsets', [(0, 128), (0, 128), (1, 128), (1, 256)]),
            ('tensor_offsets', [(0, 128)]),
            ('tensor_keep', [False, True]), ('tensor_keep', [True]),
            ('page_size', os.sysconf('SC_PAGE_SIZE') * 2),
        ]
        for field, value in changes:
            loader = make_loader(dist.group.WORLD, device)
            page_size = os.sysconf('SC_PAGE_SIZE')
            if rank == 1:
                if field == 'page_size':
                    page_size = value
                else:
                    setattr(loader, field, value)
            with mock.patch.object(impl.os, 'sysconf', return_value=page_size), \
                 mock.patch.object(impl._C, 'open') as native_open:
                try:
                    loader._check_open_params()
                except ValueError as error:
                    assert str(error) == 'Loader open parameters differ across ranks: ' + field, error
                else:
                    raise AssertionError(f'Rank {rank} did not reject {field}')
                native_open.assert_not_called()
                assert loader.loader_handle is None

        # A single-rank subgroup must skip the check even when its global rank is 1.
        subgroup = dist.new_group([1], backend=backend)
        if rank == 1:
            loader = make_loader(subgroup, device)
            with mock.patch.object(dist, 'all_gather_into_tensor') as gather:
                loader._check_open_params()
            gather.assert_not_called()
    finally:
        dist.destroy_process_group()


if __name__ == '__main__':
    if '--distributed' in sys.argv:
        exercise_distributed(sys.argv[-1])
    else:
        unittest.main()
