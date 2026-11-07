from contextlib import ExitStack, contextmanager
import unittest
from unittest import mock

import torch

import instanttensor._internal.loader as impl
from instanttensor._internal.resources import ResourceBudget


class LoaderTimingTest(unittest.TestCase):
    @contextmanager
    def timed_loader(self, load_now):
        self.now = 1000.0

        def timed(seconds, result=None):
            def call(*args, **kwargs):
                self.now += seconds
                return result
            return call

        metadata = {
            "first": {"dtype": "F32", "shape": [1], "data_offsets": [0, 4]},
            "second": {"dtype": "F32", "shape": [1], "data_offsets": [4, 8]},
        }
        with ExitStack() as stack:
            stack.enter_context(mock.patch.object(impl.time, "perf_counter", side_effect=lambda: self.now))
            stack.enter_context(mock.patch.object(impl, "env_debug", return_value=True))
            log = stack.enter_context(mock.patch.object(impl, "debug_log"))
            stack.enter_context(mock.patch.object(impl, "file_in_memory", return_value=False))
            stack.enter_context(mock.patch.object(impl, "negotiate_resources",
                side_effect=timed(2, ResourceBudget(8, 1 << 30, 1 << 30))))
            stack.enter_context(mock.patch.object(impl, "select_distributed_backend",
                side_effect=timed(3, impl.Backend.URING)))
            stack.enter_context(mock.patch.object(impl.safe_open, "_read_metadata",
                side_effect=timed(5, [(None, metadata, 8)])))
            stack.enter_context(mock.patch.object(impl._C, "open", side_effect=timed(7, 1)))
            stack.enter_context(mock.patch.object(impl._C, "close", side_effect=timed(11)))
            stack.enter_context(mock.patch.object(impl._C, "get_dl_tensor",
                side_effect=lambda handle, index, size: torch.zeros(size, dtype=torch.uint8)))
            stream = mock.Mock()
            stream.synchronize.side_effect = timed(13)
            stack.enter_context(mock.patch.object(impl.torch.cuda, "current_stream", return_value=stream))
            loader = impl.safe_open("unused.safetensors", "pt", "cuda:0",
                backend=impl.Backend.URING, load_now=load_now, chunk_size=4096,
                io_depth=1, buffer_size=4096)
            yield loader, log

    def logged_values(self, log, prefix):
        return next(call.args[1:] for call in log.call_args_list if call.args[0].startswith(prefix))

    def test_delayed_entry_and_exception_keep_stage_boundaries(self):
        for load_now in (True, False):
            for fail in (False, True):
                with self.subTest(load_now=load_now, fail=fail), self.timed_loader(load_now) as (loader, log):
                    self.now += 100  # Time spent by the caller before entering with.
                    try:
                        with loader:
                            self.now += 17
                            if fail:
                                raise ValueError("consumer failed")
                    except ValueError as error:
                        self.assertTrue(fail)
                        self.assertEqual(str(error), "consumer failed")
                    total, prepare, outside, opened, load, closed = self.logged_values(log, "Time:")
                    self.assertEqual((prepare, outside, opened, load, closed), (10, 100, 7, 30, 11))
                    self.assertEqual(total, prepare + outside + opened + load + closed)
                    self.assertEqual(self.logged_values(log, "Prepare time:"), (2, 3, 5))

    def test_throughput_counts_only_yielded_tensors(self):
        for count in (0, 1, 2):
            with self.subTest(count=count), self.timed_loader(True) as (loader, log):
                with loader:
                    tensors = loader.tensors()
                    for _ in range(count):
                        next(tensors)
                total_seconds = self.logged_values(log, "Time:")[0]
                load_seconds = self.logged_values(log, "Time:")[4]
                throughput, load_throughput, yielded_bytes = self.logged_values(log, "Throughput:")
                self.assertEqual(yielded_bytes, count * 4)
                self.assertEqual(throughput, count * 4e-9 / total_seconds)
                self.assertEqual(load_throughput, count * 4e-9 / load_seconds)


if __name__ == "__main__":
    unittest.main()
