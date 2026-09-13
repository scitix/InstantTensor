import os
import unittest
import warnings
from contextlib import ExitStack
from unittest import mock

import torch

import instanttensor._impl as impl


IO_ENV_VARS = [
    "INSTANTTENSOR_BACKEND",
    "INSTANTTENSOR_CHUNK_SIZE",
    "INSTANTTENSOR_CONCURRENCY",
    "INSTANTTENSOR_IO_DEPTH",
    "INSTANTTENSOR_MAX_FREE_MEM_USAGE",
    "INSTANTTENSOR_BUFFER_SIZE",
]


class IOParamsTest(unittest.TestCase):
    def setUp(self):
        self.saved_env = {name: os.environ.pop(name, None) for name in IO_ENV_VARS}

    def tearDown(self):
        for name, value in self.saved_env.items():
            os.environ.pop(name, None)
            if value is not None:
                os.environ[name] = value

    def determine_io_params(
        self,
        *,
        selected_backend,
        in_memory,
        world_size=1,
        chunk_size=None,
        concurrency=None,
        io_depth=None,
        buffer_size=None,
        free_bytes=1 << 50,
    ):
        loader = impl.safe_open.__new__(impl.safe_open)
        loader.filename = ["model.safetensors"]
        loader.world_size = world_size
        loader.process_group = None
        loader.device = torch.device("cuda:0")

        with ExitStack() as stack:
            stack.enter_context(mock.patch.object(impl, "file_in_memory", return_value=in_memory))
            stack.enter_context(mock.patch.object(impl, "select_backend", return_value=selected_backend))
            stack.enter_context(mock.patch.object(impl.os, "cpu_count", return_value=64))
            stack.enter_context(mock.patch.object(
                impl.torch.cuda,
                "mem_get_info",
                return_value=(free_bytes, free_bytes),
            ))
            config = impl._resolve_open_config(
                buffer_size=buffer_size,
                chunk_size=chunk_size,
                concurrency=concurrency,
                io_depth=io_depth,
                max_free_mem_usage=1.0,
                backend=selected_backend,
            )
            loader._determine_io_params(config)
        return loader

    def test_native_configuration_is_the_single_source_of_truth(self):
        self.assertEqual(
            {backend.name: backend.value for backend in impl.Backend},
            impl._C.backend_values(),
        )
        self.assertEqual(
            impl.MAX_IO_DEPTH,
            impl._C.MAX_IO_DEPTH,
        )

        page_size = os.sysconf("SC_PAGE_SIZE")
        self.assertEqual(
            impl.required_buffer_size_for_io(page_size + 1, 3, 2),
            2 * page_size * 3 * 2,
        )

    def test_mmap_default_depth_includes_worker_concurrency(self):
        loader = self.determine_io_params(
            selected_backend=impl.Backend.MMAP,
            in_memory=True,
        )

        self.assertEqual(loader.chunk_size, 2 * 1024 * 1024)
        self.assertEqual(loader.concurrency, 32)
        self.assertEqual(loader.io_depth, 3 * loader.concurrency)

    def test_cufile_default_depth_includes_worker_concurrency(self):
        loader = self.determine_io_params(
            selected_backend=impl.Backend.CUFILE,
            in_memory=False,
            world_size=2,
        )

        self.assertEqual(loader.chunk_size, 8 * 1024 * 1024)
        self.assertEqual(loader.concurrency, 16)
        self.assertEqual(loader.io_depth, 2 * loader.concurrency)

    def test_native_async_depth_does_not_depend_on_concurrency(self):
        with self.assertWarnsRegex(RuntimeWarning, "does not support concurrency"):
            loader = self.determine_io_params(
                selected_backend=impl.Backend.URING,
                in_memory=False,
                world_size=2,
                concurrency=37,
            )

        self.assertEqual(loader.concurrency, 0)
        self.assertEqual(loader.io_depth, 256)

    def test_native_async_default_concurrency_is_unused(self):
        loader = self.determine_io_params(
            selected_backend=impl.Backend.URING,
            in_memory=False,
        )

        self.assertEqual(loader.concurrency, 0)
        self.assertEqual(loader.io_depth, 512)

    def test_buffered_native_async_default_concurrency_is_unused(self):
        loader = self.determine_io_params(
            selected_backend=impl.Backend.URING_BUFFERED,
            in_memory=True,
        )

        self.assertEqual(loader.concurrency, 0)
        self.assertEqual(loader.io_depth, 32)
        loader.tensor_sizes = [1]
        loader._finalize_buffer_size(None)
        self.assertEqual(
            loader.buffer_size,
            loader.chunk_size * loader.io_depth * loader.world_size,
        )

    def test_in_memory_native_async_depth_does_not_depend_on_concurrency(self):
        with self.assertWarnsRegex(RuntimeWarning, "does not support concurrency"):
            loader = self.determine_io_params(
                selected_backend=impl.Backend.URING_BUFFERED,
                in_memory=True,
                concurrency=37,
            )

        self.assertEqual(loader.concurrency, 0)
        self.assertEqual(loader.io_depth, 32)

    def test_memory_limit_shrinks_depth_not_worker_concurrency(self):
        chunk_size = 8 * 1024 * 1024
        with self.assertWarnsRegex(RuntimeWarning, "Shrink io_depth"):
            loader = self.determine_io_params(
                selected_backend=impl.Backend.MMAP,
                in_memory=True,
                chunk_size=chunk_size,
                concurrency=4,
                io_depth=10,
                free_bytes=chunk_size * 5,
            )

        self.assertEqual(loader.concurrency, 4)
        self.assertEqual(loader.io_depth, 5)

    def test_worker_backends_reject_zero_concurrency_before_io_depth(self):
        for backend in (impl.Backend.MMAP, impl.Backend.CUFILE):
            for io_depth in (None, 0, 1):
                with self.subTest(backend=backend.name, io_depth=io_depth):
                    with self.assertRaisesRegex(ValueError, "concurrency must be greater than zero"):
                        self.determine_io_params(
                            selected_backend=backend, in_memory=False,
                            concurrency=0, io_depth=io_depth,
                        )

    def test_all_backends_reject_negative_concurrency_before_io_depth(self):
        for backend in impl.Backend:
            with self.subTest(backend=backend.name):
                with self.assertRaisesRegex(ValueError, "concurrency must not be negative"):
                    self.determine_io_params(
                        selected_backend=backend, in_memory=False,
                        concurrency=-1, io_depth=0,
                    )

    def test_worker_backends_preserve_positive_concurrency(self):
        for backend, depth_factor in ((impl.Backend.MMAP, 3), (impl.Backend.CUFILE, 2)):
            with self.subTest(backend=backend.name):
                with warnings.catch_warnings(record=True) as emitted:
                    warnings.simplefilter("always")
                    loader = self.determine_io_params(
                        selected_backend=backend, in_memory=False, concurrency=5,
                    )
                self.assertEqual(loader.concurrency, 5)
                self.assertEqual(loader.io_depth, depth_factor * 5)
                self.assertEqual(emitted, [])

    def test_async_backends_warn_and_override_positive_concurrency(self):
        for backend in (impl.Backend.AIO, impl.Backend.URING,
                        impl.Backend.AIO_BUFFERED, impl.Backend.URING_BUFFERED):
            with self.subTest(backend=backend.name):
                with self.assertWarnsRegex(RuntimeWarning, "concurrency=5 to 0"):
                    loader = self.determine_io_params(
                        selected_backend=backend, in_memory=False,
                        concurrency=5, io_depth=7,
                    )
                self.assertEqual(loader.concurrency, 0)
                self.assertEqual(loader.io_depth, 7)

    def test_async_backends_accept_zero_and_default_without_warning(self):
        for backend in (impl.Backend.AIO, impl.Backend.URING,
                        impl.Backend.AIO_BUFFERED, impl.Backend.URING_BUFFERED):
            for concurrency in (None, 0):
                with self.subTest(backend=backend.name, concurrency=concurrency):
                    with warnings.catch_warnings(record=True) as emitted:
                        warnings.simplefilter("always")
                        loader = self.determine_io_params(
                            selected_backend=backend, in_memory=False,
                            concurrency=concurrency,
                        )
                    self.assertEqual(loader.concurrency, 0)
                    self.assertEqual(emitted, [])

    def test_concurrency_environment_uses_same_validation(self):
        os.environ["INSTANTTENSOR_CONCURRENCY"] = "0"
        for backend in (impl.Backend.MMAP, impl.Backend.CUFILE):
            with self.subTest(backend=backend.name):
                with self.assertRaisesRegex(ValueError, "concurrency must be greater than zero"):
                    self.determine_io_params(selected_backend=backend, in_memory=False)
        os.environ["INSTANTTENSOR_CONCURRENCY"] = "5"
        with self.assertWarnsRegex(RuntimeWarning, "concurrency=5 to 0"):
            loader = self.determine_io_params(selected_backend=impl.Backend.AIO, in_memory=False)
        self.assertEqual(loader.concurrency, 0)

    def test_io_depth_cannot_exceed_executor_capacity(self):
        with self.assertRaisesRegex(ValueError, "io_depth must not exceed"):
            self.determine_io_params(
                selected_backend=impl.Backend.URING,
                in_memory=False,
                io_depth=impl.MAX_IO_DEPTH + 1,
            )

    def test_explicit_buffer_and_depth_must_be_compatible(self):
        chunk_size = 8 * 1024 * 1024
        with self.assertRaisesRegex(ValueError, "too small for io_depth=4"):
            self.determine_io_params(
                selected_backend=impl.Backend.URING,
                in_memory=False,
                chunk_size=chunk_size,
                io_depth=4,
                buffer_size=3 * chunk_size,
            )

    def test_explicit_buffer_shrinks_default_depth(self):
        chunk_size = 8 * 1024 * 1024
        with self.assertWarnsRegex(RuntimeWarning, "to fit buffer_size"):
            loader = self.determine_io_params(
                selected_backend=impl.Backend.URING,
                in_memory=False,
                chunk_size=chunk_size,
                buffer_size=4 * chunk_size,
            )

        self.assertEqual(loader.io_depth, 4)

    def test_explicit_buffer_must_fit_one_io_operation(self):
        chunk_size = 8 * 1024 * 1024
        with self.assertRaisesRegex(ValueError, "too small for one I/O operation"):
            self.determine_io_params(
                selected_backend=impl.Backend.URING,
                in_memory=False,
                chunk_size=chunk_size,
                buffer_size=chunk_size - 1,
            )

    def test_buffer_limit_uses_page_aligned_chunk_size(self):
        page_size = os.sysconf("SC_PAGE_SIZE")
        chunk_size = page_size + 1
        with self.assertRaisesRegex(ValueError, "too small for io_depth=2"):
            self.determine_io_params(
                selected_backend=impl.Backend.URING,
                in_memory=False,
                chunk_size=chunk_size,
                io_depth=2,
                buffer_size=3 * page_size,
            )

    def test_memory_limit_uses_page_aligned_chunk_size(self):
        page_size = os.sysconf("SC_PAGE_SIZE")
        chunk_size = page_size + 1
        with self.assertWarnsRegex(RuntimeWarning, "due to memory limit"):
            loader = self.determine_io_params(
                selected_backend=impl.Backend.URING,
                in_memory=False,
                chunk_size=chunk_size,
                io_depth=2,
                free_bytes=3 * page_size,
            )

        self.assertEqual(loader.io_depth, 1)

    def test_memory_limit_must_fit_one_io_operation(self):
        chunk_size = 8 * 1024 * 1024
        with self.assertRaisesRegex(
            RuntimeError, "too small for one I/O operation",
        ):
            self.determine_io_params(
                selected_backend=impl.Backend.URING,
                in_memory=False,
                chunk_size=chunk_size,
                free_bytes=chunk_size - 1,
            )

    def test_environment_buffer_and_depth_are_explicit(self):
        chunk_size = 8 * 1024 * 1024
        os.environ["INSTANTTENSOR_BUFFER_SIZE"] = str(3 * chunk_size)
        os.environ["INSTANTTENSOR_IO_DEPTH"] = "4"

        with self.assertRaisesRegex(ValueError, "too small for io_depth=4"):
            self.determine_io_params(
                selected_backend=impl.Backend.URING,
                in_memory=False,
                chunk_size=chunk_size,
            )

    def test_explicit_buffer_is_not_shrunk_below_io_requirement(self):
        chunk_size = 8 * 1024 * 1024
        loader = self.determine_io_params(
            selected_backend=impl.Backend.URING,
            in_memory=False,
            chunk_size=chunk_size,
            io_depth=3,
            buffer_size=4 * chunk_size,
        )
        loader.tensor_sizes = [1]
        loader.total_tensor_size = 1

        with self.assertWarnsRegex(RuntimeWarning, "Shrink buffer size"):
            loader._finalize_buffer_size(4 * chunk_size)

        self.assertEqual(loader.buffer_size, 3 * chunk_size)

    def test_final_buffer_must_fit_device_memory_budget(self):
        chunk_size = 8 * 1024 * 1024
        loader = self.determine_io_params(
            selected_backend=impl.Backend.URING,
            in_memory=False,
            chunk_size=chunk_size,
            io_depth=1,
            free_bytes=2 * chunk_size,
        )
        loader.tensor_sizes = [3 * chunk_size]
        loader.total_tensor_size = 3 * chunk_size

        with self.assertRaisesRegex(
            RuntimeError, "exceeds device memory budget",
        ):
            loader._finalize_buffer_size(None)

    def test_buffer_environment_is_resolved_once(self):
        chunk_size = 8 * 1024 * 1024
        configured_buffer_size = 4 * chunk_size
        loader = impl.safe_open.__new__(impl.safe_open)
        loader.filename = ["model.safetensors"]
        loader.world_size = 1
        loader.process_group = None
        loader.device = torch.device("cuda:0")
        loader.tensor_sizes = [1]
        loader.total_tensor_size = 1

        with ExitStack() as stack:
            env_buffer_size = stack.enter_context(mock.patch.object(
                impl, "env_buffer_size", return_value=configured_buffer_size,
            ))
            stack.enter_context(mock.patch.object(impl, "file_in_memory", return_value=False))
            stack.enter_context(mock.patch.object(
                impl, "select_backend", return_value=impl.Backend.URING,
            ))
            stack.enter_context(mock.patch.object(
                impl.torch.cuda,
                "mem_get_info",
                return_value=(1 << 50, 1 << 50),
            ))

            config = impl._resolve_open_config(
                buffer_size=None,
                chunk_size=chunk_size,
                concurrency=0,
                io_depth=4,
                max_free_mem_usage=1.0,
                backend=impl.Backend.URING,
            )
            loader._determine_io_params(config)
            loader._finalize_buffer_size(config.buffer_size)

        env_buffer_size.assert_called_once_with()

    def test_uring_requires_linux_5_6(self):
        backend_status = mock.Mock(return_value=(
            False,
            "io_uring requires Linux kernel 5.6 or newer; the detected kernel "
            "version is 5.5.0.",
            "",
        ))
        with mock.patch.object(
            impl._C, "backend_status", backend_status,
        ):
            with self.assertRaises(RuntimeError) as raised:
                impl.select_backend([impl.Backend.URING])

        self.assertEqual(
            str(raised.exception),
            "No available backend was found among candidates [URING]. "
            "io_uring requires Linux kernel 5.6 or newer; the detected kernel "
            "version is 5.5.0.",
        )
        backend_status.assert_called_once_with(impl.Backend.URING.value)

    def test_uring_warns_below_recommended_kernel(self):
        with ExitStack() as stack:
            stack.enter_context(mock.patch.object(
                impl._C,
                "backend_status",
                return_value=(
                    True,
                    "",
                    "io_uring on Linux 5.14 may be unstable; "
                    "Linux 5.15 or newer is recommended.",
                ),
            ))
            stack.enter_context(mock.patch.object(
                impl, "_emitted_backend_warnings", set(),
            ))

            with self.assertWarnsRegex(RuntimeWarning, "Linux 5.15 or newer"):
                backend = impl.select_backend([impl.Backend.URING])

        self.assertEqual(backend, impl.Backend.URING)

    def test_backend_warning_is_emitted_once(self):
        warning = (
            "io_uring on Linux 5.14 may be unstable; "
            "Linux 5.15 or newer is recommended."
        )
        with ExitStack() as stack:
            stack.enter_context(mock.patch.object(
                impl._C,
                "backend_status",
                return_value=(True, "", warning),
            ))
            stack.enter_context(mock.patch.object(
                impl, "_emitted_backend_warnings", set(),
            ))
            warn = stack.enter_context(mock.patch.object(impl.warnings, "warn"))

            impl.select_backend([impl.Backend.URING])
            impl.select_backend([impl.Backend.URING_BUFFERED])

        warn.assert_called_once()

    def test_uring_does_not_warn_on_recommended_kernel(self):
        with ExitStack() as stack:
            stack.enter_context(mock.patch.object(
                impl._C,
                "backend_status",
                return_value=(True, "", ""),
            ))
            warn = stack.enter_context(mock.patch.object(impl.warnings, "warn"))

            backend = impl.select_backend([impl.Backend.URING_BUFFERED])

        self.assertEqual(backend, impl.Backend.URING_BUFFERED)
        warn.assert_not_called()


if __name__ == "__main__":
    unittest.main()
