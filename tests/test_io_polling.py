import os
import pathlib
import shutil
import subprocess
import sys
import sysconfig
import tempfile
import unittest
import warnings


class IOPollingTest(unittest.TestCase):
    def test_gpu_early_close(self):
        import torch
        from safetensors.torch import save_file
        from instanttensor import Backend, safe_open

        if not torch.cuda.is_available():
            self.skipTest("CUDA required")
        if "--early-close" not in sys.argv:
            run = subprocess.run([sys.executable, __file__, "--early-close"],
                                 capture_output=True, text=True, timeout=60)
            self.assertEqual(run.returncode, 0, run.stdout + run.stderr)
            return
        torch.set_num_threads(1)
        with tempfile.TemporaryDirectory() as directory:
            path = pathlib.Path(directory) / "prefetch.safetensors"
            tensors = {f"t{i:03d}": torch.full((262147,), i, dtype=torch.uint8)
                       for i in range(40)}
            save_file(tensors, str(path))
            for backend in (Backend.AIO, Backend.URING, Backend.AIO_BUFFERED,
                            Backend.URING_BUFFERED, Backend.MMAP):
                for consume_first in (False, True):
                    with self.subTest(backend=backend.name, consume_first=consume_first):
                        with warnings.catch_warnings():
                            warnings.simplefilter("ignore", UserWarning)
                            with safe_open(str(path), "pt", 0, backend=backend,
                                           chunk_size=65536, io_depth=32,
                                           buffer_size=4 << 20, copy=False) as loader:
                                if consume_first:
                                    name, tensor = next(loader.tensors())
                                    self.assertTrue(torch.equal(tensor.cpu(), tensors[name]))

    @unittest.skipUnless(shutil.which("g++"), "g++ required")
    def test_production_io_state_machine(self):
        root = pathlib.Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as directory:
            executable = pathlib.Path(directory) / "test_io_polling"
            uring = pathlib.Path(directory) / "liburing"
            shutil.copytree(root / "csrc/third_party/liburing", uring,
                            ignore=shutil.ignore_patterns(".git"))
            configure = subprocess.run(
                ["./configure", "--use-libc"], cwd=uring,
                capture_output=True, text=True, timeout=120,
            )
            self.assertEqual(configure.returncode, 0, configure.stdout + configure.stderr)
            includes = [
                root / "csrc", root / "csrc/third_party/atomic_queue/include",
                root / "csrc/third_party/pybind11/include",
                root / "csrc/third_party/libaio/src", uring / "src/include",
                pathlib.Path(sysconfig.get_path("include")),
            ]
            library = pathlib.Path(sysconfig.get_config_var("LIBDIR"))
            python_library = library / sysconfig.get_config_var("LDLIBRARY")
            if not python_library.exists():
                python_library = library / f"libpython{sysconfig.get_config_var('LDVERSION')}.so"
            sanitize = os.environ.get("IO_TEST_SANITIZE")
            build = subprocess.run([
                "g++", "-std=c++17", "-pthread", "-ffunction-sections",
                "-fdata-sections", "-Wl,--gc-sections", "-g",
                *([f"-fsanitize={sanitize}", "-fno-omit-frame-pointer"] if sanitize else []),
                *(f"-I{path}" for path in includes),
                str(root / "tests/cpp/test_io_polling.cpp"), str(python_library),
                f"-Wl,-rpath,{library}", "-ldl", "-o", str(executable),
            ], capture_output=True, text=True, timeout=120)
            self.assertEqual(build.returncode, 0, build.stdout + build.stderr)
            subprocess.run([str(executable)], check=True, timeout=30)


if __name__ == "__main__" and "--early-close" in sys.argv:
    IOPollingTest("test_gpu_early_close").test_gpu_early_close()
