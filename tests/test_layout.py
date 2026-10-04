import os
from pathlib import Path
import shutil
import subprocess
import sys
import sysconfig
import tempfile
import unittest


class LayoutTest(unittest.TestCase):
    @unittest.skipUnless(shutil.which("g++"), "g++ required")
    def test_native_layout_invariants(self):
        root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as directory:
            executable = Path(directory) / "test_layout"
            uring = Path(directory) / "liburing"
            shutil.copytree(root / "csrc/third_party/liburing", uring,
                            ignore=shutil.ignore_patterns(".git"))
            configure = subprocess.run(["./configure", "--use-libc"], cwd=uring,
                                       capture_output=True, text=True, timeout=120)
            self.assertEqual(configure.returncode, 0, configure.stdout + configure.stderr)
            includes = [root / "csrc", root / "csrc/third_party/atomic_queue/include",
                        root / "csrc/third_party/pybind11/include", root / "csrc/third_party/libaio/src",
                        uring / "src/include", Path(sysconfig.get_path("include"))]
            library = Path(sysconfig.get_config_var("LIBDIR"))
            python_library = library / sysconfig.get_config_var("LDLIBRARY")
            if not python_library.exists():
                python_library = library / f"libpython{sysconfig.get_config_var('LDVERSION')}.so"
            build = subprocess.run([
                "g++", "-std=c++17", "-O2", "-pthread", "-ffunction-sections", "-fdata-sections",
                "-Wl,--gc-sections", *(f"-I{path}" for path in includes),
                str(root / "tests/cpp/test_layout.cpp"), str(python_library),
                f"-Wl,-rpath,{library}", "-ldl", "-o", str(executable),
            ], capture_output=True, text=True, timeout=120)
            self.assertEqual(build.returncode, 0, build.stdout + build.stderr)
            run = subprocess.run([str(executable)], capture_output=True, text=True, timeout=30)
            self.assertEqual(run.returncode, 0, run.stdout + run.stderr)

    def test_gpu_layout_cases(self):
        import torch
        import instanttensor._C as native
        from instanttensor import Backend
        if not torch.cuda.is_available():
            self.skipTest("CUDA required")
        for backend in (Backend.AIO, Backend.URING, Backend.MMAP, Backend.AIO_BUFFERED, Backend.URING_BUFFERED):
            if not native.backend_status(backend.value)[0]:
                continue
            with self.subTest(backend=backend.name):
                self.run_gpu_case(backend.name, 1)

    def test_two_gpu_layout_cases(self):
        import torch
        if torch.cuda.device_count() < 2:
            self.skipTest("two GPUs required")
        for backend in ("AIO", "URING"):
            with self.subTest(backend=backend):
                self.run_gpu_case(backend, 2)

    def run_gpu_case(self, backend, ranks):
        root = Path(__file__).resolve().parents[1]
        env = {key: value for key, value in os.environ.items() if not key.startswith("INSTANTTENSOR_")}
        env["PYTHONPATH"] = str(root)
        command = [sys.executable]
        if ranks > 1:
            command += ["-m", "torch.distributed.run", "--standalone", f"--nproc-per-node={ranks}"]
        command += [str(root / "tests/test_layout.py"), "--gpu", backend, str(ranks)]
        run = subprocess.run(command, cwd=root, env=env, capture_output=True, text=True, timeout=90)
        self.assertEqual(run.returncode, 0, run.stdout + run.stderr)
        self.assertEqual(run.stdout.count("layout GPU cases passed"), ranks)


def exercise_gpu(backend, ranks):
    import json
    import struct
    import warnings
    import torch
    import torch.distributed as dist
    from instanttensor import Backend, safe_open

    rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(rank)
    if ranks > 1:
        dist.init_process_group("nccl")
    cases = [
        [(4096, [49152]), (5120, [65536])],
        [(4096, [0, 4096 * ranks, 0, 4096 * ranks, 0])],
        [(4096, [0]), (5120, [0, 0])],
        [(5120, [1, 0, 5118, 40961, 0])],
    ]
    try:
        with tempfile.TemporaryDirectory() as directory:
            for case_id, files in enumerate(cases):
                paths, expected = [], {}
                for file_id, (start, sizes) in enumerate(files):
                    metadata, payload, offset = {}, bytearray(), 0
                    for size in sizes:
                        name = f"t{len(expected):03d}"
                        value = len(expected) % 127
                        metadata[name] = {"dtype": "I8" if size else "F32", "shape": [size], "data_offsets": [offset, offset + size]}
                        expected[name] = torch.full((size,), value, dtype=torch.int8 if size else torch.float32)
                        payload.extend(bytes([value]) * size)
                        offset += size
                    header = json.dumps(metadata).encode()
                    assert len(header) <= start - 8
                    header += b" " * (start - 8 - len(header))
                    path = Path(directory) / f"case{case_id}-{file_id}.safetensors"
                    path.write_bytes(struct.pack("<Q", len(header)) + header + payload)
                    paths.append(str(path))
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", UserWarning)
                    with safe_open(paths, "pt", rank,
                                   process_group=dist.group.WORLD if ranks > 1 else None,
                                   backend=Backend[backend], chunk_size=4096, io_depth=8,
                                   concurrency=2 if backend == "MMAP" else 0,
                                   buffer_size=65536, copy=False) as loader:
                        for name, tensor in loader.tensors():
                            torch.testing.assert_close(tensor.cpu(), expected[name], rtol=0, atol=0)
        print("layout GPU cases passed", flush=True)
    finally:
        if ranks > 1:
            dist.destroy_process_group()


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--gpu":
        exercise_gpu(sys.argv[2], int(sys.argv[3]))
    else:
        unittest.main()
