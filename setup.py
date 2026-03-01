#!/usr/bin/env python
import os
from setuptools import setup

# setuptools requires paths relative to setup.py directory, /-separated (no absolute paths)
runtime_library_dirs = [
    "$ORIGIN/../nvidia/cuda_runtime/lib",
    "$ORIGIN/../nvidia/nccl/lib",
    "$ORIGIN/../nvidia/cufile/lib",
]

root_path = os.path.dirname(os.path.abspath(__file__))

include_dirs = [
    f"{root_path}/csrc",
    f"{root_path}/csrc/third_party/boost/include",
    f"{root_path}/csrc/third_party/dlpack/include",
]

def get_ext_modules():
    try:
        from torch.utils.cpp_extension import CUDAExtension
    except Exception as e:
        raise RuntimeError(
            "Building instanttensor requires PyTorch installed in the current environment.\n"
            "Please run:\n"
            "  pip install torch\n"
            "  pip install --no-build-isolation .\n"
        ) from e

    debug = os.environ.get("DEBUG", "0") == "1"
    cxx_flags = ["-std=c++17", "-DUSE_C10D_NCCL"]
    cxx_flags += ["-O0", "-g"] if debug else [] # use defualt optimization level

    ext = CUDAExtension( # CUDA related headers and libraries are automatically provided
        name="instanttensor._C",
        sources=["csrc/main.cpp"], # always relative to setup.py
        include_dirs=include_dirs, # should be absolute paths
        library_dirs=[],
        libraries=["aio", "cudart", "nccl", "cufile"],  # libaio from system
        extra_compile_args={"cxx": cxx_flags, "nvcc": []},
        # extra_link_args=extra_link_args,
        runtime_library_dirs=runtime_library_dirs,  # same rpath policy as PyTorch
    )
    return [ext]

def get_cmdclass():
    from torch.utils.cpp_extension import BuildExtension
    return {"build_ext": BuildExtension}

setup(
    ext_modules=get_ext_modules(),
    cmdclass=get_cmdclass(),
)
