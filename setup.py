#!/usr/bin/env python
import os
from setuptools import setup
import torch
from torch.utils.cpp_extension import BuildExtension, CUDAExtension # CppExtension

# optionally read these from env or conda if you need custom CUDA
# but BuildExtension will auto-discover them in most cases.
# CUDA_HOME = os.environ.get('CUDA_HOME', None)

conda_prefix = os.environ.get('CONDA_PREFIX', None) or os.sys.prefix
conda_lib    = os.path.join(conda_prefix, 'lib')

if os.environ.get('DEBUG', '0') == '1':
    optimization_args = ['-g', '-O0']
else:
    optimization_args = []

setup(
    name='instanttensor',
    version='0.1',
    ext_modules=[
        CUDAExtension(
            name='instanttensor._C',
            sources=['csrc/main.cpp',],
            include_dirs=['csrc/',],
            libraries=['aio'], # 'uring', 'numa'
            library_dirs=[conda_lib], # prefer to use conda lib instead of global lib at compile time
            runtime_library_dirs=[conda_lib], # prefer to use conda lib instead of global lib at runtime
            # you can override arch or add flags here:
            extra_compile_args=['-DUSE_C10D_NCCL'] + optimization_args,
            # link cuda runtime explicitly (BuildExtension will handle this too)
        )
    ],
    cmdclass={
        'build_ext': BuildExtension
    },
)
