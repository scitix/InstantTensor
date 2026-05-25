#!/usr/bin/env python
import os
import subprocess
from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext

# setuptools requires paths relative to setup.py directory, /-separated (no absolute paths)

root_path = os.path.dirname(os.path.abspath(__file__))
libaio_dir = f"{root_path}/csrc/third_party/libaio"
libaio_src = f"{libaio_dir}/src"
libaio_static = f"{libaio_src}/libaio.a"
libaio_static_version_script = "csrc/libaio_static.map" # using custom version script avoids exposing libaio symbols
libaio_static_version_script_path = f"{root_path}/{libaio_static_version_script}"
liburing_dir = f"{root_path}/csrc/third_party/liburing"
liburing_src = f"{liburing_dir}/src"
liburing_static = f"{liburing_src}/liburing.a"
package_name = "instanttensor"

include_dirs = [
    f"{root_path}/csrc",
    f"{root_path}/csrc/third_party/dlpack/include",
    f"{root_path}/csrc/third_party/pybind11/include",
    libaio_src,  # for <libaio.h>
    f"{liburing_src}/include",  # for <liburing.h>
]

boost_libs_dir = f"{root_path}/csrc/third_party/boost/libs"
# boost_include_dirs = [f"{boost_libs_dir}/{dir}/include" for dir in os.listdir(boost_libs_dir) if os.path.isdir(f"{boost_libs_dir}/{dir}") and not dir.startswith("old")]

boost_include_dirs = [f"{boost_libs_dir}/{dir}/include" for dir in os.listdir(boost_libs_dir) if os.path.isdir(f"{boost_libs_dir}/{dir}")]

include_dirs += boost_include_dirs

class BuildExt(build_ext):
    def run(self):
        try:
            libaio_cflags = os.environ.get("LIBAIO_CFLAGS", "-g -fomit-frame-pointer -O2")
            libaio_env = {
                **os.environ,
                "ENABLE_SHARED": "0",
                "CFLAGS": f"{libaio_cflags} -fvisibility=hidden",
            }
            libaio_make_cmd = [
                "make",
                "--silent",
                "-B",
                "-C",
                libaio_dir,
            ]
            print(" ".join(libaio_make_cmd))
            subprocess.run(libaio_make_cmd, env=libaio_env, check=True)
            assert os.path.isfile(libaio_static), f"{libaio_static} not found after make"

            liburing_cflags = os.environ.get("LIBURING_CFLAGS", "-O2 -fPIC")
            liburing_env = {
                **os.environ,
                "ENABLE_SHARED": "0",
                "CFLAGS": f"{liburing_cflags} -fvisibility=hidden",
            }
            liburing_configure_cmd = ["./configure", "--use-libc"]
            print(" ".join(liburing_configure_cmd))
            subprocess.run(liburing_configure_cmd, cwd=liburing_dir, env=liburing_env, check=True)
            liburing_make_cmd = [
                "make",
                "--silent",
                "-B",
                "-C",
                liburing_dir,
                "library", # skip test and examples
            ]
            print(" ".join(liburing_make_cmd))
            subprocess.run(liburing_make_cmd, env=liburing_env, check=True)
            assert os.path.isfile(liburing_static), f"{liburing_static} not found after make"

            super().run()
        finally:
            libaio_clean_cmd = ["make", "--silent", "-C", libaio_dir, "clean"]
            print(" ".join(libaio_clean_cmd))
            subprocess.run(libaio_clean_cmd, check=True)
            liburing_clean_cmd = ["make", "--silent", "-C", liburing_dir, "clean"]
            print(" ".join(liburing_clean_cmd))
            subprocess.run(liburing_clean_cmd, check=True)
            liburing_config_log = os.path.join(liburing_dir, "config.log")
            if os.path.exists(liburing_config_log):
                os.unlink(liburing_config_log)


def get_ext_modules():
    debug = os.environ.get("DEBUG", "0") == "1"
    cxx_flags = ["-std=c++17", "-fvisibility=hidden", "-fvisibility-inlines-hidden"]
    cxx_flags += ["-O0", "-g"] if debug else []

    return [
        Extension(
            name=f"{package_name}._C",
            sources=[
                "csrc/main.cpp",
                "csrc/loader_common.cpp",
                "csrc/loader_io_cufile.cpp",
                "csrc/loader_io_aio.cpp",
                "csrc/loader_io_inmem.cpp",
                "csrc/loader_io_uring.cpp",
            ],
            include_dirs=include_dirs,
            libraries=["dl"],
            extra_objects=[libaio_static, liburing_static],
            depends=[libaio_static_version_script],
            extra_compile_args=cxx_flags,
            extra_link_args=[
                f"-Wl,--version-script={libaio_static_version_script_path}",
            ],
        )
    ]


setup(
    ext_modules=get_ext_modules(),
    cmdclass={"build_ext": BuildExt},
)
