#!/usr/bin/env python
import os
import shutil
from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext

# setuptools requires paths relative to setup.py directory, /-separated (no absolute paths)

root_path = os.path.dirname(os.path.abspath(__file__))
libaio_dir = f"{root_path}/csrc/third_party/libaio"
libaio_src = f"{libaio_dir}/src"
# libaio three names: link name (for -laio), real file (make output), soname (DT_SONAME / loader lookup)
LIBAIO_SO_LINKNAME = "libaio.so"       # -laio looks for this; we symlink it in src after make
LIBAIO_SO_FILENAME = "libaio.so.1.0.2"  # actual file built by Makefile
LIBAIO_SONAME = "libaio.so.1"         # embedded in .so; runtime loader looks for this

include_dirs = [
    f"{root_path}/csrc",
    f"{root_path}/csrc/third_party/dlpack/include",
    f"{root_path}/csrc/third_party/pybind11/include",
    libaio_src,  # for <libaio.h>
]

boost_libs_dir = f"{root_path}/csrc/third_party/boost/libs"
# boost_include_dirs = [f"{boost_libs_dir}/{dir}/include" for dir in os.listdir(boost_libs_dir) if os.path.isdir(f"{boost_libs_dir}/{dir}") and not dir.startswith("old")]

# for boost 1.74.0
boost_submodules = [
    "lockfree",
    "align",
    "array",
    "assert",
    "atomic",
    "config",
    "core",
    "integer",
    "iterator",
    "mpl",
    "parameter",
    "predef",
    "static_assert",
    "tuple",
    "type_traits",
    "utility",
    "winapi"
    "concept_check",
    "mp11",
    "conversion",
    "typeof",
    "move",
    "detail",
    "function_types",
    "fusion",
    "optional",
    "smart_ptr",
    "container_hash",
    "io",
    "preprocessor",
    "throw_exception",
]
boost_include_dirs = [f"{boost_libs_dir}/{dir}/include" for dir in boost_submodules]

include_dirs += boost_include_dirs

class BuildExt(build_ext):
    def run(self):
        # Build libaio (shared lib) using its own Makefile
        if os.system(f"make --silent -C {libaio_dir}") != 0:
            raise RuntimeError("libaio make failed")
        # Makefile produces LIBAIO_SO_FILENAME only; -laio needs LIBAIO_SO_LINKNAME. Create symlink so we link to .so not .a.
        link_path = os.path.join(libaio_src, LIBAIO_SO_LINKNAME)
        real_path = os.path.join(libaio_src, LIBAIO_SO_FILENAME)
        if os.path.isfile(real_path) and not os.path.lexists(link_path):
            os.symlink(LIBAIO_SO_FILENAME, link_path)
        super().run()
        # Copy real .so + soname symlink next to extension so rpath $ORIGIN finds LIBAIO_SONAME at runtime.
        pkg_dir = os.path.join(self.build_lib, "instanttensor")
        os.makedirs(pkg_dir, exist_ok=True)
        if os.path.isfile(real_path):
            shutil.copy2(real_path, os.path.join(pkg_dir, LIBAIO_SONAME))


def get_ext_modules():
    debug = os.environ.get("DEBUG", "0") == "1"
    cxx_flags = ["-std=c++17", "-DUSE_C10D_NCCL"]
    cxx_flags += ["-O0", "-g"] if debug else []

    return [
        Extension(
            name="instanttensor._C",
            sources=["csrc/main.cpp"],
            include_dirs=include_dirs,
            library_dirs=[libaio_src],
            libraries=["dl", "aio"],
            extra_compile_args=cxx_flags,
            extra_link_args=["-Wl,-rpath,$ORIGIN"],
        )
    ]


setup(
    ext_modules=get_ext_modules(),
    cmdclass={"build_ext": BuildExt},
)
