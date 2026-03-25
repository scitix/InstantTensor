#!/usr/bin/env python
import os
import shutil
from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext

# setuptools requires paths relative to setup.py directory, /-separated (no absolute paths)

root_path = os.path.dirname(os.path.abspath(__file__))
libaio_dir = f"{root_path}/csrc/third_party/libaio"
libaio_src = f"{libaio_dir}/src"
# Vendored libaio: use a private SONAME so DT_NEEDED is not "libaio.so.1" (avoids clashing with
# the system library in RPATH/LD_LIBRARY_PATH). Same flow for pip install . / -e . / wheel / sdist:
# build_ext always runs make + link + copy beside _C.so. Makefile: libname=$(soname).0.2 (minor/micro).
LIBAIO_SONAME = "libinstanttensor_aio.so.1"
LIBAIO_SO_FILENAME = "libinstanttensor_aio.so.1.0.2"
LIBAIO_SO_LINKNAME = "libinstanttensor_aio.so"  # -linstanttensor_aio resolves via this symlink
LIBAIO_LINK_SHORT = "instanttensor_aio"  # gcc -l{LIBAIO_LINK_SHORT} -> libinstanttensor_aio.so
package_name = "instanttensor"

include_dirs = [
    f"{root_path}/csrc",
    f"{root_path}/csrc/third_party/dlpack/include",
    f"{root_path}/csrc/third_party/pybind11/include",
    libaio_src,  # for <libaio.h>
]

boost_libs_dir = f"{root_path}/csrc/third_party/boost/libs"
# boost_include_dirs = [f"{boost_libs_dir}/{dir}/include" for dir in os.listdir(boost_libs_dir) if os.path.isdir(f"{boost_libs_dir}/{dir}") and not dir.startswith("old")]

boost_include_dirs = [f"{boost_libs_dir}/{dir}/include" for dir in os.listdir(boost_libs_dir) if os.path.isdir(f"{boost_libs_dir}/{dir}")]

include_dirs += boost_include_dirs

def rm_rf(path: str) -> None:
    """Remove a file or directory, even if it is a symlink."""
    if not os.path.lexists(path):   # Remove even if it is a symlink
        return
    if os.path.islink(path):        # Remove symlink
        os.unlink(path)
    elif os.path.isdir(path):       # Remove directory
        shutil.rmtree(path)
    else:                           # Remove file
        os.unlink(path)

class BuildExt(build_ext):
    def run(self):
        # Build libaio using its own Makefile (force rebuild with -B)
        # Override upstream soname=libaio.so.1 so the shared object is not named like distro libaio.
        make_cmd = f"make --silent -B -C {libaio_dir} soname={LIBAIO_SONAME}"
        print(make_cmd)
        if os.system(make_cmd) != 0:
            raise RuntimeError("libaio make failed")
        # Makefile produces LIBAIO_SO_FILENAME only; -laio needs LIBAIO_SO_LINKNAME. Create symlink so we link to .so not .a.
        link_path = os.path.join(libaio_src, LIBAIO_SO_LINKNAME)
        real_path = os.path.join(libaio_src, LIBAIO_SO_FILENAME)
        assert os.path.isfile(real_path), f"{LIBAIO_SO_FILENAME} not found after make"
        if os.path.lexists(link_path):
            print(f"removing existing {link_path}")
            rm_rf(link_path)
        print(f"creating symlink {link_path} -> {LIBAIO_SO_FILENAME} ({real_path})")
        os.symlink(LIBAIO_SO_FILENAME, link_path)
        super().run()
        # Copy LIBAIO_SONAME next to extension so rpath $ORIGIN finds it at runtime.
        ext_fullpath = self.get_ext_fullpath(f"{package_name}._C")
        target_dir = os.path.dirname(os.path.abspath(ext_fullpath))
        os.makedirs(target_dir, exist_ok=True)
        install_path = os.path.join(target_dir, LIBAIO_SONAME)
        print(f"copying {real_path} to {install_path}")
        shutil.copy2(real_path, install_path)
        clean_cmd = f"make --silent -C {libaio_dir} clean soname={LIBAIO_SONAME}"
        print(clean_cmd)
        if os.system(clean_cmd) != 0:
            raise RuntimeError("libaio make clean failed")


def get_ext_modules():
    debug = os.environ.get("DEBUG", "0") == "1"
    cxx_flags = ["-std=c++17"]
    cxx_flags += ["-O0", "-g"] if debug else []

    return [
        Extension(
            name=f"{package_name}._C",
            sources=["csrc/main.cpp"],
            include_dirs=include_dirs,
            library_dirs=[libaio_src],
            libraries=["dl", LIBAIO_LINK_SHORT],
            extra_compile_args=cxx_flags,
            extra_link_args=["-Wl,-rpath,$ORIGIN"],
        )
    ]


setup(
    ext_modules=get_ext_modules(),
    cmdclass={"build_ext": BuildExt},
)
