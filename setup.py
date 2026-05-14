import multiprocessing
import os
import platform
import shutil
import subprocess
import sys
import sysconfig
from distutils.command.clean import clean

from setuptools import Extension, find_packages, setup


# Env Variables
IS_DARWIN = platform.system() == "Darwin"
IS_WINDOWS = platform.system() == "Windows"

# Accelerator platform: "cuda" (default) or "maca"
ACCELERATOR = os.environ.get("ACCELERATOR", "cuda").lower()

BASE_DIR = os.path.dirname(os.path.realpath(__file__))

# Only run cmake build for actual build commands, not metadata collection
BUILD_COMMANDS = {
    "build",
    "build_ext",
    "install",
    "develop",
    "bdist_wheel",
    "bdist_egg",
    "editable_wheel",
}
RUN_BUILD_DEPS = any(arg in BUILD_COMMANDS for arg in sys.argv)


def _ensure_maca_cudart_shim():
    """On MACA, compile and load a complete cudart shim before importing torch.

    MACA's libsymbol_cu.so provides CUDA runtime symbols but without the
    @@libcudart.so.12 version tags that PyTorch's .so files require.
    We build a single shared library (accelerator/csrc/maca/cudart_shim.c) that:
      1. Forwards ~79 symbols to libsymbol_cu.so via dlsym
      2. Stubs ~11 symbols for APIs missing from MACA entirely
      3. Tags ALL exported symbols with @@libcudart.so.12 via a version script
    """
    import ctypes

    csrc = os.path.join(BASE_DIR, "accelerator", "csrc", "maca")
    build_dir = os.path.join(BASE_DIR, "build")
    os.makedirs(build_dir, exist_ok=True)

    shim_so = os.path.join(build_dir, "libcudart_shim.so")
    shim_src = os.path.join(csrc, "cudart_shim.c")
    version_script = os.path.join(csrc, "libcudart.version")

    inputs = [shim_src, version_script]
    if not os.path.exists(shim_so) or any(
        os.path.exists(s) and os.path.getmtime(s) > os.path.getmtime(shim_so)
        for s in inputs
    ):
        subprocess.check_call(
            [
                "gcc",
                "-shared",
                "-fPIC",
                "-o",
                shim_so,
                shim_src,
                f"-Wl,--version-script={version_script}",
                "-Wl,-soname,libcudart.so.12",
                "-ldl",
            ]
        )

    ctypes.CDLL(shim_so, mode=ctypes.RTLD_GLOBAL)


if ACCELERATOR == "maca":
    _ensure_maca_cudart_shim()


def make_relative_rpath_args(path):
    if IS_DARWIN:
        return ["-Wl,-rpath,@loader_path/" + path]
    elif IS_WINDOWS:
        return []
    else:
        return ["-Wl,-rpath,$ORIGIN/" + path]


def _subprocess_env_without_pip_build_overlay():
    """Copy of the process environment with pip's ephemeral overlay removed from PYTHONPATH."""
    env = os.environ.copy()
    old_pp = env.pop("PYTHONPATH", None)
    if old_pp:
        kept = [
            p
            for p in old_pp.split(os.pathsep)
            if p and "pip-build-env" not in p.replace("\\", "/")
        ]
        if kept:
            env["PYTHONPATH"] = os.pathsep.join(kept)
    return env


def get_pytorch_dir():
    """Directory of the installed ``torch`` package (contains ``lib/libtorch.so``).

    Pip can prepend a temporary ``pip-build-env-*/overlay/...`` to ``PYTHONPATH`` while
    building; that tree is removed before ``cmake --build`` runs, so resolving torch
    there bakes broken absolute paths into the CMake-generated rules. Resolve torch in
    a subprocess with ephemeral entries stripped from ``PYTHONPATH``.
    """
    env = _subprocess_env_without_pip_build_overlay()
    try:
        out = subprocess.check_output(
            [
                sys.executable,
                "-c",
                "import os, torch; print(os.path.dirname(os.path.realpath(torch.__file__)))",
            ],
            env=env,
            text=True,
        ).strip()
    except subprocess.CalledProcessError as e:
        raise RuntimeError(
            "Could not locate PyTorch (import torch failed). "
            "Install torch in this environment before building torch_fl."
        ) from e
    if not out or not os.path.isdir(out):
        raise RuntimeError(f"Invalid PyTorch directory resolved: {out!r}")
    return out


def get_flaggems_cmake_dir():
    """Directory containing ``FlagGemsConfig.cmake`` from the ``flag_gems`` wheel, if installed."""
    env = _subprocess_env_without_pip_build_overlay()
    probe = r"""
import importlib.util as u
import os
import sys

spec = u.find_spec("flag_gems")
if spec is None:
    sys.exit(2)
locs = getattr(spec, "submodule_search_locations", None)
if locs:
    root = locs[0]
else:
    root = os.path.dirname(spec.origin or "")
cfg = os.path.join(root, "lib", "cmake", "FlagGems")
if not os.path.isfile(os.path.join(cfg, "FlagGemsConfig.cmake")):
    sys.exit(3)
sys.stdout.write(cfg)
"""
    try:
        out = subprocess.check_output(
            [sys.executable, "-c", probe],
            env=env,
            text=True,
        ).strip()
    except subprocess.CalledProcessError:
        return None
    return out if out and os.path.isdir(out) else None


def build_deps():
    build_dir = os.path.join(BASE_DIR, "build")
    cache_path = os.path.join(build_dir, "CMakeCache.txt")
    if os.path.isfile(cache_path):
        try:
            with open(cache_path, encoding="utf-8", errors="ignore") as f:
                stale = "pip-build-env" in f.read()
        except OSError:
            stale = False
        if stale:
            shutil.rmtree(build_dir, ignore_errors=True)
    os.makedirs(build_dir, exist_ok=True)

    pytorch_dir = get_pytorch_dir()
    cmake_args = [
        "-DCMAKE_INSTALL_PREFIX="
        + os.path.realpath(os.path.join(BASE_DIR, "torch_fl")),
        "-DPYTHON_INCLUDE_DIR=" + sysconfig.get_paths().get("include"),
        "-DPYTORCH_INSTALL_DIR=" + pytorch_dir,
    ]

    cmake_args.append(f"-DACCELERATOR={ACCELERATOR}")

    # FlagGems C++ operators: directory containing FlagGemsConfig.cmake (env, pip, or CMake glob)
    flaggems_dir = os.environ.get("FLAGGEMS_DIR") or os.environ.get("FlagGems_DIR")
    if not flaggems_dir:
        flaggems_dir = get_flaggems_cmake_dir()
    if flaggems_dir:
        cmake_args.append(f"-DFlagGems_DIR={flaggems_dir}")

    if ACCELERATOR == "maca":
        # Muxi MACA SDK: no nvcc needed. CMakeLists.txt pre-creates
        # torch::cudart to skip PyTorch's cuda.cmake entirely.
        maca_path = (
            os.environ.get("MACA_PATH") or os.environ.get("MACA_HOME") or "/opt/maca"
        )
        cmake_args.append(f"-DMACA_PATH={maca_path}")
    else:
        # Add CUDA toolkit path if available
        cuda_home = os.environ.get("CUDA_HOME") or os.environ.get("CUDA_PATH")
        if cuda_home:
            cmake_args.append(f"-DCMAKE_CUDA_COMPILER={cuda_home}/bin/nvcc")

    subprocess.check_call(
        ["cmake", BASE_DIR] + cmake_args, cwd=build_dir, env=os.environ
    )

    build_args = [
        "--build",
        ".",
        "--target",
        "install",
        "--config",  # For multi-config generators
        "Release",
        "--",
    ]

    if IS_WINDOWS:
        build_args += ["/m:" + str(multiprocessing.cpu_count())]
    else:
        build_args += ["-j", str(multiprocessing.cpu_count())]

    command = ["cmake"] + build_args
    subprocess.check_call(command, cwd=build_dir, env=os.environ)


class BuildClean(clean):
    def run(self):
        for i in ["build", "install", "torch_fl/lib"]:
            dirs = os.path.join(BASE_DIR, i)
            if os.path.exists(dirs) and os.path.isdir(dirs):
                shutil.rmtree(dirs)

        for dirpath, _, filenames in os.walk(os.path.join(BASE_DIR, "torch_fl")):
            for filename in filenames:
                if filename.endswith(".so"):
                    os.remove(os.path.join(dirpath, filename))


def main():
    if RUN_BUILD_DEPS:
        build_deps()

    if IS_WINDOWS:
        # /NODEFAULTLIB makes sure we only link to DLL runtime
        # and matches the flags set for protobuf and ONNX
        extra_link_args: list[str] = ["/NODEFAULTLIB:LIBCMT.LIB"] + [
            *make_relative_rpath_args("lib")
        ]
        # /MD links against DLL runtime
        # and matches the flags set for protobuf and ONNX
        # /EHsc is about standard C++ exception handling
        extra_compile_args: list[str] = ["/MD", "/FS", "/EHsc"]
    else:
        extra_link_args = [*make_relative_rpath_args("lib")]
        extra_compile_args = [
            "-Wall",
            "-Wextra",
            "-Wno-strict-overflow",
            "-Wno-unused-parameter",
            "-Wno-missing-field-initializers",
            "-Wno-unknown-pragmas",
            "-fno-strict-aliasing",
        ]

    ext_modules = [
        Extension(
            name="torch_fl._C",
            sources=["torch_fl/csrc/stub.c"],
            language="c",
            extra_compile_args=extra_compile_args,
            libraries=["torch_bindings"],
            library_dirs=[os.path.join(BASE_DIR, "torch_fl/lib")],
            extra_link_args=extra_link_args,
        )
    ]

    package_data = {
        "torch_fl": [
            "lib/*.so*",
            "lib/*.dylib*",
            "lib/*.dll",
            "lib/*.lib",
            "backends.conf",
        ]
    }

    setup(
        name="torch_fl",
        version="0.1.0",
        description="FlagGems operators as a custom PyTorch device (flagos)",
        author="FlagGems Team",
        packages=find_packages(include=["torch_fl*", "accelerator*"]),
        package_dir={"": "."},
        package_data=package_data,
        ext_modules=ext_modules,
        cmdclass={
            "clean": BuildClean,  # type: ignore[misc]
        },
        include_package_data=False,
        python_requires=">=3.8",
        install_requires=[
            "torch",
        ],
    )


if __name__ == "__main__":
    main()
