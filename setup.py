"""
使用 torch.utils.cpp_extension 构建 C++ extension。
推荐使用此方式构建（而非 Bazel），以最大程度兼容 PyTorch 生态。

安装:
    pip install -e .

验证:
    python -c "from bitscom._lowbit_c import ProcessGroupLowBit; print('OK')"
"""

import os
import shutil
import site
import sysconfig
import json

from setuptools import setup, find_packages
import torch
from torch.utils.cpp_extension import BuildExtension, CppExtension, CUDAExtension


_PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))


def _existing(path):
    return os.path.isdir(path)


def _use_nvidia_cuda_headers(cuda_home):
    return not (cuda_home and os.path.abspath(cuda_home) == "/usr/local/corex-4.4.0")


def _collect_include_dirs():
    include_dirs = [
        _PROJECT_ROOT,
        os.path.join(_PROJECT_ROOT, "cpp", "include"),
    ]

    cuda_home = os.environ.get("CUDA_HOME")
    if not cuda_home and os.path.isdir("/usr/local/corex-4.4.0"):
        cuda_home = "/usr/local/corex-4.4.0"
    if not cuda_home:
        cuda_home = "/usr/local/cuda"

    sys_cuda_candidates = [
        os.path.join(cuda_home, "include"),
        os.path.join(cuda_home, "targets", "x86_64-linux", "include"),
    ]
    include_dirs.extend([p for p in sys_cuda_candidates if _existing(p)])

    env_include = os.environ.get("NCCL_INCLUDE_DIR")
    if env_include:
        include_dirs.append(env_include)

    conda_prefix = os.environ.get("CONDA_PREFIX")
    if conda_prefix:
        candidates = [
            os.path.join(conda_prefix, "lib", "python3.12", "site-packages", "nvidia", "nccl", "include"),
            os.path.join(conda_prefix, "include"),
        ]
        if _use_nvidia_cuda_headers(cuda_home):
            candidates.extend(
                [
                    os.path.join(
                        conda_prefix,
                        "lib",
                        "python3.12",
                        "site-packages",
                        "nvidia",
                        "cu13",
                        "include",
                    ),
                    os.path.join(
                        conda_prefix,
                        "lib",
                        "python3.12",
                        "site-packages",
                        "nvidia",
                        "cuda_runtime",
                        "include",
                    ),
                ]
            )
        include_dirs.extend([p for p in candidates if _existing(p)])

    # Probe active Python site-packages to find pip/conda-provided CUDA/NCCL headers.
    site_roots = []
    try:
        site_roots.extend(site.getsitepackages())
    except Exception:
        pass

    purelib = sysconfig.get_paths().get("purelib")
    if purelib:
        site_roots.append(purelib)

    site_roots.append(os.path.dirname(torch.__file__))

    nvidia_include_suffixes = [
        os.path.join("nvidia", "nccl", "include"),
    ]
    if _use_nvidia_cuda_headers(cuda_home):
        nvidia_include_suffixes.extend(
            [
                os.path.join("nvidia", "cu13", "include"),
                os.path.join("nvidia", "cuda_runtime", "include"),
                os.path.join("nvidia", "cublas", "include"),
                os.path.join("nvidia", "curand", "include"),
            ]
        )
    for root in site_roots:
        for suffix in nvidia_include_suffixes:
            candidate = os.path.join(root, suffix)
            if _existing(candidate):
                include_dirs.append(candidate)

    # CUDA_HOME already applied above to prioritize toolkit headers over conda ones.

    # De-duplicate while preserving order.
    seen = set()
    uniq = []
    for p in include_dirs:
        if p not in seen:
            uniq.append(p)
            seen.add(p)
    return uniq


def _torch_lib_dir():
    return os.path.join(os.path.dirname(torch.__file__), "lib")


def _collect_library_dirs():
    lib_dirs = [_torch_lib_dir()]

    env_lib = os.environ.get("NCCL_LIB_DIR")
    if env_lib and _existing(env_lib):
        lib_dirs.append(env_lib)

    conda_prefix = os.environ.get("CONDA_PREFIX")
    if conda_prefix:
        for p in [
            os.path.join(conda_prefix, "lib"),
            os.path.join(conda_prefix, "lib64"),
        ]:
            if _existing(p):
                lib_dirs.append(p)

    # Probe nvidia nccl runtime package location from Python env.
    site_roots = []
    try:
        site_roots.extend(site.getsitepackages())
    except Exception:
        pass
    purelib = sysconfig.get_paths().get("purelib")
    if purelib:
        site_roots.append(purelib)

    for root in site_roots:
        candidate = os.path.join(root, "nvidia", "nccl", "lib")
        if _existing(candidate):
            lib_dirs.append(candidate)

    seen = set()
    uniq = []
    for p in lib_dirs:
        if p not in seen:
            uniq.append(p)
            seen.add(p)
    return uniq


def _cuda_compiler_from_env():
    return (
        os.environ.get("BITSCOM_CUDA_COMPILER")
        or os.environ.get("CUDA_NVCC_EXECUTABLE")
        or os.environ.get("NVCC")
    )


def _has_nvcc():
    compiler = _cuda_compiler_from_env()
    if compiler:
        return shutil.which(compiler) is not None or os.path.exists(compiler)
    if shutil.which("nvcc"):
        return True
    cuda_home = os.environ.get("CUDA_HOME", "/usr/local/cuda")
    return os.path.exists(os.path.join(cuda_home, "bin", "nvcc"))


def _maybe_set_cuda_nvcc():
    compiler = os.environ.get("BITSCOM_CUDA_COMPILER")
    if compiler and not os.environ.get("CUDA_NVCC_EXECUTABLE"):
        # Let PyTorch extension build treat this compiler as nvcc.
        os.environ["CUDA_NVCC_EXECUTABLE"] = compiler


class BuildExtensionWithCompileCommands(BuildExtension):
    """BuildExtension that emits compile_commands.json for clangd."""

    def build_extensions(self):
        compiler = getattr(self, "compiler", None)
        spawn = getattr(compiler, "spawn", None)
        if compiler is None or spawn is None:
            super().build_extensions()
            return

        records = []
        project_root = os.path.abspath(os.path.dirname(__file__))

        def recording_spawn(cmd):
            try:
                # distutils typically uses: <compiler> ... -c <src> -o <obj>
                if "-c" in cmd:
                    src = cmd[cmd.index("-c") + 1]
                elif "--compile" in cmd:
                    # nvcc may use --compile <src>
                    src = cmd[cmd.index("--compile") + 1]
                else:
                    src = None

                if src and src.endswith((".c", ".cc", ".cpp", ".cxx", ".cu")):
                    src_abs = os.path.abspath(src)
                    records.append(
                        {
                            "directory": project_root,
                            "file": src_abs,
                            # Use argv-style records to avoid shell-quoting ambiguity,
                            # which is especially important for nvcc options.
                            "arguments": [str(p) for p in cmd],
                        }
                    )
            except Exception:
                # Never block compilation because of compile db recording.
                pass

            return spawn(cmd)

        compiler.spawn = recording_spawn
        try:
            super().build_extensions()
        finally:
            compiler.spawn = spawn

        if records:
            # Keep last command for each source file to avoid noisy duplicates.
            by_file = {entry["file"]: entry for entry in records}
            output = os.path.join(project_root, "compile_commands.json")
            with open(output, "w", encoding="utf-8") as f:
                json.dump(list(by_file.values()), f, indent=2)


_maybe_set_cuda_nvcc()

ext_modules = [
    CppExtension(
        name="bitscom._lowbit_c",
        sources=[
            "cpp/src/process_group_lowbit.cc",
            "cpp/src/bindings.cc",
        ],
        include_dirs=_collect_include_dirs(),
        library_dirs=_collect_library_dirs(),
        libraries=[
            "torch",
            "torch_cpu",
            "torch_cuda",
            "torch_python",
            "c10",
            "c10_cuda",
        ],
        extra_compile_args={
            "cxx": ["-std=c++17", "-O2", "-DUSE_C10D_NCCL"],
        },
    ),
]

if _has_nvcc():
    ext_modules.append(
        CUDAExtension(
            name="bitscom._lowbit_cuda",
            sources=[
                "cpp/src/lowbit_cuda_bindings.cpp",
                "cpp/src/lowbit_cuda_kernels.cu",
            ],
            include_dirs=_collect_include_dirs(),
            library_dirs=_collect_library_dirs(),
            libraries=["torch", "torch_cpu", "torch_cuda", "c10", "c10_cuda"],
            extra_compile_args={
                "cxx": ["-std=c++17", "-O2"],
                "nvcc": ["-O2", "-std=c++17"],
            },
        )
    )

setup(
    name="bitscom",
    version="0.1.0",
    description="Low-bit distributed communication primitives for PyTorch",
    author="Aerithy",
    package_dir={"": "python"},
    packages=find_packages(where="python"),
    ext_modules=ext_modules,
    cmdclass={"build_ext": BuildExtensionWithCompileCommands},
    python_requires=">=3.8",
    install_requires=[
        # "torch>=2.0",
    ],
)
