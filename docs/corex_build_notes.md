# Corex clang++ build notes

This document summarizes the changes made in this environment to build bitscom with the modified CUDA toolkit at /usr/local/corex-4.4.0 and clang++ as the CUDA compiler.

## Scope

- Target CUDA toolkit: /usr/local/corex-4.4.0
- CUDA compiler: /usr/local/corex-4.4.0/bin/clang++
- Python env: /root/miniconda3/envs/ytli_test

## Code changes

File: setup.py

1) Prefer the corex CUDA toolkit headers by default.
   - If CUDA_HOME is not set and /usr/local/corex-4.4.0 exists, use it.
   - Add the corex include paths before other CUDA headers.

2) Avoid mixing conda CUDA headers when using corex.
   - When CUDA_HOME points to /usr/local/corex-4.4.0, skip conda-provided
     cuda_runtime/cublas/curand headers to avoid nv/target mismatch errors.
   - Keep NCCL headers available from conda or site-packages.

3) Allow a custom CUDA compiler.
   - Add BITSCOM_CUDA_COMPILER and map it to CUDA_NVCC_EXECUTABLE so
     PyTorch's extension build uses clang++ for .cu compilation.

## Build command used

From the repo root:

CUDA_HOME=/usr/local/corex-4.4.0 \
BITSCOM_CUDA_COMPILER=/usr/local/corex-4.4.0/bin/clang++ \
/root/miniconda3/envs/ytli_test/bin/python -m pip install -e . --no-build-isolation

## Tests run

/root/miniconda3/envs/ytli_test/bin/python -m pytest -q

Result: 49 passed, 13 skipped.

Note: PyTorch warned about destroy_process_group() not being called, but tests
still completed successfully.

---

# CoreX 4.5.0 on the line-side machines (cloud powerone1 / edge powerone2)

`powerone1` and `powerone2` are local SSH aliases for the cloud-side and
edge-side boxes; they are not project names. Both run CoreX 4.5.0 on Iluvatar
TG-V300X with the conda env `udtca` (python 3.12 / torch 2.10.0). Four
host-specific problems show up there; verified on powerone2 (edge) on
2026-10-23 with bitscom 0cec3f6.

## 1. `nvcc not found at '<corex-toolkit>/ixdriver/bin/nvcc'`

CoreX's torch does **not** use the `CUDA_HOME` environment variable. In
`torch/utils/cpp_extension.py`:

```python
COREX_HOME = _find_corex_home()   # dladdr()s the loaded libcudart.so
CUDA_HOME = COREX_HOME           # when torch.corex
```

`_find_corex_home()` takes the path of whichever `libcudart.so` is actually
loaded (`ctypes.CDLL('libcudart.so')` + `dladdr`) and strips two directory
levels. On these hosts `LD_LIBRARY_PATH` starts with
`<corex-toolkit>/ixdriver/lib64`, so `COREX_HOME` becomes
`<corex-toolkit>/ixdriver`, which has no `bin/nvcc`:

```
error: nvcc not found at '/data/tianshu/20260720/corex/corex/corex-toolkit/ixdriver/bin/nvcc'.
Ensure CUDA path '/data/tianshu/20260720/corex/corex/corex-toolkit/ixdriver' is correct.
```

Fix: put `<repo-root>/.corex-4.5.0/lib64` first on `LD_LIBRARY_PATH`, so
`COREX_HOME` resolves to the bundle. Check with
`python -c "import torch.utils.cpp_extension as e; print(e.COREX_HOME)"`.

## 2. `cuda_runtime.h` missing under CUDA_HOME

Using the launcher's `CUDA_HOME=/usr/local/corex` also fails because that
install has only 36 top-level headers (`cuda_fp16.h`, `sm_*_intrinsics.h`,
...) and no `cuda_runtime.h`; the file ships under
`<corex-toolkit>/ixdriver/include/cuda_runtime.h`, which is not on any default
search path.

Fix: `CUDA_HOME=<repo-root>/.corex-4.5.0` (same bundle as above). It has
`bin/clang++` (the same clang 22 as the toolkit's `ixcc`), `include/`
(177 entries, including `cuda_runtime.h`, `nccl.h`, `IX/`) and
`lib64/libcudart.so`.

## 3. `cannot find -lnccl`

The `udtca` conda env ships no `libnccl`, and the bundle's `lib64` only has
libcudart, so `setup.py` falls back to `-lnccl` and the link step dies:

```
/home/.../compiler_compat/ld: cannot find -lnccl: No such file or directory
```

Fix: `export NCCL_LIB_DIR=/usr/local/corex/lib64` (the complete 4.5.0 install
has `libnccl.so` -> `libnccl.so.4.4.0.2289`).

## 4. `<climits>` not found when compiling .cu

```
clang++: warning: future releases of the clang compiler will prefer GCC
installations containing libstdc++ include directories;
'/usr/lib/gcc/x86_64-linux-gnu/11' would be chosen over '/usr/lib/gcc/x86_64-linux-gnu/12'
/.../ixcc/lib64/clang/22/include/__clang_cuda_runtime_wrapper.h:58:10:
fatal error: 'climits' file not found
```

CoreX clang++ picks `/usr/lib/gcc/x86_64-linux-gnu/12`, which on these hosts
only contains the gcc-12 C compiler (`cc1`, no `cc1plus`); there is no
`/usr/include/c++/12` either, so clang's implicit C++ include dir
(`/usr/lib/gcc/.../12/../../../../include/c++`) resolves to nothing.

Fix: `export CPLUS_INCLUDE_PATH=/usr/include/c++/11:/usr/include/x86_64-linux-gnu/c++/11`
(the host's `g++` is 11.4.0, and `libstdc++-11-dev` is what is installed).

## Working command

powerone2's `~/.bashrc` exports the four settings above, so there it is
simply:

```bash
conda activate udtca && cd /data/tangruijing/udtca/bitscom
python -m pip install -e . --no-build-isolation
```

Without the `.bashrc` block:

```bash
CX=/data/tangruijing/udtca/.corex-4.5.0
export LD_LIBRARY_PATH=$CX/lib64:$LD_LIBRARY_PATH
export CUDA_HOME=$CX
export BITSCOM_CUDA_COMPILER=$CX/bin/clang++
export NCCL_LIB_DIR=/usr/local/corex/lib64
export CPLUS_INCLUDE_PATH=/usr/include/c++/11:/usr/include/x86_64-linux-gnu/c++/11
python -m pip install -e . --no-build-isolation
```

Verified with `python -c "from bitscom._lowbit_c import ProcessGroupLowBit"`
(also exposes `revoke_lowbit_comm`, `lowbit_comm_ptr`) and
`import bitscom._lowbit_cuda`.

Note: none of this is needed at training time; the launcher
(`udtca-config/qwen14b/run_single_node.sh`) sets its own CoreX paths.
