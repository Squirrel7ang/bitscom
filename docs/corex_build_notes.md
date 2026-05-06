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
