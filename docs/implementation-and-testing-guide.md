# bitscom 改造说明与测试指南

## 1. 这次做了什么

本次改造围绕 5 个目标展开：

1. 让“低比特通信”在 Python 层有可运行、可验证的完整实现（而不是仅占位）。
2. 让 backend 注册在扩展缺失时行为可控且可诊断。
3. 统一 bitwidth 约束，避免文档与实现不一致。
4. 补齐高覆盖单元测试，保证核心逻辑可以在非分布式、非 GPU 环境下验证。
5. 增加可执行的性能基准测试流程，并默认关闭以避免 CI 波动。

## 2. 怎么实现的

### 2.1 新增低比特量化与打包模块

新增文件：`python/bitscom/quantization.py`

实现内容：

- 支持 bitwidth：`1, 2, 4, 8, 12, 16`
- `quantize_tensor` / `dequantize_tensor`
- `quantize_tensor_blockwise` / `dequantize_tensor_blockwise`
- `pack_lowbit` / `unpack_lowbit`
- `compress_tensor` / `decompress_tensor`
- `roundtrip_tensor`

设计说明：

- 使用 per-tensor 与 blockwise 对称量化（按最大绝对值计算 scale）。
- 先量化为 int16，再做 bit-pack，便于统一支持多 bitwidth。
- 对 1-bit 使用两级表示，避免数值边界导致的除零问题。
- 所有路径尽量使用张量向量化操作，减少 Python 循环。

### 2.2 API 层增强

修改文件：`python/bitscom/api.py`

新增能力：

- `LowBitGroup` 的 bitwidth 严格校验（只允许支持集合）。
- `simulate_quantization` 模式：
  - 在调用 collective 前先做一次量化-反量化模拟。
  - 用于在非 lowbit backend 环境下验证量化误差与行为。
- 新增 `compress` / `decompress` 方法，直接暴露压缩与解压能力。

### 2.3 backend 注册健壮化

修改文件：`python/bitscom/lowbit_backend.py`

新增能力：

- C++ 扩展延迟可用性检测（`try/except import`）。
- 扩展缺失时给出明确错误信息。
- 注册幂等（重复调用 `register_lowbit_backend` 不重复注册）。
- 新增 `is_extension_available()`。

### 2.4 包导出与文档整理

修改文件：

- `python/bitscom/__init__.py`
- `README.md`

更新内容：

- 补充对新能力的统一导出。
- README 明确了：项目定位、已实现能力、仍待实现能力、测试与性能测试入口。

### 2.5 测试体系完善

新增/修改：

- 新增 `tests/test_quantization.py`：量化与 pack/unpack 核心正确性。
- 新增 `tests/test_api.py`：API 行为与 async/sync 分支。
- 新增 `tests/test_lowbit_backend.py`：注册流程与异常分支。
- 新增 `tests/test_perf_quantization.py`：性能基准测试。
- 新增 `tests/test_single_gpu_train_e2e.py`：单卡 lowbit backend 训练 e2e（含性能指标输出）。
- 新增 `tests/conftest.py`：测试环境路径处理。
- 修改 `tests/test_e2e.py`：加入 integration 标记与未初始化分布式时 skip。
- 修改 `pytest.ini`：限制 `testpaths=tests`，避免误收集 bazel 外部测试。

## 3. 采用了什么优化方法

本次优化分为“算法层”和“工程层”。

### 3.1 算法层优化

- 低比特压缩：
  - 将浮点数据映射到低比特整数域后进行 bit-pack，减少传输字节数。
- 向量化位运算：
  - 对 `1/2/4/8 bit` 使用统一的 shift+mask 向量化路径。
  - 对 `12 bit` 使用 3-byte 打包（2 个值 -> 3 字节）的专门路径。
- 误差控制：
  - 使用 scale 控制量化范围，保证反量化误差上界可预期。

### 3.2 工程层优化

- 幂等注册与清晰错误信息，减少分布式初始化阶段排障成本。
- 性能测试默认 gated（`BITSCOM_RUN_PERF=1`），降低日常回归不稳定性。
- 通过 `pytest.ini` 限制收集范围，提高测试执行确定性与速度。

## 4. 核心逻辑在哪

核心文件分布如下：

- 量化与打包：`python/bitscom/quantization.py`
- API 封装：`python/bitscom/api.py`
- backend 注册：`python/bitscom/lowbit_backend.py`
- 包入口导出：`python/bitscom/__init__.py`
- C++ backend 骨架（NCCL 包装）：`cpp/src/process_group_lowbit.cc`
- Python/C++ 绑定：`cpp/src/bindings.cc`

测试相关：

- 单元测试：`tests/test_quantization.py`, `tests/test_api.py`, `tests/test_lowbit_backend.py`
- 集成测试：`tests/test_e2e.py`
- 性能测试：`tests/test_perf_quantization.py`
- 单卡训练 e2e：`tests/test_single_gpu_train_e2e.py`
- pytest 配置：`pytest.ini`

## 5. 单测怎么跑（具体流程）

下面给出两种方式，任选其一。

### 方式 A：直接激活 conda 环境

1. 进入仓库根目录

```bash
cd /home/aerith/bitscom
```

2. 激活环境

```bash
conda activate bitscom
```

3. 安装测试依赖（如未安装）

```bash
python -m pip install pytest
```

4. 运行单元测试与可安全执行的测试

```bash
pytest -q
```

5. 可选：仅跑某一类测试

```bash
pytest -q tests/test_quantization.py
pytest -q tests/test_api.py
pytest -q tests/test_lowbit_backend.py
pytest -q tests/test_single_gpu_train_e2e.py
```

### 方式 B：不激活环境，直接 conda run

```bash
cd /home/aerith/bitscom
conda run -n bitscom pytest -q
```

### 结果说明

- 当前状态应通过：`38 passed, 4 skipped`
- `skipped` 主要来自 integration 场景（例如未初始化分布式）。

## 5.1 单卡训练 e2e（新增）

为避免必须依赖多卡环境，新增了单卡训练 e2e：

- `test_single_gpu_lowbit_training_e2e`
  - 使用随机输入训练 TinyResNet 若干步。
  - 训练中梯度走 `LowBitGroup(bitwidth=4).all_reduce`。
  - 断言参数确实发生更新。
- `test_single_gpu_lowbit_training_cifar10_small_steps`
  - 使用 CIFAR10 做 10 step 小步训练。
  - 默认不下载数据；若本地无数据集可通过环境变量开启下载。

运行命令：

```bash
/home/aerith/miniforge3/envs/bitscom/bin/python -m pytest -q tests/test_single_gpu_train_e2e.py
```

仅跑 CIFAR10 小步版本：

```bash
BITSCOM_ALLOW_DOWNLOAD=1 /home/aerith/miniforge3/envs/bitscom/bin/python -m pytest -q tests/test_single_gpu_train_e2e.py -k cifar10
```

输出指标（每次训练循环都会打印）：

- `avg_step_time_ms`：平均每 step 耗时
- `throughput_samples_per_s`：吞吐（样本/秒）
- `peak_memory_mb`：CUDA 峰值显存（MB）

## 6. 基准测试怎么跑（具体流程）

性能测试默认关闭，执行时需要显式打开环境变量。

1. 进入仓库目录

```bash
cd /home/aerith/bitscom
```

2. 在 bitscom 环境中运行 performance 标记测试

```bash
BITSCOM_RUN_PERF=1 conda run -n bitscom pytest -q -m performance
```

3. 当前基准测试项

- `test_quantization_roundtrip_benchmark_prints_metrics`
  - 对比 `roundtrip_tensor(bitwidth=4)` 与 `x.clone()` 的耗时数量级。
- `test_compression_ratio_benchmark`
  - 验证压缩后字节数小于 fp32 原始字节数。

4. 结果预期

- 当前状态应通过：`2 passed`

## 7. 仍未完成的能力（后续建议）

- C++ 侧真实 CUDA quantize/dequantize kernel 仍待实现。
- `allgather` / `reduce_scatter` 在 C++ backend 内仍是直通 NCCL，未走低比特路径。
- error-feedback 已接入 lowbit allreduce（legacy/EF21/EF21+），其余集合通信仍需补齐。
- 缺少多机多卡的系统级 benchmark 报告脚本（吞吐、延迟、误差联合评估）。
