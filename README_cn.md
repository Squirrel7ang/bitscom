# bitscom

bitscom 是一个面向 PyTorch 的低比特分布式通信库。
主要提供：

- 一个自定义 `torch.distributed` 后端：`lowbit`
- 一个 Python 包装 API：`LowBitGroup`
- 一个用于开发与测试的 Python 量化/打包模块

## 仓库目标

核心目标：

- 在分布式训练中，用低比特量化后的表示替代全精度张量进行通信，降低通信流量。

当前架构：

- C++ 后端（`ProcessGroupLowBit`）包装 NCCL process group 调用。
- Python API（`bitscom.LowBitGroup`）暴露集合通信接口。
- Python 量化模块（`bitscom.quantization`）提供可独立测试的压缩/解压逻辑。

## 已实现功能

- 支持量化比特宽度：`1, 2, 4, 8, 12, 16`
- Tensor 量化/反量化工具
- 对应比特宽度的打包/解包逻辑
- `1/2/4/8` bit 的 CUDA 量化/打包/解包 fast path
- 量化随机舍入（stochastic rounding，CPU + CUDA）
- `LowBitGroup.compress()` 与 `LowBitGroup.decompress()` 辅助接口
- 可选模拟模式（`simulate_quantization=True`），用于在通信前验证量化影响
- 低比特 all-reduce 的分块量化与 FP16 放缩因子
- `<8bit` 的 Python 低比特 all-reduce 管线（`all-to-all -> local reduce -> all-gather`）
- 分层方案 A 低比特 all-reduce（`local reduce -> inter lowbit all-reduce -> local broadcast`）
- 多机 CUDA 双流流水调度（`warmup -> steady -> cooldown`）
- 单机仅 `local_group` 快速路径：一次全精度 all-reduce，不走分块流水
- C++ 后端低比特 allreduce 路径第一段量化的 error-feedback
- 显式后端注册参数：`bitwidth` / `error_feedback` / `block_size` / `stage2_error_feedback`
- C++ 扩展缺失时的稳健注册报错
- 构建时自动生成 `compile_commands.json`（用于 clangd）

## 仍待完善

以下事项仍为后续工作：

- C++ 后端中 `allgather` 与 `reduce_scatter` 的低比特实现（当前仍透传到 NCCL）
- 多机端到端 benchmark 与规模化分析
- 进一步统一 C++/Python 量化路径以降低重复实现

## 环境配置（编译前）

扩展编译依赖可用的 Python + PyTorch + CUDA 工具链。

1. 创建并激活独立 conda 环境：

```bash
conda create -n bitscom python=3.12 -y
conda activate bitscom
```

2. 安装基础构建工具：

```bash
python -m pip install --upgrade pip setuptools wheel
```

3. 安装带 CUDA 支持的 PyTorch（示例：CUDA 13.0 wheel）：

```bash
python -m pip install torch --index-url https://download.pytorch.org/whl/cu130
```

4. 可选：若 CUDA/NCCL 不在默认路径，显式导出环境变量：

```bash
export CUDA_HOME=/usr/local/cuda
export NCCL_INCLUDE_DIR=/path/to/nccl/include
export NCCL_LIB_DIR=/path/to/nccl/lib
```

5. 编译前快速自检：

```bash
python -c "import torch; print(torch.__version__, torch.version.cuda)"
nvcc --version
```

## 构建

开发推荐可编辑安装：

```bash
pip install -e .
```

若使用本仓库当前验证环境（conda）：

```bash
/home/aerith/miniforge3/envs/bitscom/bin/python -m pip install -e . --no-build-isolation
```

Corex clang++ 编译链（Corex CUDA 位于 /usr/local/corex-4.4.0）：

```bash
CUDA_HOME=/usr/local/corex-4.4.0 \
BITSCOM_CUDA_COMPILER=/usr/local/corex-4.4.0/bin/clang++ \
/root/miniconda3/envs/ytli_test/bin/python -m pip install -e . --no-build-isolation
```

当前扩展构建会在仓库根目录生成 `compile_commands.json`，便于 clangd 索引。

## 测试

单测与非分布式集成安全测试：

```bash
pytest -q
```

量化专项测试（含 CUDA 路径与随机舍入）：

```bash
/home/aerith/miniforge3/envs/bitscom/bin/python -m pytest -q tests/test_quantization.py
```

Pipeline-A API 行为测试：

```bash
/home/aerith/miniforge3/envs/bitscom/bin/python -m pytest -q tests/test_api.py
```

Pipeline-A 分布式正确性测试（NCCL，至少 2 卡）：

```bash
BITSCOM_RUN_DIST=1 /home/aerith/miniforge3/envs/bitscom/bin/python -m pytest -q tests/test_pipeline_a_correctness.py
```

分布式 e2e（多进程）测试：

```bash
torchrun --nproc_per_node=2 tests/test_e2e.py
```

单卡端到端训练测试（无需多卡）：

```bash
/home/aerith/miniforge3/envs/bitscom/bin/python -m pytest -q tests/test_single_gpu_train_e2e.py
```

单卡 e2e 会输出以下性能指标：

- `avg_step_time_ms`
- `throughput_samples_per_s`
- `peak_memory_mb`

运行 CIFAR10 小步版本（10 step）并允许自动下载数据：

```bash
BITSCOM_ALLOW_DOWNLOAD=1 /home/aerith/miniforge3/envs/bitscom/bin/python -m pytest -q tests/test_single_gpu_train_e2e.py -k cifar10
```

## 性能测试

性能测试默认关闭，以保证 CI 稳定：

```bash
BITSCOM_RUN_PERF=1 pytest -q -m performance
```

### 量化 CPU vs GPU 性能曲线

生成 `quantize_tensor` 的调用延迟/加速曲线（CPU 原始路径 vs CUDA 路径）：

```bash
/home/aerith/miniforge3/envs/bitscom/bin/python benchmarks/quantize_perf_curve.py --bitwidth 4 --min-pow2 10 --max-pow2 23
```

输出文件：

- `benchmarks/outputs/quantize_curve_bw4.csv`
- `benchmarks/outputs/quantize_curve_bw4.png`

结果预览：

![Quantization CPU vs GPU curve](benchmarks/outputs/quantize_curve_bw4.png)

### Error-Feedback 曲线（单比特宽度）

生成 EF 与无 EF 的通信误差 + 训练损失对比曲线：

```bash
/home/aerith/miniforge3/envs/bitscom/bin/python benchmarks/error_feedback_comparison.py --bitwidth 2 --steps 180 --train-steps 140 --vec-size 524288 --batch-size 256 --device cuda:0
```

输出文件：

- `benchmarks/outputs/ef_benefit_curve_bw2.csv`
- `benchmarks/outputs/ef_training_curve_bw2.csv`
- `benchmarks/outputs/ef_comparison_bw2.png`

结果预览：

![Error-feedback single-bitwidth comparison](benchmarks/outputs/ef_comparison_bw2.png)

### Error-Feedback 汇总（多比特宽度）

生成跨多个比特宽度（默认 `2/4/8`）的汇总图：

```bash
/home/aerith/miniforge3/envs/bitscom/bin/python benchmarks/error_feedback_multibw.py --bitwidths 2 4 8 --device cuda:0 --steps 160 --train-steps 120 --vec-size 524288 --batch-size 256
```

关键输出：

- `benchmarks/outputs/ef_multibitwidth_summary.png`

结果预览：

![Error-feedback multi-bitwidth summary](benchmarks/outputs/ef_multibitwidth_summary.png)

### Error-Feedback 模式多模型对比（Loss + 开销）

在多个模型上对比 error-feedback 模式（none / legacy / ef21 / ef21_plus），
输出每个模式的 loss 曲线与综合汇总图：

```bash
python benchmarks/error_feedback_multimodel_summary.py \
	--modes none legacy ef21 ef21_plus \
	--models resnet50 bert gpt2 \
	--bitwidth 4 \
	--nproc 2
```

关键输出：

- `benchmarks/outputs/ef_modes_resnet50_loss_curve_bw4.csv`
- `benchmarks/outputs/ef_modes_resnet50_loss_curve_bw4.png`
- `benchmarks/outputs/ef_modes_bert_loss_curve_bw4.csv`
- `benchmarks/outputs/ef_modes_bert_loss_curve_bw4.png`
- `benchmarks/outputs/ef_modes_gpt2_loss_curve_bw4.csv`
- `benchmarks/outputs/ef_modes_gpt2_loss_curve_bw4.png`
- `benchmarks/outputs/ef_modes_throughput_summary_bw4.csv`
- `benchmarks/outputs/ef_modes_throughput_summary_bw4.png`

脚本会为每个 EF 模式自动调用 `torchrun`。如果已经有 CSV，
可跳过训练仅重绘：

```bash
python benchmarks/error_feedback_multimodel_summary.py --skip-run
```

### 集合通信对比实验：allreduce / reduce_scatter

对比低比特集体通信与全精度 NCCL 基线，输出每步误差与耗时 CSV 和曲线图。

单比特宽度（2 GPU）：

```bash
torchrun --nproc_per_node=2 benchmarks/collective_compare.py \
	--collective allreduce \
	--bitwidth 4 \
	--steps 60 \
	--numel 262144
```

输出文件：

- `benchmarks/outputs/collective_compare_allreduce_bw4.csv`
- `benchmarks/outputs/collective_compare_allreduce_bw4.png`

多比特宽度汇总：

```bash
python benchmarks/collective_compare_multibw.py \
	--collective reduce_scatter \
	--bitwidths 1 2 4 8 \
	--steps 60 \
	--numel 262144 \
	--nproc 2
```

汇总输出：

- `benchmarks/outputs/collective_compare_reduce_scatter_summary.csv`
- `benchmarks/outputs/collective_compare_reduce_scatter_summary.png`

## 后端注册参数

可通过显式参数注册后端：

```python
import bitscom

# 使用显式参数注册 lowbit backend。
bitscom.init(bitwidth=4, error_feedback=True)

# 指定 error-feedback 模式（可选）：none / legacy / ef21 / ef21_plus
bitscom.init(bitwidth=4, error_feedback_mode="ef21")
bitscom.init(bitwidth=4, error_feedback_mode="ef21_plus")
```

说明：

- `error_feedback=True` 等价于 `error_feedback_mode="legacy"`。
- `error_feedback_mode="legacy"` 与 `"ef21"` 仅作用于低比特 allreduce 第一段量化。
- `error_feedback_mode="ef21_plus"` 会额外在 allreduce 第二段量化（reduced shard）启用 error feedback。
- 同一进程内若以不同参数重复注册，会按设计抛出错误。

## 分层 All-Reduce 参数

在 Python 侧使用分层方案 A（传入 `local_group` + `inter_group`）时，
可通过参数控制 local-group 是否量化：

```python
group.all_reduce(
	tensor,
	local_group=local_group,
	inter_group=inter_group,
	chunk_size=1 << 20,
	local_quantize=False,  # 默认：local 全精度，inter 低比特
)
```

当 `local_quantize=True` 时，会保留原始行为：local-group 与 inter-group
阶段都走量化通信。

## 目录结构

- `cpp/`：C++ 后端与绑定
- `python/bitscom/`：Python 包
- `tests/`：单测、集成测试与性能测试
