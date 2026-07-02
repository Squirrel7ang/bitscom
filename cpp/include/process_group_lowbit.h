#pragma once

#include <torch/csrc/distributed/c10d/Backend.hpp>
#include <torch/csrc/distributed/c10d/ProcessGroupNCCL.hpp>
#include <torch/csrc/distributed/c10d/Store.hpp>
#include <torch/csrc/distributed/c10d/Types.hpp>
#include <torch/csrc/distributed/c10d/Work.hpp>
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime.h>
#include <nccl.h>

#include <condition_variable>
#include <chrono>
#include <deque>
#include <exception>
#include <functional>
#include <mutex>
#include <memory>
#include <optional>
#include <thread>
#include <string>
#include <tuple>
#include <unordered_map>
#include <vector>

namespace bitscom {

struct CudaEventHandle;
struct LowBitAllreduceTask;
class ProcessGroupLowBit;

// ============================================================
//  Sparse / Selective Quantization Strategy Interface
// ============================================================

/// Describes a contiguous segment of a flat tensor.
/// The segment spans elements [offset, offset+numel).
/// When `drop` is true the segment is not communicated at all and is filled
/// with zero after the allreduce (sparsification).
/// When `drop` is false:
///   - `quantize = true`  → compressed via the low-bit pipeline.
///   - `quantize = false` → communicated with a standard dense allreduce.
struct QuantizationSegment {
    int64_t offset   = 0;
    int64_t numel    = 0;
    bool    quantize = true;
    bool    drop     = false;
};

/// Abstract strategy that decides which parts of a tensor are important
/// (dense) vs compressible (quantized).
///
/// Usage:
///   1. Subclass and implement prepare() / partition().
///   2. Call pg->setPartitionStrategy(strategy) BEFORE the first allreduce.
///   3. The allreduce pipeline calls prepare() once per invocation, then
///      partition() for each tensor.
///
/// When `isActive()` returns false the existing full-quantization path is used.
class ITensorPartitionStrategy {
public:
    virtual ~ITensorPartitionStrategy() = default;

    /// Pre-processing hook. Called once before the tensors are processed.
    /// May modify tensors in-place, perform collective communication, etc.
    /// `comm` is the lowbit NCCL communicator (usable for priority exchange).
    /// `stream` is the launcher CUDA stream.
    /// Returns true to proceed with this strategy; false to fall back to
    /// full quantization (e.g. when there are too few elements to benefit).
    virtual bool prepare(
        std::vector<at::Tensor>& tensors,
        int rank,
        int world_size,
        ncclComm_t comm,
        c10::cuda::CUDAStream stream);

    /// Partition a flat (1-D) tensor into segments.
    /// The union of all segments MUST cover [0, tensor.numel()).
    /// Segments are processed in order; quantized segments go through the
    /// low-bit pipeline, dense segments go through NCCL allreduce.
    virtual std::vector<QuantizationSegment> partition(
        const at::Tensor& flat_tensor) = 0;

    /// Human-readable name for logging / debugging.
    virtual const char* name() const = 0;

    /// Whether this strategy is active. When false the default full-
    /// quantization path is used (equivalent to a single quantized segment).
    virtual bool isActive() const { return true; }
};

struct LowBitScheduledHandle {
    ProcessGroupLowBit* owner = nullptr;
    std::shared_ptr<LowBitAllreduceTask> task;
};

struct LowBitOptions {
    int bitwidth = 4;
    bool error_feedback = false;
    std::string error_feedback_mode = "auto";
    int block_size = 256;
    bool stage2_error_feedback = true;
    std::chrono::milliseconds timeout = std::chrono::milliseconds(600000);
};

enum class ErrorFeedbackMode {
    kDisabled = 0,
    kLegacy = 1,
    kEF21 = 2,
    kEF21Plus = 3,
};

// Work wrapper: 包装底层 NCCL Work，后续可加 unpack 回调
class WorkLowBit : public c10d::Work {
public:
    WorkLowBit(
        c10::intrusive_ptr<c10d::Work> nccl_work,
        std::function<bool()> post_hook = nullptr);

    bool isCompleted() override;
    bool isSuccess() const override;
    bool wait(std::chrono::milliseconds timeout = c10d::kUnsetTimeout) override;
    c10::intrusive_ptr<c10::ivalue::Future> getFuture() override;

private:
    bool runPostHook();

    c10::intrusive_ptr<c10d::Work> nccl_work_;
    std::function<bool()> post_hook_;
    bool post_hook_ran_ = false;
    bool post_hook_success_ = true;
};

class WorkBitscom : public c10d::Work {
public:
    WorkBitscom();
    explicit WorkBitscom(std::function<bool()> wait_fn);
    explicit WorkBitscom(std::function<bool(bool)> progress_fn);

    bool isCompleted() override;
    bool isSuccess() const override;
    bool wait(std::chrono::milliseconds timeout = c10d::kUnsetTimeout) override;
    c10::intrusive_ptr<c10::ivalue::Future> getFuture() override;

    void markCompleted(bool success = true);
    void markFailed(std::exception_ptr error);

private:
    bool runWaitFn();
    bool runProgressFn(bool block);

    mutable std::mutex mutex_;
    std::condition_variable cv_;
    bool completed_ = false;
    bool success_ = true;
    std::exception_ptr error_;
    c10::intrusive_ptr<c10::ivalue::Future> future_;
    std::function<bool()> wait_fn_;
    std::function<bool(bool)> progress_fn_;
    std::mutex progress_mutex_;
};

// ProcessGroupLowBit: 继承 c10d::Backend
// 内部持有 ProcessGroupNCCL 做实际通信
class ProcessGroupLowBit : public c10d::Backend {
public:
    ProcessGroupLowBit(
        const c10::intrusive_ptr<c10d::Store>& store,
        int rank,
        int size,
        LowBitOptions options = LowBitOptions());

    ~ProcessGroupLowBit() override;

    const std::string getBackendName() const override {
        return "lowbit";
    }

    // ---- 集合通信原语 ----

    c10::intrusive_ptr<c10d::Work> allreduce(
        std::vector<at::Tensor>& tensors,
        const c10d::AllreduceOptions& opts = c10d::AllreduceOptions()) override;

    c10::intrusive_ptr<c10d::Work> allgather(
        std::vector<std::vector<at::Tensor>>& output_tensors,
        std::vector<at::Tensor>& input_tensors,
        const c10d::AllgatherOptions& opts = c10d::AllgatherOptions()) override;

    c10::intrusive_ptr<c10d::Work> reduce_scatter(
        std::vector<at::Tensor>& output_tensors,
        std::vector<std::vector<at::Tensor>>& input_tensors,
        const c10d::ReduceScatterOptions& opts = c10d::ReduceScatterOptions()) override;

    c10::intrusive_ptr<c10d::Work> broadcast(
        std::vector<at::Tensor>& tensors,
        const c10d::BroadcastOptions& opts = c10d::BroadcastOptions()) override;

    c10::intrusive_ptr<c10d::Work> alltoall(
        std::vector<at::Tensor>& output_tensors,
        std::vector<at::Tensor>& input_tensors,
        const c10d::AllToAllOptions& opts = c10d::AllToAllOptions()) override;

    c10::intrusive_ptr<c10d::Work> alltoall_base(
        at::Tensor& output_tensor,
        at::Tensor& input_tensor,
        std::vector<int64_t>& output_split_sizes,
        std::vector<int64_t>& input_split_sizes,
        const c10d::AllToAllOptions& opts = c10d::AllToAllOptions()) override;

    bool progressLowBit(bool block = false);
    std::shared_ptr<LowBitScheduledHandle> scheduleLowBitAllreduce(at::Tensor tensor);
    bool launchScheduledLowBitPhase2(const std::shared_ptr<LowBitScheduledHandle>& handle);
    bool launchScheduledLowBitRestore(const std::shared_ptr<LowBitScheduledHandle>& handle);
    bool scheduledLowBitIsCompleted(
        const std::shared_ptr<LowBitScheduledHandle>& handle,
        bool block = false);
    bool scheduledLowBitWait(const std::shared_ptr<LowBitScheduledHandle>& handle);
    bool scheduledLowBitBlockCurrentStream(const std::shared_ptr<LowBitScheduledHandle>& handle);

    // ---- Sparse / Selective Quantization Strategy ----

    /// Set a tensor partition strategy for selective quantization.
    /// Pass nullptr to revert to full quantization (default).
    void setPartitionStrategy(
        std::shared_ptr<ITensorPartitionStrategy> strategy);
    std::shared_ptr<ITensorPartitionStrategy> getPartitionStrategy() const;

private:
    // 底层 NCCL process group
    c10::intrusive_ptr<c10d::ProcessGroupNCCL> nccl_pg_;
    c10::intrusive_ptr<c10d::Store> store_;
    ncclComm_t lowbit_comm_ = nullptr;

    LowBitOptions options_;

    ErrorFeedbackMode error_feedback_mode_ = ErrorFeedbackMode::kDisabled;

    // ---- pack/unpack 占位 ----
    // 将 float tensor 量化 + 打包为 uint8 buffer，返回 (packed, scale)
    std::tuple<at::Tensor, at::Tensor> pack(const at::Tensor& input);
    // 将 uint8 buffer 解包 + 反量化为 float tensor
    at::Tensor unpack(
        const at::Tensor& packed,
        int64_t numel,
        const at::Tensor& scale,
        c10::Device device,
        at::ScalarType out_dtype);

    bool shouldUseLowBitAllreduce(const c10d::AllreduceOptions& opts) const;
    c10::intrusive_ptr<c10d::Work> allreduceLowBit(
        std::vector<at::Tensor>& tensors,
        const c10d::AllreduceOptions& opts);
    /// Mixed dense+quantized allreduce (used when partition_strategy_ is active).
    c10::intrusive_ptr<c10d::Work> allreduceLowBitMixed(
        std::vector<at::Tensor>& tensors,
        const c10d::AllreduceOptions& opts);
    c10::intrusive_ptr<c10d::Work> launchLowBitAllreduceOrdered(
        std::vector<at::Tensor> tensors,
        const c10d::AllreduceOptions& opts,
        std::optional<int> device_index,
        std::shared_ptr<CudaEventHandle> ready_event);
    std::shared_ptr<LowBitAllreduceTask> launchLowBitPhase1Ordered(
        std::vector<at::Tensor> tensors,
        std::optional<int> device_index,
        std::shared_ptr<CudaEventHandle> ready_event,
        bool track_for_progress);
    bool progressLowBitTasks(
        const std::shared_ptr<LowBitAllreduceTask>& target,
        bool block);
    bool lowBitWorksReady(
        const std::vector<c10::intrusive_ptr<c10d::Work>>& works,
        bool block,
        const char* label);
    void launchLowBitPhase2(const std::shared_ptr<LowBitAllreduceTask>& task);
    void launchLowBitRestore(const std::shared_ptr<LowBitAllreduceTask>& task);
    bool runLowBitAllreduce(
        std::vector<at::Tensor> tensors,
        const c10d::AllreduceOptions& opts,
        std::optional<int> device_index,
        std::shared_ptr<CudaEventHandle> ready_event);
    void enqueueLowBitTask(std::function<void()> task);
    void launcherLoop();
    c10::cuda::CUDAStream getLauncherStream(int device_index);
    c10::cuda::CUDAStream getLowBitStream(int device_index, int slot);
    void initLowBitComm();

    bool useStage1ErrorFeedback() const;
    bool useStage2ErrorFeedback() const;

    struct ResidualShardKey {
        int64_t tensor_id = 0;
        int64_t shard_idx = 0;

        bool operator==(const ResidualShardKey& other) const {
            return tensor_id == other.tensor_id && shard_idx == other.shard_idx;
        }
    };

    struct ResidualShardKeyHash {
        size_t operator()(const ResidualShardKey& key) const {
            size_t h1 = std::hash<int64_t>{}(key.tensor_id);
            size_t h2 = std::hash<int64_t>{}(key.shard_idx);
            return h1 ^ (h2 + 0x9e3779b97f4a7c15ULL + (h1 << 6) + (h1 >> 2));
        }
    };

      // Error-feedback residual cache (keyed by TensorImpl address).
      std::mutex residual_mutex_;
      std::unordered_map<int64_t, at::Tensor> residual_cache_;
      std::unordered_map<ResidualShardKey, at::Tensor, ResidualShardKeyHash>
          residual_cache_stage2_;

      std::mutex launcher_mutex_;
      std::condition_variable launcher_cv_;
      std::deque<std::function<void()>> launcher_queue_;
      std::thread launcher_thread_;
      bool launcher_shutdown_ = false;
      std::unordered_map<int, std::unique_ptr<c10::cuda::CUDAStream>>
          launcher_streams_;
      std::mutex lowbit_progress_mutex_;
      std::deque<std::shared_ptr<LowBitAllreduceTask>> active_lowbit_tasks_;

      // Sparse / selective quantization strategy (nullptr → full quantization).
      std::shared_ptr<ITensorPartitionStrategy> partition_strategy_;
};

// 工厂函数，用于 Python 侧 register_backend
c10::intrusive_ptr<c10d::Backend> createProcessGroupLowBit(
    const c10::intrusive_ptr<c10d::Store>& store,
    int rank,
    int size,
    const std::chrono::milliseconds& timeout,
    int bitwidth,
    bool error_feedback,
    const std::string& error_feedback_mode,
    int block_size,
    bool stage2_error_feedback);

}  // namespace bitscom
