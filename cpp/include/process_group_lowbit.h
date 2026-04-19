#pragma once

#include <torch/csrc/distributed/c10d/Backend.hpp>
#include <torch/csrc/distributed/c10d/ProcessGroupNCCL.hpp>
#include <torch/csrc/distributed/c10d/Store.hpp>
#include <torch/csrc/distributed/c10d/Types.hpp>
#include <torch/csrc/distributed/c10d/Work.hpp>

#include <chrono>
#include <functional>
#include <mutex>
#include <memory>
#include <string>
#include <tuple>
#include <unordered_map>
#include <vector>

namespace bitscom {

struct LowBitOptions {
    int bitwidth = 4;
    bool error_feedback = false;
    std::chrono::milliseconds timeout = std::chrono::milliseconds(600000);
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

// ProcessGroupLowBit: 继承 c10d::Backend
// 内部持有 ProcessGroupNCCL 做实际通信
class ProcessGroupLowBit : public c10d::Backend {
public:
    ProcessGroupLowBit(
        const c10::intrusive_ptr<c10d::Store>& store,
        int rank,
        int size,
        LowBitOptions options = LowBitOptions());

    ~ProcessGroupLowBit() override = default;

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

private:
    // 底层 NCCL process group
    c10::intrusive_ptr<c10d::ProcessGroupNCCL> nccl_pg_;

    LowBitOptions options_;

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

      // Error-feedback residual cache (keyed by TensorImpl address).
      std::mutex residual_mutex_;
      std::unordered_map<int64_t, at::Tensor> residual_cache_;
};

// 工厂函数，用于 Python 侧 register_backend
c10::intrusive_ptr<c10d::Backend> createProcessGroupLowBit(
    const c10::intrusive_ptr<c10d::Store>& store,
    int rank,
    int size,
      const std::chrono::milliseconds& timeout,
      int bitwidth,
      bool error_feedback);

}  // namespace bitscom
