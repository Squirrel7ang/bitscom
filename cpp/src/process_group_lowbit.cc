// cpp/src/process_group_lowbit.cc
#include "process_group_lowbit.h"

#include <ATen/ATen.h>

#include <torch/csrc/distributed/c10d/ProcessGroupNCCL.hpp>
#include <iostream>

namespace bitscom {

// ==================== WorkLowBit ====================

WorkLowBit::WorkLowBit(
    c10::intrusive_ptr<c10d::Work> nccl_work,
    std::function<bool()> post_hook)
    : c10d::Work(),
      nccl_work_(std::move(nccl_work)),
      post_hook_(std::move(post_hook)) {}

bool WorkLowBit::isCompleted() {
    if (!nccl_work_->isCompleted()) {
        return false;
    }
    return runPostHook();
}

bool WorkLowBit::isSuccess() const {
    if (!nccl_work_->isSuccess()) {
        return false;
    }
    return post_hook_ran_ ? post_hook_success_ : true;
}

bool WorkLowBit::wait(std::chrono::milliseconds timeout) {
    bool success = nccl_work_->wait(timeout);
    if (!success) {
        return false;
    }
    return runPostHook();
}

c10::intrusive_ptr<c10::ivalue::Future> WorkLowBit::getFuture() {
    return nccl_work_->getFuture();
}

bool WorkLowBit::runPostHook() {
    if (post_hook_ran_) {
        return post_hook_success_;
    }
    if (post_hook_) {
        post_hook_success_ = post_hook_();
    }
    post_hook_ran_ = true;
    return post_hook_success_;
}

// ==================== ProcessGroupLowBit ====================

ProcessGroupLowBit::ProcessGroupLowBit(
    const c10::intrusive_ptr<c10d::Store>& store,
    int rank,
    int size,
    LowBitOptions options)
    : c10d::Backend(rank, size), options_(std::move(options)) {

    // 创建底层 NCCL ProcessGroup
    auto nccl_options = c10d::ProcessGroupNCCL::Options::create();
    nccl_options->timeout = options_.timeout;
    nccl_pg_ = c10::make_intrusive<c10d::ProcessGroupNCCL>(
        store, rank, size, std::move(nccl_options));

    std::cout << "[LowBit] ProcessGroupLowBit created: rank=" << rank
              << " size=" << size
              << " bitwidth=" << options_.bitwidth << std::endl;
}

// ---- pack/unpack 占位实现 ----

std::tuple<at::Tensor, at::Tensor> ProcessGroupLowBit::pack(const at::Tensor& input) {
    auto flat = input.contiguous().view(-1).to(at::kFloat);
    const int bitwidth = options_.bitwidth;
    TORCH_CHECK(
        bitwidth == 1 || bitwidth == 2 || bitwidth == 4 || bitwidth >= 8,
        "unsupported bitwidth for pack: ", bitwidth);

    if (bitwidth >= 8) {
        auto scale = at::ones({1}, flat.options());
        auto packed = flat.to(at::kHalf).view(at::kByte).contiguous();
        return std::make_tuple(packed, scale);
    }

    const int qmin = (bitwidth == 1) ? 0 : -(1 << (bitwidth - 1));
    const int qmax = (bitwidth == 1) ? 1 : ((1 << (bitwidth - 1)) - 1);

    auto max_abs_t = at::abs(flat).max();
    float max_abs = max_abs_t.item<float>();
    float scale_v = (max_abs == 0.0f) ? 1.0f : (max_abs / static_cast<float>(qmax));
    auto scale = at::full({1}, scale_v, flat.options());

    auto q = at::round(flat / scale_v).clamp(qmin, qmax).to(at::kInt);
    auto values = (q - qmin).to(at::kInt).contiguous().view(-1);

    const int per_byte = 8 / bitwidth;
    const int64_t numel = values.numel();
    const int64_t pad = (per_byte - (numel % per_byte)) % per_byte;
    if (pad > 0) {
        auto zeros = at::zeros({pad}, values.options());
        values = at::cat({values, zeros}, 0);
    }

    values = values.view({-1, per_byte});
    auto shifts = at::arange(0, per_byte, values.options()) * bitwidth;
    auto packed = at::sum(at::bitwise_left_shift(values, shifts), 1).to(at::kByte);
    return std::make_tuple(packed.contiguous(), scale);
}

at::Tensor ProcessGroupLowBit::unpack(
    const at::Tensor& packed,
    int64_t numel,
    const at::Tensor& scale,
    c10::Device device,
    at::ScalarType out_dtype) {
    const int bitwidth = options_.bitwidth;

    if (bitwidth >= 8) {
        auto half_view = packed.contiguous().view(at::kHalf).view({numel});
        return half_view.to(device, out_dtype);
    }

    const int qmin = (bitwidth == 1) ? 0 : -(1 << (bitwidth - 1));
    const int mask = (1 << bitwidth) - 1;
    const int per_byte = 8 / bitwidth;

    auto packed_i = packed.contiguous().view(-1).to(at::kInt);
    auto shifts = at::arange(0, per_byte, packed_i.options()) * bitwidth;
    auto expanded = at::bitwise_and(
        at::bitwise_right_shift(packed_i.unsqueeze(1), shifts),
        mask).reshape(-1);
    auto q = expanded.slice(0, 0, numel).to(at::kFloat) + static_cast<float>(qmin);

    float scale_v = scale.item<float>();
    auto out = (q * scale_v).to(device, out_dtype);
    return out;
}

bool ProcessGroupLowBit::shouldUseLowBitAllreduce(
    const c10d::AllreduceOptions& opts) const {
    return options_.bitwidth < 8 &&
        opts.reduceOp == c10d::ReduceOp::SUM &&
        getSize() > 1;
}

c10::intrusive_ptr<c10d::Work> ProcessGroupLowBit::allreduceLowBit(
    std::vector<at::Tensor>& tensors,
    const c10d::AllreduceOptions& opts) {

    if (tensors.empty()) {
        return nccl_pg_->allreduce(tensors, opts);
    }

    struct TensorPipelineState {
        at::Tensor original;
        at::Tensor flat;
        int64_t shard_len = 0;

        std::vector<at::Tensor> send_packed;
        std::vector<at::Tensor> recv_packed;
        std::vector<at::Tensor> send_scales;
        std::vector<at::Tensor> recv_scales;
    };

    auto state = std::make_shared<std::vector<TensorPipelineState>>();
    state->reserve(tensors.size());

    const int world_size = getSize();

    c10d::AllToAllOptions alltoall_opts;
    std::vector<c10::intrusive_ptr<c10d::Work>> phase1_works;

    for (auto& tensor : tensors) {
        TensorPipelineState s;
        s.original = tensor;
        s.flat = tensor.contiguous().view(-1);
        auto corrected = s.flat.to(at::kFloat);

        TORCH_CHECK(
            s.flat.numel() % world_size == 0,
            "lowbit allreduce requires tensor.numel() divisible by world_size, got numel=",
            s.flat.numel(), " world_size=", world_size);

        if (options_.error_feedback) {
            const int64_t key = static_cast<int64_t>(
                reinterpret_cast<uintptr_t>(s.original.unsafeGetTensorImpl()));
            at::Tensor residual;
            {
                std::lock_guard<std::mutex> lock(residual_mutex_);
                auto it = residual_cache_.find(key);
                if (it != residual_cache_.end()) {
                    residual = it->second;
                }
            }

            if (!residual.defined() ||
                residual.numel() != corrected.numel() ||
                residual.device() != corrected.device() ||
                residual.scalar_type() != at::kFloat) {
                residual = at::zeros_like(corrected);
            }
            corrected = corrected + residual;
        }

        s.shard_len = s.flat.numel() / world_size;
        auto shards = corrected.split(s.shard_len);

        s.send_packed.reserve(world_size);
        s.recv_packed.reserve(world_size);
        s.send_scales.reserve(world_size);
        s.recv_scales.reserve(world_size);

        std::vector<at::Tensor> sent_fp_shards;
        if (options_.error_feedback) {
            sent_fp_shards.reserve(world_size);
        }

        for (const auto& shard : shards) {
            at::Tensor packed, scale;
            std::tie(packed, scale) = pack(shard);

            if (options_.error_feedback) {
                auto approx = unpack(
                    packed,
                    s.shard_len,
                    scale,
                    corrected.device(),
                    at::kFloat);
                sent_fp_shards.push_back(approx);
            }

            s.send_packed.push_back(packed);
            s.recv_packed.push_back(at::empty_like(packed));
            s.send_scales.push_back(scale);
            s.recv_scales.push_back(at::empty_like(scale));
        }

        if (options_.error_feedback) {
            const int64_t key = static_cast<int64_t>(
                reinterpret_cast<uintptr_t>(s.original.unsafeGetTensorImpl()));
            auto sent_approx = at::cat(sent_fp_shards, 0);
            auto new_residual = (corrected - sent_approx).contiguous();
            std::lock_guard<std::mutex> lock(residual_mutex_);
            residual_cache_[key] = new_residual;
        }

        phase1_works.push_back(nccl_pg_->alltoall(s.recv_packed, s.send_packed, alltoall_opts));
        phase1_works.push_back(nccl_pg_->alltoall(s.recv_scales, s.send_scales, alltoall_opts));

        state->push_back(std::move(s));
    }

    auto anchor = phase1_works[0];
    auto post_hook = [this, state, phase1_works, world_size]() mutable -> bool {
        for (auto& w : phase1_works) {
            if (!w->wait()) {
                return false;
            }
        }

        c10d::AllgatherOptions allgather_opts;
        for (auto& s : *state) {
            auto local_sum = at::zeros({s.shard_len}, s.flat.options().dtype(at::kFloat));

            for (int src = 0; src < world_size; ++src) {
                auto fp = unpack(
                    s.recv_packed[src],
                    s.shard_len,
                    s.recv_scales[src],
                    s.flat.device(),
                    at::kFloat);
                local_sum.add_(fp);
            }

            at::Tensor reduced_packed, reduced_scale;
            std::tie(reduced_packed, reduced_scale) = pack(local_sum);

            std::vector<std::vector<at::Tensor>> gathered_packed(1);
            gathered_packed[0].reserve(world_size);
            for (int i = 0; i < world_size; ++i) {
                gathered_packed[0].push_back(at::empty_like(reduced_packed));
            }

            std::vector<at::Tensor> packed_input = {reduced_packed};
            auto wg_packed = nccl_pg_->allgather(gathered_packed, packed_input, allgather_opts);
            if (!wg_packed->wait()) {
                return false;
            }

            std::vector<std::vector<at::Tensor>> gathered_scales(1);
            gathered_scales[0].reserve(world_size);
            for (int i = 0; i < world_size; ++i) {
                gathered_scales[0].push_back(at::empty_like(reduced_scale));
            }

            std::vector<at::Tensor> scale_input = {reduced_scale};
            auto wg_scale = nccl_pg_->allgather(gathered_scales, scale_input, allgather_opts);
            if (!wg_scale->wait()) {
                return false;
            }

            std::vector<at::Tensor> out_shards;
            out_shards.reserve(world_size);
            for (int r = 0; r < world_size; ++r) {
                auto fp_shard = unpack(
                    gathered_packed[0][r],
                    s.shard_len,
                    gathered_scales[0][r],
                    s.flat.device(),
                    at::kFloat);
                out_shards.push_back(fp_shard);
            }

            auto restored = at::cat(out_shards, 0).view_as(s.original).to(s.original.scalar_type());
            s.original.copy_(restored);
        }
        return true;
    };

    return c10::make_intrusive<WorkLowBit>(std::move(anchor), std::move(post_hook));
}

// ---- 集合通信原语 ----

c10::intrusive_ptr<c10d::Work> ProcessGroupLowBit::allreduce(
    std::vector<at::Tensor>& tensors,
    const c10d::AllreduceOptions& opts) {

    if (shouldUseLowBitAllreduce(opts)) {
        return allreduceLowBit(tensors, opts);
    }

    // 非 <8 bit SUM 场景保持与 ProcessGroupNCCL 语义一致
    return nccl_pg_->allreduce(tensors, opts);
}

c10::intrusive_ptr<c10d::Work> ProcessGroupLowBit::broadcast(
    std::vector<at::Tensor>& tensors,
    const c10d::BroadcastOptions& opts) {
    // broadcast 通常不需要压缩，直接转发到 NCCL
    return nccl_pg_->broadcast(tensors, opts);
}

c10::intrusive_ptr<c10d::Work> ProcessGroupLowBit::allgather(
    std::vector<std::vector<at::Tensor>>& output_tensors,
    std::vector<at::Tensor>& input_tensors,
    const c10d::AllgatherOptions& opts) {

    // TODO: 对 input 做 pack，对 output 做 unpack
    // 当前占位：直接转发到 NCCL
    return nccl_pg_->allgather(output_tensors, input_tensors, opts);
}

c10::intrusive_ptr<c10d::Work> ProcessGroupLowBit::reduce_scatter(
    std::vector<at::Tensor>& output_tensors,
    std::vector<std::vector<at::Tensor>>& input_tensors,
    const c10d::ReduceScatterOptions& opts) {

    // TODO: 对 input 做 pack，对 output 做 unpack
    // 当前占位：直接转发到 NCCL
    return nccl_pg_->reduce_scatter(output_tensors, input_tensors, opts);
}

c10::intrusive_ptr<c10d::Work> ProcessGroupLowBit::alltoall(
    std::vector<at::Tensor>& output_tensors,
    std::vector<at::Tensor>& input_tensors,
    const c10d::AllToAllOptions& opts) {

    // 复用 NCCL 的 alltoall 实现，供 Python dist.all_to_all 直接调用。
    return nccl_pg_->alltoall(output_tensors, input_tensors, opts);
}

c10::intrusive_ptr<c10d::Work> ProcessGroupLowBit::alltoall_base(
    at::Tensor& output_tensor,
    at::Tensor& input_tensor,
    std::vector<int64_t>& output_split_sizes,
    std::vector<int64_t>& input_split_sizes,
    const c10d::AllToAllOptions& opts) {

    // 复用 NCCL 的 alltoall_base 实现，兼容 all_to_all_single 路径。
    return nccl_pg_->alltoall_base(
        output_tensor,
        input_tensor,
        output_split_sizes,
        input_split_sizes,
        opts);
}

// ---- 工厂函数 ----

c10::intrusive_ptr<c10d::Backend> createProcessGroupLowBit(
    const c10::intrusive_ptr<c10d::Store>& store,
    int rank,
    int size,
    const std::chrono::milliseconds& timeout,
    int bitwidth,
    bool error_feedback) {

    LowBitOptions opts;
    opts.timeout = timeout;
    opts.bitwidth = bitwidth;
    opts.error_feedback = error_feedback;
    return c10::make_intrusive<ProcessGroupLowBit>(
        store, rank, size, std::move(opts));
}

}  // namespace bitscom
