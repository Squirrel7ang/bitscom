// cpp/src/process_group_lowbit.cc
#include "process_group_lowbit.h"

#include <ATen/ATen.h>
#include <c10/cuda/CUDAGuard.h>
#include <ATen/core/jit_type.h>

#include <torch/csrc/distributed/c10d/ProcessGroupNCCL.hpp>
#include <algorithm>
#include <cctype>
#include <chrono>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <sys/stat.h>
#include <unistd.h>

namespace bitscom {

namespace {

std::string toLower(std::string value) {
    std::transform(value.begin(), value.end(), value.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });
    return value;
}

ErrorFeedbackMode parseErrorFeedbackMode(const LowBitOptions& options) {
    std::string mode = options.error_feedback_mode;
    if (mode.empty() || mode == "auto") {
        mode = options.error_feedback ? "legacy" : "none";
    }
    mode = toLower(mode);

    if (mode == "none" || mode == "off" || mode == "disabled") {
        return ErrorFeedbackMode::kDisabled;
    }
    if (mode == "legacy" || mode == "ef") {
        return ErrorFeedbackMode::kLegacy;
    }
    if (mode == "ef21") {
        return ErrorFeedbackMode::kEF21;
    }
    if (mode == "ef21+" || mode == "ef21_plus") {
        return ErrorFeedbackMode::kEF21Plus;
    }

    TORCH_CHECK(false, "unsupported error_feedback_mode: ", options.error_feedback_mode);
}

const char* errorFeedbackModeName(ErrorFeedbackMode mode) {
    switch (mode) {
        case ErrorFeedbackMode::kDisabled:
            return "none";
        case ErrorFeedbackMode::kLegacy:
            return "legacy";
        case ErrorFeedbackMode::kEF21:
            return "ef21";
        case ErrorFeedbackMode::kEF21Plus:
            return "ef21_plus";
    }
    return "none";
}

double nowSeconds() {
    using clock = std::chrono::system_clock;
    auto now = clock::now().time_since_epoch();
    return std::chrono::duration<double>(now).count();
}

void ensureTimingDir() {
    mkdir("debug_logs", 0755);
    mkdir("debug_logs/timing", 0755);
}

void lowbitBackendTiming(const void* pg, int rank, const std::string& message) {
    ensureTimingDir();
    const auto pid = static_cast<long long>(getpid());
    std::ostringstream path;
    path << "debug_logs/timing/lowbit_backend_pid" << pid << "_rank" << rank << ".log";
    std::ofstream out(path.str(), std::ios::app);
    out << "[lowbit-backend-timing pid=" << pid
        << " rank=" << rank
        << " pg=" << pg
        << " tid=" << std::this_thread::get_id()
        << " t=" << std::fixed << std::setprecision(6) << nowSeconds()
        << "] " << message << "\n";
}

void checkCuda(cudaError_t err, const char* what) {
    TORCH_CHECK(
        err == cudaSuccess,
        what,
        " failed: ",
        cudaGetErrorString(err));
}

}  // namespace

struct CudaEventHandle {
    cudaEvent_t event = nullptr;
    int device_index = -1;

    explicit CudaEventHandle(int device) : device_index(device) {
        c10::cuda::CUDAGuard device_guard(device_index);
        checkCuda(
            cudaEventCreateWithFlags(&event, cudaEventDisableTiming),
            "cudaEventCreateWithFlags");
    }

    ~CudaEventHandle() {
        if (event != nullptr) {
            c10::cuda::CUDAGuard device_guard(device_index);
            cudaEventDestroy(event);
        }
    }

    CudaEventHandle(const CudaEventHandle&) = delete;
    CudaEventHandle& operator=(const CudaEventHandle&) = delete;
};

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

// ==================== WorkBitscom ====================

WorkBitscom::WorkBitscom()
    : c10d::Work(),
      future_(c10::make_intrusive<c10::ivalue::Future>(c10::BoolType::get())) {}

WorkBitscom::WorkBitscom(std::function<bool()> wait_fn)
    : c10d::Work(),
      future_(c10::make_intrusive<c10::ivalue::Future>(c10::BoolType::get())),
      wait_fn_(std::move(wait_fn)) {}

bool WorkBitscom::isCompleted() {
    std::lock_guard<std::mutex> lock(mutex_);
    return completed_;
}

bool WorkBitscom::isSuccess() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return completed_ && success_ && error_ == nullptr;
}

bool WorkBitscom::wait(std::chrono::milliseconds timeout) {
    if (!runWaitFn()) {
        return false;
    }

    std::unique_lock<std::mutex> lock(mutex_);
    if (timeout == c10d::kUnsetTimeout) {
        cv_.wait(lock, [this]() { return completed_; });
    } else {
        if (!cv_.wait_for(lock, timeout, [this]() { return completed_; })) {
            return false;
        }
    }
    if (error_) {
        std::rethrow_exception(error_);
    }
    return success_;
}

c10::intrusive_ptr<c10::ivalue::Future> WorkBitscom::getFuture() {
    return future_;
}

void WorkBitscom::markCompleted(bool success) {
    {
        std::lock_guard<std::mutex> lock(mutex_);
        if (completed_) {
            return;
        }
        success_ = success;
        completed_ = true;
    }
    future_->markCompleted(c10::IValue(success));
    cv_.notify_all();
}

void WorkBitscom::markFailed(std::exception_ptr error) {
    {
        std::lock_guard<std::mutex> lock(mutex_);
        if (completed_) {
            return;
        }
        success_ = false;
        error_ = std::move(error);
        completed_ = true;
    }
    future_->markCompleted(c10::IValue(false));
    cv_.notify_all();
}

bool WorkBitscom::runWaitFn() {
    std::function<bool()> wait_fn;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        if (completed_) {
            return success_;
        }
        wait_fn = std::move(wait_fn_);
    }

    if (!wait_fn) {
        return true;
    }

    try {
        bool success = wait_fn();
        markCompleted(success);
        return success;
    } catch (...) {
        markFailed(std::current_exception());
        std::rethrow_exception(std::current_exception());
    }
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

    error_feedback_mode_ = parseErrorFeedbackMode(options_);
    TORCH_CHECK(options_.block_size > 0, "block_size must be > 0, got ", options_.block_size);

    std::cout << "[LowBit] ProcessGroupLowBit created: rank=" << rank
              << " size=" << size
              << " bitwidth=" << options_.bitwidth
              << " error_feedback=" << errorFeedbackModeName(error_feedback_mode_)
              << std::endl;
    lowbitBackendTiming(
        this,
        getRank(),
        "ProcessGroupLowBit created size=" + std::to_string(size) +
            " bitwidth=" + std::to_string(options_.bitwidth) +
            " error_feedback=" + errorFeedbackModeName(error_feedback_mode_));

    // NCCL collectives are launched from the caller's host thread to preserve
    // cross-communicator launch ordering with the training runtime.
}

ProcessGroupLowBit::~ProcessGroupLowBit() {
    lowbitBackendTiming(this, getRank(), "ProcessGroupLowBit destructor enter");
    {
        std::lock_guard<std::mutex> lock(launcher_mutex_);
        launcher_shutdown_ = true;
    }
    launcher_cv_.notify_all();
    if (launcher_thread_.joinable()) {
        launcher_thread_.join();
    }
    lowbitBackendTiming(this, getRank(), "ProcessGroupLowBit destructor exit");
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
    const int64_t numel = flat.numel();
    if (numel == 0) {
        auto scale = at::empty({0}, flat.options().dtype(at::kHalf));
        auto packed = at::empty({0}, flat.options().dtype(at::kByte));
        return std::make_tuple(packed, scale);
    }

    const int64_t block_size = options_.block_size;
    const int64_t num_blocks = (numel + block_size - 1) / block_size;
    const int64_t padded = num_blocks * block_size;
    if (padded != numel) {
        auto zeros = at::zeros({padded - numel}, flat.options());
        flat = at::cat({flat, zeros}, 0);
    }

    auto blocks = flat.view({num_blocks, block_size});
    auto abs_blocks = at::abs(blocks);
    auto max_abs = std::get<0>(abs_blocks.max(1));
    auto scale = max_abs / static_cast<float>(qmax);
    scale = at::where(max_abs > 0, scale, at::ones_like(scale));
    auto scale_half = scale.to(at::kHalf);

    auto scale_f = scale_half.to(at::kFloat);
    auto normalized = abs_blocks / scale_f.unsqueeze(1);
    auto mag = at::round(normalized);
    auto signed_vals = (bitwidth == 1) ? mag : mag * at::sign(blocks);
    auto q = signed_vals.clamp(qmin, qmax).to(at::kInt);
    auto values = (q.view({-1}).slice(0, 0, numel) - qmin)
                      .to(at::kInt)
                      .contiguous()
                      .view(-1);

    const int per_byte = 8 / bitwidth;
    const int64_t packed_numel = values.numel();
    const int64_t pad = (per_byte - (packed_numel % per_byte)) % per_byte;
    if (pad > 0) {
        auto zeros = at::zeros({pad}, values.options());
        values = at::cat({values, zeros}, 0);
    }

    values = values.view({-1, per_byte});
    auto shifts = at::arange(0, per_byte, values.options()) * bitwidth;
    auto packed = at::sum(at::bitwise_left_shift(values, shifts), 1).to(at::kByte);
    return std::make_tuple(packed.contiguous(), scale_half);
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

    if (numel == 0) {
        return q.to(device, out_dtype);
    }

    const int64_t block_size = options_.block_size;
    const int64_t num_blocks = scale.numel();
    const int64_t expected_blocks = (numel + block_size - 1) / block_size;
    TORCH_CHECK(
        num_blocks == expected_blocks,
        "scale blocks mismatch: got ", num_blocks, " expected ", expected_blocks);

    const int64_t padded = num_blocks * block_size;
    if (padded != numel) {
        auto zeros = at::zeros({padded - numel}, q.options());
        q = at::cat({q, zeros}, 0);
    }

    auto q_blocks = q.view({num_blocks, block_size});
    auto scale_f = scale.to(at::kFloat).view({num_blocks, 1});
    auto out = (q_blocks * scale_f).view({-1}).slice(0, 0, numel).to(device, out_dtype);
    return out;
}

bool ProcessGroupLowBit::shouldUseLowBitAllreduce(
    const c10d::AllreduceOptions& opts) const {
    return options_.bitwidth < 8 &&
        opts.reduceOp == c10d::ReduceOp::SUM &&
        getSize() > 1;
}

bool ProcessGroupLowBit::useStage1ErrorFeedback() const {
    return error_feedback_mode_ != ErrorFeedbackMode::kDisabled;
}

bool ProcessGroupLowBit::useStage2ErrorFeedback() const {
    return options_.stage2_error_feedback &&
        error_feedback_mode_ == ErrorFeedbackMode::kEF21Plus;
}

c10::intrusive_ptr<c10d::Work> ProcessGroupLowBit::allreduceLowBit(
    std::vector<at::Tensor>& tensors,
    const c10d::AllreduceOptions& opts) {
    std::vector<at::Tensor> tensors_copy = tensors;
    std::optional<int> device_index;
    std::shared_ptr<CudaEventHandle> ready_event;
    if (!tensors.empty() && tensors[0].defined() && tensors[0].is_cuda()) {
        device_index = tensors[0].device().index();
        c10::cuda::CUDAGuard device_guard(*device_index);
        auto producer_stream = c10::cuda::getCurrentCUDAStream(*device_index);
        ready_event = std::make_shared<CudaEventHandle>(*device_index);
        checkCuda(
            cudaEventRecord(ready_event->event, producer_stream.stream()),
            "cudaEventRecord");
        lowbitBackendTiming(
            this,
            getRank(),
            "producer ready event recorded device=" + std::to_string(*device_index));
    }
    lowbitBackendTiming(
        this,
        getRank(),
        "allreduceLowBit ordered launch tensors=" + std::to_string(tensors.size()) +
            (tensors.empty() ? "" : " numel=" + std::to_string(tensors[0].numel())) +
            (device_index.has_value() ? " device=" + std::to_string(*device_index) : ""));
    try {
        return launchLowBitAllreduceOrdered(
            std::move(tensors_copy),
            opts,
            device_index,
            std::move(ready_event));
    } catch (const std::exception& e) {
        lowbitBackendTiming(
            this,
            getRank(),
            std::string("allreduceLowBit ordered launch failed exception=") + e.what());
        auto work = c10::make_intrusive<WorkBitscom>();
        work->markFailed(std::current_exception());
        return work;
    } catch (...) {
        lowbitBackendTiming(this, getRank(), "allreduceLowBit ordered launch failed unknown exception");
        auto work = c10::make_intrusive<WorkBitscom>();
        work->markFailed(std::current_exception());
        return work;
    }
}

c10::intrusive_ptr<c10d::Work> ProcessGroupLowBit::launchLowBitAllreduceOrdered(
    std::vector<at::Tensor> tensors,
    const c10d::AllreduceOptions& opts,
    std::optional<int> device_index,
    std::shared_ptr<CudaEventHandle> ready_event) {
    std::optional<c10::cuda::CUDAStream> launcher_stream;
    if (device_index.has_value()) {
        c10::cuda::CUDAGuard device_guard(*device_index);
        launcher_stream = getLauncherStream(*device_index);
        checkCuda(
            cudaStreamWaitEvent(launcher_stream->stream(), ready_event->event, 0),
            "cudaStreamWaitEvent");
        lowbitBackendTiming(
            this,
            getRank(),
            "ordered launcher stream wait_event queued device=" + std::to_string(*device_index));
    }
    c10::cuda::OptionalCUDAStreamGuard stream_guard(launcher_stream);
    lowbitBackendTiming(
        this,
        getRank(),
        "ordered launch enter tensors=" + std::to_string(tensors.size()) +
            (device_index.has_value() ? " device=" + std::to_string(*device_index) : ""));

    if (tensors.empty()) {
        return nccl_pg_->allreduce(tensors, opts);
    }

    struct TensorPipelineState {
        at::Tensor original;
        at::Tensor flat;
        int64_t tensor_id = 0;
        int64_t shard_len = 0;

        std::vector<at::Tensor> send_packed;
        std::vector<at::Tensor> recv_packed;
        std::vector<at::Tensor> send_scales;
        std::vector<at::Tensor> recv_scales;
    };

    auto state = std::make_shared<std::vector<TensorPipelineState>>();
    state->reserve(tensors.size());

    const int world_size = getSize();
    const int rank = getRank();

    const bool stage1_ef = useStage1ErrorFeedback();
    const bool stage2_ef = useStage2ErrorFeedback();

    c10d::AllToAllOptions alltoall_opts;
    std::vector<c10::intrusive_ptr<c10d::Work>> phase1_works;

    for (auto& tensor : tensors) {
        TensorPipelineState s;
        s.original = tensor;
        s.flat = tensor.contiguous().view(-1);
        lowbitBackendTiming(
            this,
            getRank(),
            "ordered local prep flat done tensor_numel=" + std::to_string(s.flat.numel()));
        s.tensor_id = static_cast<int64_t>(
            reinterpret_cast<uintptr_t>(s.original.unsafeGetTensorImpl()));
        auto corrected = s.flat.to(at::kFloat);
        lowbitBackendTiming(
            this,
            getRank(),
            "ordered local prep to_float launched tensor_numel=" + std::to_string(s.flat.numel()));

        TORCH_CHECK(
            s.flat.numel() % world_size == 0,
            "lowbit allreduce requires tensor.numel() divisible by world_size, got numel=",
            s.flat.numel(), " world_size=", world_size);

        if (stage1_ef) {
            const int64_t key = s.tensor_id;
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
        if (stage1_ef) {
            sent_fp_shards.reserve(world_size);
        }

        for (const auto& shard : shards) {
            at::Tensor packed, scale;
            std::tie(packed, scale) = pack(shard);
            lowbitBackendTiming(
                this,
                getRank(),
                "ordered local pack shard launched shard_numel=" + std::to_string(shard.numel()));

            if (stage1_ef) {
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

        if (stage1_ef) {
            const int64_t key = s.tensor_id;
            auto sent_approx = at::cat(sent_fp_shards, 0);
            auto new_residual = (corrected - sent_approx).contiguous();
            std::lock_guard<std::mutex> lock(residual_mutex_);
            residual_cache_[key] = new_residual;
        }

        lowbitBackendTiming(
            this,
            getRank(),
            "ordered phase1 packed alltoall enter tensor_numel=" + std::to_string(s.flat.numel()));
        phase1_works.push_back(nccl_pg_->alltoall(s.recv_packed, s.send_packed, alltoall_opts));
        lowbitBackendTiming(
            this,
            getRank(),
            "ordered phase1 packed alltoall returned tensor_numel=" + std::to_string(s.flat.numel()));
        lowbitBackendTiming(
            this,
            getRank(),
            "ordered phase1 scales alltoall enter tensor_numel=" + std::to_string(s.flat.numel()));
        phase1_works.push_back(nccl_pg_->alltoall(s.recv_scales, s.send_scales, alltoall_opts));
        lowbitBackendTiming(
            this,
            getRank(),
            "ordered phase1 scales alltoall returned tensor_numel=" + std::to_string(s.flat.numel()));
        lowbitBackendTiming(
            this,
            getRank(),
            "ordered phase1 alltoall launched tensor_numel=" + std::to_string(s.flat.numel()));

        state->push_back(std::move(s));
    }

    auto wait_fn = [this, state, phase1_works, world_size, rank, stage2_ef, device_index]() mutable -> bool {
        std::optional<c10::cuda::CUDAStream> stream;
        if (device_index.has_value()) {
            c10::cuda::CUDAGuard device_guard(*device_index);
            stream = getLauncherStream(*device_index);
        }
        c10::cuda::OptionalCUDAStreamGuard stream_guard(stream);

        lowbitBackendTiming(this, getRank(), "ordered phase1 wait enter works=" + std::to_string(phase1_works.size()));
        for (auto& w : phase1_works) {
            if (!w->wait()) {
                lowbitBackendTiming(this, getRank(), "ordered phase1 wait failed");
                return false;
            }
        }
        lowbitBackendTiming(this, getRank(), "ordered phase1 wait done");

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

            at::Tensor reduce_input = local_sum;
            if (stage2_ef) {
                ResidualShardKey key{s.tensor_id, static_cast<int64_t>(rank)};
                at::Tensor residual;
                {
                    std::lock_guard<std::mutex> lock(residual_mutex_);
                    auto it = residual_cache_stage2_.find(key);
                    if (it != residual_cache_stage2_.end()) {
                        residual = it->second;
                    }
                }

                if (!residual.defined() ||
                    residual.numel() != local_sum.numel() ||
                    residual.device() != local_sum.device() ||
                    residual.scalar_type() != at::kFloat) {
                    residual = at::zeros_like(local_sum);
                }
                reduce_input = local_sum + residual;
            }

            at::Tensor reduced_packed, reduced_scale;
            std::tie(reduced_packed, reduced_scale) = pack(reduce_input);

            if (stage2_ef) {
                auto approx = unpack(
                    reduced_packed,
                    s.shard_len,
                    reduced_scale,
                    s.flat.device(),
                    at::kFloat);
                auto new_residual = (reduce_input - approx).contiguous();
                ResidualShardKey key{s.tensor_id, static_cast<int64_t>(rank)};
                std::lock_guard<std::mutex> lock(residual_mutex_);
                residual_cache_stage2_[key] = new_residual;
            }

            std::vector<std::vector<at::Tensor>> gathered_packed(1);
            gathered_packed[0].reserve(world_size);
            for (int i = 0; i < world_size; ++i) {
                gathered_packed[0].push_back(at::empty_like(reduced_packed));
            }

            std::vector<at::Tensor> packed_input = {reduced_packed};
            auto wg_packed = nccl_pg_->allgather(gathered_packed, packed_input, allgather_opts);
            lowbitBackendTiming(this, getRank(), "ordered allgather packed launched");
            if (!wg_packed->wait()) {
                lowbitBackendTiming(this, getRank(), "ordered allgather packed wait failed");
                return false;
            }
            lowbitBackendTiming(this, getRank(), "ordered allgather packed wait done");

            std::vector<std::vector<at::Tensor>> gathered_scales(1);
            gathered_scales[0].reserve(world_size);
            for (int i = 0; i < world_size; ++i) {
                gathered_scales[0].push_back(at::empty_like(reduced_scale));
            }

            std::vector<at::Tensor> scale_input = {reduced_scale};
            auto wg_scale = nccl_pg_->allgather(gathered_scales, scale_input, allgather_opts);
            lowbitBackendTiming(this, getRank(), "ordered allgather scales launched");
            if (!wg_scale->wait()) {
                lowbitBackendTiming(this, getRank(), "ordered allgather scales wait failed");
                return false;
            }
            lowbitBackendTiming(this, getRank(), "ordered allgather scales wait done");

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
            lowbitBackendTiming(this, getRank(), "ordered restore done tensor_numel=" + std::to_string(s.flat.numel()));
        }

        if (stream.has_value()) {
            checkCuda(cudaStreamSynchronize(stream->stream()), "cudaStreamSynchronize");
            lowbitBackendTiming(this, getRank(), "ordered launcher stream synchronized");
        }
        return true;
    };

    return c10::make_intrusive<WorkBitscom>(std::move(wait_fn));
}

bool ProcessGroupLowBit::runLowBitAllreduce(
    std::vector<at::Tensor> tensors,
    const c10d::AllreduceOptions& opts,
    std::optional<int> device_index,
    std::shared_ptr<CudaEventHandle> ready_event) {
    std::optional<c10::cuda::CUDAStream> launcher_stream;
    if (device_index.has_value()) {
        c10::cuda::CUDAGuard device_guard(*device_index);
        launcher_stream = getLauncherStream(*device_index);
        checkCuda(
            cudaStreamWaitEvent(launcher_stream->stream(), ready_event->event, 0),
            "cudaStreamWaitEvent");
        lowbitBackendTiming(
            this,
            getRank(),
            "launcher stream wait_event queued device=" + std::to_string(*device_index));
    }
    c10::cuda::OptionalCUDAStreamGuard stream_guard(launcher_stream);
    lowbitBackendTiming(
        this,
        getRank(),
        "runLowBitAllreduce enter tensors=" + std::to_string(tensors.size()) +
            (device_index.has_value() ? " device=" + std::to_string(*device_index) : ""));

    if (tensors.empty()) {
        auto work = nccl_pg_->allreduce(tensors, opts);
        bool success = work->wait();
        lowbitBackendTiming(
            this,
            getRank(),
            std::string("runLowBitAllreduce empty done success=") +
                (success ? "true" : "false"));
        return success;
    }

    struct TensorPipelineState {
        at::Tensor original;
        at::Tensor flat;
        int64_t tensor_id = 0;
        int64_t shard_len = 0;

        std::vector<at::Tensor> send_packed;
        std::vector<at::Tensor> recv_packed;
        std::vector<at::Tensor> send_scales;
        std::vector<at::Tensor> recv_scales;
    };

    auto state = std::make_shared<std::vector<TensorPipelineState>>();
    state->reserve(tensors.size());

    const int world_size = getSize();
    const int rank = getRank();

    const bool stage1_ef = useStage1ErrorFeedback();
    const bool stage2_ef = useStage2ErrorFeedback();

    c10d::AllToAllOptions alltoall_opts;
    std::vector<c10::intrusive_ptr<c10d::Work>> phase1_works;

    for (auto& tensor : tensors) {
        TensorPipelineState s;
        s.original = tensor;
        s.flat = tensor.contiguous().view(-1);
        lowbitBackendTiming(
            this,
            getRank(),
            "local prep flat done tensor_numel=" + std::to_string(s.flat.numel()));
        s.tensor_id = static_cast<int64_t>(
            reinterpret_cast<uintptr_t>(s.original.unsafeGetTensorImpl()));
        auto corrected = s.flat.to(at::kFloat);
        lowbitBackendTiming(
            this,
            getRank(),
            "local prep to_float launched tensor_numel=" + std::to_string(s.flat.numel()));

        TORCH_CHECK(
            s.flat.numel() % world_size == 0,
            "lowbit allreduce requires tensor.numel() divisible by world_size, got numel=",
            s.flat.numel(), " world_size=", world_size);

        if (stage1_ef) {
            const int64_t key = s.tensor_id;
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
        if (stage1_ef) {
            sent_fp_shards.reserve(world_size);
        }

        for (const auto& shard : shards) {
            at::Tensor packed, scale;
            std::tie(packed, scale) = pack(shard);
            lowbitBackendTiming(
                this,
                getRank(),
                "local pack shard launched shard_numel=" + std::to_string(shard.numel()));

            if (stage1_ef) {
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

        if (stage1_ef) {
            const int64_t key = s.tensor_id;
            auto sent_approx = at::cat(sent_fp_shards, 0);
            auto new_residual = (corrected - sent_approx).contiguous();
            std::lock_guard<std::mutex> lock(residual_mutex_);
            residual_cache_[key] = new_residual;
        }

        lowbitBackendTiming(
            this,
            getRank(),
            "phase1 packed alltoall enter tensor_numel=" + std::to_string(s.flat.numel()));
        phase1_works.push_back(nccl_pg_->alltoall(s.recv_packed, s.send_packed, alltoall_opts));
        lowbitBackendTiming(
            this,
            getRank(),
            "phase1 packed alltoall returned tensor_numel=" + std::to_string(s.flat.numel()));
        lowbitBackendTiming(
            this,
            getRank(),
            "phase1 scales alltoall enter tensor_numel=" + std::to_string(s.flat.numel()));
        phase1_works.push_back(nccl_pg_->alltoall(s.recv_scales, s.send_scales, alltoall_opts));
        lowbitBackendTiming(
            this,
            getRank(),
            "phase1 scales alltoall returned tensor_numel=" + std::to_string(s.flat.numel()));
        lowbitBackendTiming(
            this,
            getRank(),
            "phase1 alltoall launched tensor_numel=" + std::to_string(s.flat.numel()));

        state->push_back(std::move(s));
    }

    auto post_hook = [this, state, phase1_works, world_size, rank, stage2_ef]() mutable -> bool {
        lowbitBackendTiming(this, getRank(), "phase1 wait enter works=" + std::to_string(phase1_works.size()));
        for (auto& w : phase1_works) {
            if (!w->wait()) {
                lowbitBackendTiming(this, getRank(), "phase1 wait failed");
                return false;
            }
        }
        lowbitBackendTiming(this, getRank(), "phase1 wait done");

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

            at::Tensor reduce_input = local_sum;
            if (stage2_ef) {
                // EF21+: keep a residual on the reduced shard quantization step.
                ResidualShardKey key{s.tensor_id, static_cast<int64_t>(rank)};
                at::Tensor residual;
                {
                    std::lock_guard<std::mutex> lock(residual_mutex_);
                    auto it = residual_cache_stage2_.find(key);
                    if (it != residual_cache_stage2_.end()) {
                        residual = it->second;
                    }
                }

                if (!residual.defined() ||
                    residual.numel() != local_sum.numel() ||
                    residual.device() != local_sum.device() ||
                    residual.scalar_type() != at::kFloat) {
                    residual = at::zeros_like(local_sum);
                }
                reduce_input = local_sum + residual;
            }

            at::Tensor reduced_packed, reduced_scale;
            std::tie(reduced_packed, reduced_scale) = pack(reduce_input);

            if (stage2_ef) {
                auto approx = unpack(
                    reduced_packed,
                    s.shard_len,
                    reduced_scale,
                    s.flat.device(),
                    at::kFloat);
                auto new_residual = (reduce_input - approx).contiguous();
                ResidualShardKey key{s.tensor_id, static_cast<int64_t>(rank)};
                std::lock_guard<std::mutex> lock(residual_mutex_);
                residual_cache_stage2_[key] = new_residual;
            }

            std::vector<std::vector<at::Tensor>> gathered_packed(1);
            gathered_packed[0].reserve(world_size);
            for (int i = 0; i < world_size; ++i) {
                gathered_packed[0].push_back(at::empty_like(reduced_packed));
            }

            std::vector<at::Tensor> packed_input = {reduced_packed};
            auto wg_packed = nccl_pg_->allgather(gathered_packed, packed_input, allgather_opts);
            lowbitBackendTiming(this, getRank(), "allgather packed launched");
            if (!wg_packed->wait()) {
                lowbitBackendTiming(this, getRank(), "allgather packed wait failed");
                return false;
            }
            lowbitBackendTiming(this, getRank(), "allgather packed wait done");

            std::vector<std::vector<at::Tensor>> gathered_scales(1);
            gathered_scales[0].reserve(world_size);
            for (int i = 0; i < world_size; ++i) {
                gathered_scales[0].push_back(at::empty_like(reduced_scale));
            }

            std::vector<at::Tensor> scale_input = {reduced_scale};
            auto wg_scale = nccl_pg_->allgather(gathered_scales, scale_input, allgather_opts);
            lowbitBackendTiming(this, getRank(), "allgather scales launched");
            if (!wg_scale->wait()) {
                lowbitBackendTiming(this, getRank(), "allgather scales wait failed");
                return false;
            }
            lowbitBackendTiming(this, getRank(), "allgather scales wait done");

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
            lowbitBackendTiming(this, getRank(), "restore done tensor_numel=" + std::to_string(s.flat.numel()));
        }
        return true;
    };

    bool success = post_hook();
    if (success && launcher_stream.has_value()) {
        checkCuda(cudaStreamSynchronize(launcher_stream->stream()), "cudaStreamSynchronize");
        lowbitBackendTiming(this, getRank(), "launcher stream synchronized");
    }
    return success;
}

void ProcessGroupLowBit::enqueueLowBitTask(std::function<void()> task) {
    size_t queue_size = 0;
    {
        std::lock_guard<std::mutex> lock(launcher_mutex_);
        launcher_queue_.push_back(std::move(task));
        queue_size = launcher_queue_.size();
    }
    lowbitBackendTiming(this, getRank(), "enqueueLowBitTask queued queue_size=" + std::to_string(queue_size));
    launcher_cv_.notify_one();
}

void ProcessGroupLowBit::launcherLoop() {
    while (true) {
        std::function<void()> task;
        {
            std::unique_lock<std::mutex> lock(launcher_mutex_);
            launcher_cv_.wait(lock, [this]() {
                return launcher_shutdown_ || !launcher_queue_.empty();
            });
            if (launcher_shutdown_ && launcher_queue_.empty()) {
                return;
            }
            task = std::move(launcher_queue_.front());
            launcher_queue_.pop_front();
            lowbitBackendTiming(this, getRank(), "launcher pop queue_remaining=" + std::to_string(launcher_queue_.size()));
        }
        task();
    }
}

c10::cuda::CUDAStream ProcessGroupLowBit::getLauncherStream(int device_index) {
    c10::cuda::CUDAGuard device_guard(device_index);
    auto it = launcher_streams_.find(device_index);
    if (it != launcher_streams_.end()) {
        return *(it->second);
    }

    auto stream = c10::cuda::getStreamFromPool(false, device_index);
    auto inserted = launcher_streams_.emplace(
        device_index,
        std::make_unique<c10::cuda::CUDAStream>(stream));
    lowbitBackendTiming(
        this,
        getRank(),
        "created launcher stream device=" + std::to_string(device_index));
    return *(inserted.first->second);
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

    if (output_tensors.empty() || input_tensors.empty()) {
        return nccl_pg_->reduce_scatter(output_tensors, input_tensors, opts);
    }

    if (options_.bitwidth >= 16 ||
        opts.reduceOp != c10d::ReduceOp::SUM ||
        getSize() <= 1) {
        return nccl_pg_->reduce_scatter(output_tensors, input_tensors, opts);
    }

    TORCH_CHECK(
        output_tensors.size() == input_tensors.size(),
        "reduce_scatter expects output_tensors.size == input_tensors.size, got output=",
        output_tensors.size(), " input=", input_tensors.size());

    const int world_size = getSize();
    const int rank = getRank();

    struct ReduceScatterState {
        at::Tensor output;
        int64_t out_numel = 0;
        std::vector<at::Tensor> recv_packed;
        std::vector<at::Tensor> recv_scales;
        std::vector<at::Tensor> send_packed;
        std::vector<at::Tensor> send_scales;
    };

    auto state = std::make_shared<std::vector<ReduceScatterState>>();
    state->reserve(output_tensors.size());

    c10d::AllToAllOptions alltoall_opts;
    std::vector<c10::intrusive_ptr<c10d::Work>> phase1_works;

    for (size_t idx = 0; idx < output_tensors.size(); ++idx) {
        auto& output = output_tensors[idx];
        auto& inputs = input_tensors[idx];

        TORCH_CHECK(
            inputs.size() == static_cast<size_t>(world_size),
            "reduce_scatter expects input_tensors[", idx, "] size == world_size, got ",
            inputs.size(), " vs ", world_size);

        ReduceScatterState s;
        s.output = output;
        s.out_numel = output.numel();

        TORCH_CHECK(
            s.out_numel == inputs[rank].numel(),
            "reduce_scatter output numel mismatch at index ", idx,
            ": output=", s.out_numel, " input[rank]=", inputs[rank].numel());

        s.send_packed.reserve(world_size);
        s.recv_packed.reserve(world_size);
        s.send_scales.reserve(world_size);
        s.recv_scales.reserve(world_size);

        for (int shard_idx = 0; shard_idx < world_size; ++shard_idx) {
            auto& shard = inputs[shard_idx];
            TORCH_CHECK(
                shard.numel() == s.out_numel,
                "reduce_scatter expects equal shard sizes at index ", idx,
                ": shard=", shard.numel(), " output=", s.out_numel);

            at::Tensor packed, scale;
            std::tie(packed, scale) = pack(shard);
            s.send_packed.push_back(packed);
            s.send_scales.push_back(scale);
            s.recv_packed.push_back(at::empty_like(packed));
            s.recv_scales.push_back(at::empty_like(scale));
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

        for (auto& s : *state) {
            auto local_sum = at::zeros({s.out_numel}, s.output.options().dtype(at::kFloat));
            for (int src = 0; src < world_size; ++src) {
                auto fp = unpack(
                    s.recv_packed[src],
                    s.out_numel,
                    s.recv_scales[src],
                    s.output.device(),
                    at::kFloat);
                local_sum.add_(fp);
            }
            s.output.copy_(local_sum.to(s.output.scalar_type()));
        }
        return true;
    };

    return c10::make_intrusive<WorkLowBit>(std::move(anchor), std::move(post_hook));
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
    bool error_feedback,
    const std::string& error_feedback_mode,
    int block_size,
    bool stage2_error_feedback) {

    LowBitOptions opts;
    opts.timeout = timeout;
    opts.bitwidth = bitwidth;
    opts.error_feedback = error_feedback;
    opts.error_feedback_mode = error_feedback_mode;
    opts.block_size = block_size;
    opts.stage2_error_feedback = stage2_error_feedback;
    return c10::make_intrusive<ProcessGroupLowBit>(
        store, rank, size, std::move(opts));
}

}  // namespace bitscom
