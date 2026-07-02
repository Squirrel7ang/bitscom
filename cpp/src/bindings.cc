// cpp/src/bindings.cc
//
// Pybind11 绑定：将 ProcessGroupLowBit 暴露给 Python
//
#include <pybind11/pybind11.h>
#include <pybind11/chrono.h>
#include <pybind11/functional.h>
#include <pybind11/stl.h>
#include <torch/csrc/distributed/c10d/Backend.hpp>
#include <torch/extension.h>

#include <sstream>
#include <string>

#include "process_group_lowbit.h"

namespace py = pybind11;

// 工厂函数：供 Python 侧 register_backend 使用
c10::intrusive_ptr<c10d::Backend> createBackend(
    const c10::intrusive_ptr<c10d::Store>& store,
    int rank,
    int size,
    const std::chrono::milliseconds& timeout,
    int bitwidth,
    bool error_feedback,
    const std::string& error_feedback_mode,
    int block_size,
    bool stage2_error_feedback) {
    return bitscom::createProcessGroupLowBit(
        store,
        rank,
        size,
        timeout,
        bitwidth,
        error_feedback,
        error_feedback_mode,
        block_size,
        stage2_error_feedback);
}

// Trampoline class: allows Python to subclass ITensorPartitionStrategy
class PyTensorPartitionStrategy : public bitscom::ITensorPartitionStrategy {
public:
    using bitscom::ITensorPartitionStrategy::ITensorPartitionStrategy;

    bool prepare(
        std::vector<at::Tensor>& tensors,
        int rank,
        int world_size,
        ncclComm_t comm,
        c10::cuda::CUDAStream stream) override {
        namespace py = pybind11;
        py::gil_scoped_acquire gil;
        py::function overload = py::get_override(this, "prepare");
        if (overload) {
            // Pass comm and stream as opaque uintptr_t; Python overrides
            // typically ignore them (they use torch.distributed for comm).
            auto result = overload(
                tensors,
                rank,
                world_size,
                reinterpret_cast<uintptr_t>(comm),
                reinterpret_cast<uintptr_t>(stream.stream()));
            return result.cast<bool>();
        }
        return bitscom::ITensorPartitionStrategy::prepare(
            tensors, rank, world_size, comm, stream);
    }

    std::vector<bitscom::QuantizationSegment> partition(
        const at::Tensor& flat_tensor) override {
        namespace py = pybind11;
        py::gil_scoped_acquire gil;
        py::function overload = py::get_override(this, "partition");
        if (overload) {
            auto result = overload(flat_tensor);
            return result.cast<std::vector<bitscom::QuantizationSegment>>();
        }
        // Default: one segment covering everything, all quantized
        return {{0, flat_tensor.numel(), true}};
    }

    const char* name() const override {
        namespace py = pybind11;
        py::gil_scoped_acquire gil;
        py::function overload = py::get_override(this, "name");
        if (overload) {
            auto result = overload();
            // Cache the string to avoid dangling pointer.
            auto str = result.cast<std::string>();
            char* buf = new char[str.size() + 1];
            std::memcpy(buf, str.c_str(), str.size() + 1);
            return buf;  // Leaked, but acceptable for a small debug string.
        }
        return "python_strategy";
    }

    bool isActive() const override {
        namespace py = pybind11;
        py::gil_scoped_acquire gil;
        py::function overload = py::get_override(this, "is_active");
        if (overload) {
            auto result = overload();
            return result.cast<bool>();
        }
        return bitscom::ITensorPartitionStrategy::isActive();
    }
};

PYBIND11_MODULE(_lowbit_c, m) {
    m.doc() = "bitscom lowbit distributed backend";

    // ---- QuantizationSegment ----
    py::class_<bitscom::QuantizationSegment>(m, "QuantizationSegment")
        .def(py::init<>())
        .def_readwrite("offset", &bitscom::QuantizationSegment::offset)
        .def_readwrite("numel", &bitscom::QuantizationSegment::numel)
        .def_readwrite("quantize", &bitscom::QuantizationSegment::quantize)
        .def_readwrite("drop", &bitscom::QuantizationSegment::drop)
        .def("__repr__", [](const bitscom::QuantizationSegment& s) {
            std::ostringstream oss;
            oss << "QuantizationSegment(offset=" << s.offset
                << ", numel=" << s.numel
                << ", quantize=" << (s.quantize ? "True" : "False")
                << ", drop=" << (s.drop ? "True" : "False") << ")";
            return oss.str();
        });

    // ---- ITensorPartitionStrategy (with trampoline for Python subclassing) ----
    py::class_<
        bitscom::ITensorPartitionStrategy,
        PyTensorPartitionStrategy,
        std::shared_ptr<bitscom::ITensorPartitionStrategy>>(
        m, "TensorPartitionStrategy")
        .def(py::init<>())
        .def(
            "prepare",
            [](bitscom::ITensorPartitionStrategy& self,
               std::vector<at::Tensor>& tensors,
               int rank,
               int world_size,
               uintptr_t comm_ptr,
               uintptr_t stream_ptr) -> bool {
                auto comm = reinterpret_cast<ncclComm_t>(comm_ptr);
                // Reconstruct a CUDAStream from the raw pointer.  When
                // getStreamFromExternal is not available we fall back to
                // the default stream — the Python-side prepare() rarely
                // uses the stream directly.
                (void)stream_ptr;
                auto device_index = c10::cuda::current_device();
                auto stream = c10::cuda::getDefaultCUDAStream(device_index);
                return self.prepare(tensors, rank, world_size, comm, stream);
            },
            py::arg("tensors"),
            py::arg("rank"),
            py::arg("world_size"),
            py::arg("comm_ptr"),
            py::arg("stream_ptr"))
        .def("partition", &bitscom::ITensorPartitionStrategy::partition,
             py::arg("flat_tensor"))
        .def("name", &bitscom::ITensorPartitionStrategy::name)
        .def("is_active", &bitscom::ITensorPartitionStrategy::isActive);

    // ---- Default full-quantization strategy (convenience) ----
    class FullQuantizationStrategy : public bitscom::ITensorPartitionStrategy {
    public:
        std::vector<bitscom::QuantizationSegment> partition(
            const at::Tensor& flat_tensor) override {
            return {{0, flat_tensor.numel(), true}};
        }
        const char* name() const override { return "full_quantization"; }
        bool isActive() const override { return false; }
    };

    py::class_<
        FullQuantizationStrategy,
        bitscom::ITensorPartitionStrategy,
        std::shared_ptr<FullQuantizationStrategy>>(
        m, "FullQuantizationStrategy")
        .def(py::init<>());

    // 暴露 LowBitOptions
    py::class_<bitscom::LowBitOptions>(m, "LowBitOptions")
        .def(py::init<>())
        .def_readwrite("bitwidth", &bitscom::LowBitOptions::bitwidth)
        .def_readwrite("error_feedback", &bitscom::LowBitOptions::error_feedback)
        .def_readwrite("error_feedback_mode", &bitscom::LowBitOptions::error_feedback_mode)
        .def_readwrite("block_size", &bitscom::LowBitOptions::block_size)
        .def_readwrite("stage2_error_feedback", &bitscom::LowBitOptions::stage2_error_feedback);

    py::class_<bitscom::LowBitScheduledHandle, std::shared_ptr<bitscom::LowBitScheduledHandle>>(
        m,
        "LowBitScheduledHandle")
        .def("launch_phase2", [](const std::shared_ptr<bitscom::LowBitScheduledHandle>& handle) {
            return handle->owner->launchScheduledLowBitPhase2(handle);
        })
        .def("launch_restore", [](const std::shared_ptr<bitscom::LowBitScheduledHandle>& handle) {
            return handle->owner->launchScheduledLowBitRestore(handle);
        })
        .def(
            "is_completed",
            [](const std::shared_ptr<bitscom::LowBitScheduledHandle>& handle) {
                return handle->owner->scheduledLowBitIsCompleted(handle, false);
            })
        .def(
            "wait",
            [](const std::shared_ptr<bitscom::LowBitScheduledHandle>& handle) {
                return handle->owner->scheduledLowBitWait(handle);
            })
        .def(
            "block_current_stream",
            [](const std::shared_ptr<bitscom::LowBitScheduledHandle>& handle) {
                return handle->owner->scheduledLowBitBlockCurrentStream(handle);
            });

    // 暴露 ProcessGroupLowBit（作为 Backend 的子类）
    py::class_<
        bitscom::ProcessGroupLowBit,
        c10d::Backend,
        c10::intrusive_ptr<bitscom::ProcessGroupLowBit>>(m, "ProcessGroupLowBit")
        .def(
            py::init([](const c10::intrusive_ptr<c10d::Store>& store,
                        int rank,
                        int size,
                        bitscom::LowBitOptions options) {
                return c10::make_intrusive<bitscom::ProcessGroupLowBit>(
                    store, rank, size, std::move(options));
            }),
            py::arg("store"),
            py::arg("rank"),
            py::arg("size"),
            py::arg("options") = bitscom::LowBitOptions())
        .def(
            "progress_lowbit",
            &bitscom::ProcessGroupLowBit::progressLowBit,
            py::arg("block") = false)
        .def(
            "schedule_lowbit_allreduce",
            &bitscom::ProcessGroupLowBit::scheduleLowBitAllreduce,
            py::arg("tensor"))
        .def(
            "launch_scheduled_lowbit_phase2",
            &bitscom::ProcessGroupLowBit::launchScheduledLowBitPhase2,
            py::arg("handle"))
        .def(
            "launch_scheduled_lowbit_restore",
            &bitscom::ProcessGroupLowBit::launchScheduledLowBitRestore,
            py::arg("handle"))
        .def(
            "scheduled_lowbit_is_completed",
            &bitscom::ProcessGroupLowBit::scheduledLowBitIsCompleted,
            py::arg("handle"),
            py::arg("block") = false)
        .def(
            "scheduled_lowbit_wait",
            &bitscom::ProcessGroupLowBit::scheduledLowBitWait,
            py::arg("handle"))
        .def(
            "scheduled_lowbit_block_current_stream",
            &bitscom::ProcessGroupLowBit::scheduledLowBitBlockCurrentStream,
            py::arg("handle"))
        .def(
            "set_partition_strategy",
            &bitscom::ProcessGroupLowBit::setPartitionStrategy,
            py::arg("strategy"),
            "Set a sparse/selective quantization strategy. Pass None to revert to "
            "full quantization.")
        .def(
            "get_partition_strategy",
            &bitscom::ProcessGroupLowBit::getPartitionStrategy,
            "Get the current partition strategy (may be None).");

    // Helper: 在 ProcessGroup 的 lowbit backend 上设置策略。
    m.def(
        "_set_strategy_on_pg",
        [](const c10::intrusive_ptr<c10d::ProcessGroup>& pg,
           std::shared_ptr<bitscom::ITensorPartitionStrategy> strategy) {
            auto backend = pg->getBackend(c10::DeviceType::CUDA);
            auto lowbit =
                dynamic_cast<bitscom::ProcessGroupLowBit*>(backend.get());
            if (!lowbit) {
                throw std::runtime_error(
                    "ProcessGroup backend is not a ProcessGroupLowBit. "
                    "Did you call init_process_group(backend='lowbit')?");
            }
            lowbit->setPartitionStrategy(std::move(strategy));
        },
        py::arg("pg"),
        py::arg("strategy"),
        "Set a partition strategy on a lowbit-backed process group.");

    // Helper: 从 ProcessGroup 获取策略名称（用于测试验证）。
    m.def(
        "_get_strategy_name_from_pg",
        [](const c10::intrusive_ptr<c10d::ProcessGroup>& pg) -> std::string {
            auto backend = pg->getBackend(c10::DeviceType::CUDA);
            auto lowbit =
                dynamic_cast<bitscom::ProcessGroupLowBit*>(backend.get());
            if (!lowbit) {
                throw std::runtime_error(
                    "ProcessGroup backend is not a ProcessGroupLowBit.");
            }
            auto s = lowbit->getPartitionStrategy();
            if (!s) return "none";
            return std::string(s->name());
        },
        py::arg("pg"),
        "Get the name of the current partition strategy on a lowbit-backed "
        "process group.");

    // 暴露工厂函数
    m.def("create_backend", &createBackend,
          py::arg("store"),
          py::arg("rank"),
          py::arg("size"),
            py::arg("timeout") = std::chrono::milliseconds(600000),
            py::arg("bitwidth") = 4,
                        py::arg("error_feedback") = false,
                        py::arg("error_feedback_mode") = "auto",
                        py::arg("block_size") = 256,
                        py::arg("stage2_error_feedback") = true);
}
