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
    bool stage2_error_feedback,
    bool sparse_enabled,
    int sparse_projection_rank,
    float sparse_compression_ratio,
    int sparse_row_width,
    int sparse_priority_mode,
    int sparse_priority_quantize_bitwidth,
    int sparse_non_priority_mode,
    int sparse_non_priority_quantize_bitwidth) {
    return bitscom::createProcessGroupLowBit(
        store,
        rank,
        size,
        timeout,
        bitwidth,
        error_feedback,
        error_feedback_mode,
        block_size,
        stage2_error_feedback,
        sparse_enabled,
        sparse_projection_rank,
        sparse_compression_ratio,
        sparse_row_width,
        sparse_priority_mode,
        sparse_priority_quantize_bitwidth,
        sparse_non_priority_mode,
        sparse_non_priority_quantize_bitwidth);
}

PYBIND11_MODULE(_lowbit_c, m) {
    m.doc() = "bitscom lowbit distributed backend";

    // 暴露 LowBitOptions
    py::class_<bitscom::LowBitOptions>(m, "LowBitOptions")
        .def(py::init<>())
        .def_readwrite("bitwidth", &bitscom::LowBitOptions::bitwidth)
        .def_readwrite("error_feedback", &bitscom::LowBitOptions::error_feedback)
        .def_readwrite("error_feedback_mode", &bitscom::LowBitOptions::error_feedback_mode)
        .def_readwrite("block_size", &bitscom::LowBitOptions::block_size)
        .def_readwrite("stage2_error_feedback", &bitscom::LowBitOptions::stage2_error_feedback)
        .def_readwrite("sparse_enabled", &bitscom::LowBitOptions::sparse_enabled)
        .def_readwrite("sparse_projection_rank", &bitscom::LowBitOptions::sparse_projection_rank)
        .def_readwrite("sparse_compression_ratio", &bitscom::LowBitOptions::sparse_compression_ratio)
        .def_readwrite("sparse_row_width", &bitscom::LowBitOptions::sparse_row_width)
        .def_readwrite("sparse_priority_mode", &bitscom::LowBitOptions::sparse_priority_mode)
        .def_readwrite("sparse_priority_quantize_bitwidth", &bitscom::LowBitOptions::sparse_priority_quantize_bitwidth)
        .def_readwrite("sparse_non_priority_mode", &bitscom::LowBitOptions::sparse_non_priority_mode)
        .def_readwrite("sparse_non_priority_quantize_bitwidth", &bitscom::LowBitOptions::sparse_non_priority_quantize_bitwidth);

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
            py::arg("handle"));

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
                        py::arg("stage2_error_feedback") = true,
                        py::arg("sparse_enabled") = false,
                        py::arg("sparse_projection_rank") = 4,
                        py::arg("sparse_compression_ratio") = 0.1f,
                        py::arg("sparse_row_width") = 128,
                        py::arg("sparse_priority_mode") = 0,
                        py::arg("sparse_priority_quantize_bitwidth") = 4,
                        py::arg("sparse_non_priority_mode") = 1,
                        py::arg("sparse_non_priority_quantize_bitwidth") = 4);
}
