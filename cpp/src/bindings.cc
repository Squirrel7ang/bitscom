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
    const std::string& error_feedback_mode) {
    return bitscom::createProcessGroupLowBit(
        store, rank, size, timeout, bitwidth, error_feedback, error_feedback_mode);
}

PYBIND11_MODULE(_lowbit_c, m) {
    m.doc() = "bitscom lowbit distributed backend";

    // 暴露 LowBitOptions
    py::class_<bitscom::LowBitOptions>(m, "LowBitOptions")
        .def(py::init<>())
        .def_readwrite("bitwidth", &bitscom::LowBitOptions::bitwidth)
        .def_readwrite("error_feedback", &bitscom::LowBitOptions::error_feedback)
        .def_readwrite("error_feedback_mode", &bitscom::LowBitOptions::error_feedback_mode);

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
            py::arg("options") = bitscom::LowBitOptions());

    // 暴露工厂函数
    m.def("create_backend", &createBackend,
          py::arg("store"),
          py::arg("rank"),
          py::arg("size"),
            py::arg("timeout") = std::chrono::milliseconds(600000),
            py::arg("bitwidth") = 4,
                        py::arg("error_feedback") = false,
                        py::arg("error_feedback_mode") = "auto");
}
