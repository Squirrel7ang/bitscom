#include <pybind11/pybind11.h>
#include <torch/extension.h>

#include <tuple>
#include <utility>

namespace py = pybind11;

namespace bitscom {
std::pair<at::Tensor, double> quantize_cuda(
    const at::Tensor& input,
    int64_t bitwidth,
    bool stochastic_rounding);
at::Tensor dequantize_cuda(
    const at::Tensor& q_tensor,
    double scale);
at::Tensor pack_lowbit_cuda(
    const at::Tensor& q_tensor,
    int64_t bitwidth);
at::Tensor unpack_lowbit_cuda(
    const at::Tensor& packed,
    int64_t bitwidth,
    int64_t numel);
std::tuple<at::Tensor, at::Tensor, int64_t> blockwise_quantize_pack_cuda(
    const at::Tensor& input,
    int64_t bitwidth,
    int64_t block_size,
    bool stochastic_rounding);
at::Tensor blockwise_unpack_dequantize_cuda(
    const at::Tensor& packed,
    const at::Tensor& scales,
    int64_t bitwidth,
    int64_t numel,
    int64_t block_size,
    at::ScalarType dtype);
}  // namespace bitscom

PYBIND11_MODULE(_lowbit_cuda, m) {
    m.doc() = "bitscom CUDA quantization kernels";

    m.def(
        "quantize_cuda",
        [](const at::Tensor& input, int64_t bitwidth, bool stochastic_rounding) {
            auto result = bitscom::quantize_cuda(input, bitwidth, stochastic_rounding);
            return py::make_tuple(result.first, result.second);
        },
        py::arg("input"),
        py::arg("bitwidth"),
        py::arg("stochastic_rounding") = false);

    m.def(
        "dequantize_cuda",
        &bitscom::dequantize_cuda,
        py::arg("q_tensor"),
        py::arg("scale"));

    m.def(
        "pack_lowbit_cuda",
        &bitscom::pack_lowbit_cuda,
        py::arg("q_tensor"),
        py::arg("bitwidth"));

    m.def(
        "unpack_lowbit_cuda",
        &bitscom::unpack_lowbit_cuda,
        py::arg("packed"),
        py::arg("bitwidth"),
        py::arg("numel"));

    m.def(
        "blockwise_quantize_pack_cuda",
        &bitscom::blockwise_quantize_pack_cuda,
        py::arg("input"),
        py::arg("bitwidth"),
        py::arg("block_size"),
        py::arg("stochastic_rounding") = false);

    m.def(
        "blockwise_unpack_dequantize_cuda",
        &bitscom::blockwise_unpack_dequantize_cuda,
        py::arg("packed"),
        py::arg("scales"),
        py::arg("bitwidth"),
        py::arg("numel"),
        py::arg("block_size"),
        py::arg("dtype") = at::kFloat);
}

// ---- Dispatcher 注册 ----
// _lowbit_c 是纯 CppExtension（无 nvcc），无法直接链接这里的 CUDA kernel。
// 把融合 kernel 注册到 dispatcher，C++ 侧即可通过 c10::Dispatcher 调用。
TORCH_LIBRARY(bitscom_lowbit, m) {
    m.def(
        "blockwise_quantize_pack(Tensor input, int bitwidth, int block_size, "
        "bool stochastic_rounding) -> (Tensor, Tensor, int)");
    m.def(
        "blockwise_unpack_dequantize(Tensor packed, Tensor scales, int bitwidth, "
        "int numel, int block_size, ScalarType dtype) -> Tensor");
}

TORCH_LIBRARY_IMPL(bitscom_lowbit, CUDA, m) {
    m.impl(
        "blockwise_quantize_pack",
        &bitscom::blockwise_quantize_pack_cuda);
    m.impl(
        "blockwise_unpack_dequantize",
        &bitscom::blockwise_unpack_dequantize_cuda);
}
