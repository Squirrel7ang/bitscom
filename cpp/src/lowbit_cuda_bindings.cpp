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
