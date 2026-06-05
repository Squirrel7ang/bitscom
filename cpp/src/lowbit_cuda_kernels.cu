#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

#include <chrono>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <utility>

namespace {

inline void check_cuda(cudaError_t err, const char* msg) {
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string(msg) + ": " + cudaGetErrorString(err));
    }
}

inline std::pair<int, int> quant_bounds(int bitwidth) {
    if (bitwidth == 1) {
        return {0, 1};
    }
    int qmax = (1 << (bitwidth - 1)) - 1;
    int qmin = -(1 << (bitwidth - 1));
    return {qmin, qmax};
}

__global__ void quantize_kernel(
    const float* input,
    int16_t* output,
    int64_t n,
    float scale,
    int qmin,
    int qmax,
    bool stochastic_rounding,
    uint64_t seed) {
    int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= n) {
        return;
    }

    float scaled = input[idx] / scale;
    float v = 0.0f;
    if (stochastic_rounding) {
        float lower = floorf(scaled);
        float p = scaled - lower;

        // Stateless per-element RNG for deterministic-on-seed stochastic rounding.
        uint64_t x = seed ^ (static_cast<uint64_t>(idx) + 0x9e3779b97f4a7c15ULL);
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        uint32_t r = static_cast<uint32_t>((x * 0x2545F4914F6CDD1DULL) >> 32);
        float u = static_cast<float>(r) / 4294967296.0f;

        v = lower + ((u < p) ? 1.0f : 0.0f);
    } else {
        v = nearbyintf(scaled);
    }

    if (v < static_cast<float>(qmin)) {
        v = static_cast<float>(qmin);
    }
    if (v > static_cast<float>(qmax)) {
        v = static_cast<float>(qmax);
    }
    output[idx] = static_cast<int16_t>(v);
}

__global__ void dequantize_kernel(
    const int16_t* input,
    float* output,
    int64_t n,
    float scale) {
    int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= n) {
        return;
    }
    output[idx] = static_cast<float>(input[idx]) * scale;
}

__global__ void pack_lowbit_kernel(
    const int16_t* q,
    uint8_t* packed,
    int64_t numel,
    int bitwidth,
    int qmin) {
    int per_byte = 8 / bitwidth;
    int64_t byte_idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    int64_t num_bytes = (numel + per_byte - 1) / per_byte;
    if (byte_idx >= num_bytes) {
        return;
    }

    uint8_t out = 0;
    int mask = (1 << bitwidth) - 1;
    for (int j = 0; j < per_byte; ++j) {
        int64_t elem = byte_idx * per_byte + j;
        int v = 0;
        if (elem < numel) {
            v = static_cast<int>(q[elem]) - qmin;
            if (v < 0) {
                v = 0;
            }
            if (v > mask) {
                v = mask;
            }
        }
        out |= static_cast<uint8_t>((v & mask) << (j * bitwidth));
    }
    packed[byte_idx] = out;
}

__global__ void unpack_lowbit_kernel(
    const uint8_t* packed,
    int16_t* q,
    int64_t numel,
    int bitwidth,
    int qmin) {
    int per_byte = 8 / bitwidth;
    int mask = (1 << bitwidth) - 1;

    int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= numel) {
        return;
    }

    int64_t byte_idx = idx / per_byte;
    int lane = idx % per_byte;
    int raw = (packed[byte_idx] >> (lane * bitwidth)) & mask;
    q[idx] = static_cast<int16_t>(raw + qmin);
}

__device__ inline float sr_uniform(uint64_t seed, int64_t idx) {
    uint64_t x = seed ^ (static_cast<uint64_t>(idx) + 0x9e3779b97f4a7c15ULL);
    x ^= x >> 12;
    x ^= x << 25;
    x ^= x >> 27;
    uint32_t r = static_cast<uint32_t>((x * 0x2545F4914F6CDD1DULL) >> 32);
    return static_cast<float>(r) / 4294967296.0f;
}

template <typename scalar_t>
__global__ void blockwise_quantize_pack_kernel(
    const scalar_t* input,
    uint8_t* packed,
    at::Half* scales,
    int64_t numel,
    int bitwidth,
    int block_size,
    int qmin,
    int qmax,
    int per_byte,
    bool stochastic_rounding,
    uint64_t seed) {
    extern __shared__ float shared_max[];

    int block_idx = blockIdx.x;
    int64_t block_start = static_cast<int64_t>(block_idx) * block_size;
    int64_t block_end = block_start + static_cast<int64_t>(block_size);
    if (block_end > numel) {
        block_end = numel;
    }

    float local_max = 0.0f;
    for (int64_t idx = block_start + threadIdx.x; idx < block_end; idx += blockDim.x) {
        float v = static_cast<float>(input[idx]);
        local_max = fmaxf(local_max, fabsf(v));
    }

    shared_max[threadIdx.x] = local_max;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            shared_max[threadIdx.x] = fmaxf(shared_max[threadIdx.x], shared_max[threadIdx.x + stride]);
        }
        __syncthreads();
    }

    float max_abs = shared_max[0];
    float scale = (max_abs > 0.0f) ? (max_abs / static_cast<float>(qmax)) : 1.0f;
    if (threadIdx.x == 0) {
        scales[block_idx] = static_cast<at::Half>(scale);
    }
    __syncthreads();

    float scale_for_norm = static_cast<float>(static_cast<at::Half>(scale));
    int mask = (1 << bitwidth) - 1;
    int64_t byte_start = block_start / per_byte;
    int64_t bytes_in_block = ((block_end - block_start) + per_byte - 1) / per_byte;

    for (int64_t local_byte = threadIdx.x; local_byte < bytes_in_block; local_byte += blockDim.x) {
        int64_t elem_base = block_start + local_byte * per_byte;
        uint8_t out = 0;
        for (int lane = 0; lane < per_byte; ++lane) {
            int64_t elem = elem_base + lane;
            int q = 0;
            if (elem < block_end) {
                float v = static_cast<float>(input[elem]);
                float norm = fabsf(v) / scale_for_norm;
                float mag;
                if (stochastic_rounding) {
                    float lower = floorf(norm);
                    float p = fminf(fmaxf(norm - lower, 0.0f), 1.0f);
                    mag = lower + ((sr_uniform(seed, elem) < p) ? 1.0f : 0.0f);
                } else {
                    mag = nearbyintf(norm);
                }
                float signed_v = (bitwidth == 1) ? mag : (mag * ((v > 0.0f) - (v < 0.0f)));
                signed_v = fminf(fmaxf(signed_v, static_cast<float>(qmin)), static_cast<float>(qmax));
                q = static_cast<int>(signed_v) - qmin;
                if (q < 0) {
                    q = 0;
                }
                if (q > mask) {
                    q = mask;
                }
            }
            out |= static_cast<uint8_t>((q & mask) << (lane * bitwidth));
        }
        packed[byte_start + local_byte] = out;
    }
}

template <typename scalar_t>
__global__ void blockwise_unpack_dequantize_kernel(
    const uint8_t* packed,
    const at::Half* scales,
    scalar_t* output,
    int64_t numel,
    int bitwidth,
    int block_size,
    int qmin,
    int per_byte) {
    int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= numel) {
        return;
    }

    int64_t byte_idx = idx / per_byte;
    int lane = idx % per_byte;
    int mask = (1 << bitwidth) - 1;
    int raw = (packed[byte_idx] >> (lane * bitwidth)) & mask;
    int q = raw + qmin;
    int64_t block_idx = idx / block_size;
    float scale = static_cast<float>(scales[block_idx]);
    output[idx] = static_cast<scalar_t>(static_cast<float>(q) * scale);
}

}  // namespace

namespace bitscom {

std::pair<at::Tensor, double> quantize_cuda(
    const at::Tensor& input,
    int64_t bitwidth,
    bool stochastic_rounding) {
    TORCH_CHECK(input.is_cuda(), "quantize_cuda expects CUDA tensor");
    TORCH_CHECK(input.is_floating_point(), "quantize_cuda expects floating tensor");
    TORCH_CHECK(bitwidth == 1 || bitwidth == 2 || bitwidth == 4 || bitwidth == 8,
                "CUDA quantization currently supports 1/2/4/8 bit");

    auto bounds = quant_bounds(static_cast<int>(bitwidth));
    int qmin = bounds.first;
    int qmax = bounds.second;

    auto flat = input.contiguous().view(-1).to(at::kFloat);
    float max_abs = at::abs(flat).max().item<float>();
    float scale = (max_abs == 0.0f) ? 1.0f : (max_abs / static_cast<float>(qmax));

    auto q = at::empty_like(flat, flat.options().dtype(at::kShort));
    int64_t n = flat.numel();
    int threads = 256;
    int blocks = static_cast<int>((n + threads - 1) / threads);
    auto stream = at::cuda::getCurrentCUDAStream().stream();

    uint64_t seed = static_cast<uint64_t>(
        std::chrono::high_resolution_clock::now().time_since_epoch().count());
    seed ^= static_cast<uint64_t>(reinterpret_cast<uintptr_t>(flat.data_ptr<float>()));

    quantize_kernel<<<blocks, threads, 0, stream>>>(
        flat.data_ptr<float>(),
        q.data_ptr<int16_t>(),
        n,
        scale,
        qmin,
        qmax,
        stochastic_rounding,
        seed);
    check_cuda(cudaGetLastError(), "quantize_kernel launch failed");

    return {q.view(input.sizes()), static_cast<double>(scale)};
}

at::Tensor dequantize_cuda(
    const at::Tensor& q_tensor,
    double scale) {
    TORCH_CHECK(q_tensor.is_cuda(), "dequantize_cuda expects CUDA tensor");
    TORCH_CHECK(q_tensor.scalar_type() == at::kShort,
                "dequantize_cuda expects int16 tensor");

    auto flat = q_tensor.contiguous().view(-1);
    auto out = at::empty_like(flat, flat.options().dtype(at::kFloat));

    int64_t n = flat.numel();
    int threads = 256;
    int blocks = static_cast<int>((n + threads - 1) / threads);
    auto stream = at::cuda::getCurrentCUDAStream().stream();

    dequantize_kernel<<<blocks, threads, 0, stream>>>(
        flat.data_ptr<int16_t>(),
        out.data_ptr<float>(),
        n,
        static_cast<float>(scale));
    check_cuda(cudaGetLastError(), "dequantize_kernel launch failed");

    return out.view(q_tensor.sizes());
}

at::Tensor pack_lowbit_cuda(
    const at::Tensor& q_tensor,
    int64_t bitwidth) {
    TORCH_CHECK(q_tensor.is_cuda(), "pack_lowbit_cuda expects CUDA tensor");
    TORCH_CHECK(q_tensor.scalar_type() == at::kShort,
                "pack_lowbit_cuda expects int16 tensor");
    TORCH_CHECK(bitwidth == 1 || bitwidth == 2 || bitwidth == 4 || bitwidth == 8,
                "CUDA packing currently supports 1/2/4/8 bit");

    auto bounds = quant_bounds(static_cast<int>(bitwidth));
    int qmin = bounds.first;

    auto flat = q_tensor.contiguous().view(-1);
    int64_t numel = flat.numel();
    int per_byte = 8 / static_cast<int>(bitwidth);
    int64_t num_bytes = (numel + per_byte - 1) / per_byte;

    auto packed = at::empty({num_bytes}, flat.options().dtype(at::kByte));

    int threads = 256;
    int blocks = static_cast<int>((num_bytes + threads - 1) / threads);
    auto stream = at::cuda::getCurrentCUDAStream().stream();

    pack_lowbit_kernel<<<blocks, threads, 0, stream>>>(
        flat.data_ptr<int16_t>(),
        packed.data_ptr<uint8_t>(),
        numel,
        static_cast<int>(bitwidth),
        qmin);
    check_cuda(cudaGetLastError(), "pack_lowbit_kernel launch failed");

    return packed;
}

at::Tensor unpack_lowbit_cuda(
    const at::Tensor& packed,
    int64_t bitwidth,
    int64_t numel) {
    TORCH_CHECK(packed.is_cuda(), "unpack_lowbit_cuda expects CUDA tensor");
    TORCH_CHECK(packed.scalar_type() == at::kByte,
                "unpack_lowbit_cuda expects uint8 tensor");
    TORCH_CHECK(bitwidth == 1 || bitwidth == 2 || bitwidth == 4 || bitwidth == 8,
                "CUDA unpacking currently supports 1/2/4/8 bit");

    auto bounds = quant_bounds(static_cast<int>(bitwidth));
    int qmin = bounds.first;

    auto out = at::empty({numel}, packed.options().dtype(at::kShort));

    int threads = 256;
    int blocks = static_cast<int>((numel + threads - 1) / threads);
    auto stream = at::cuda::getCurrentCUDAStream().stream();

    unpack_lowbit_kernel<<<blocks, threads, 0, stream>>>(
        packed.contiguous().data_ptr<uint8_t>(),
        out.data_ptr<int16_t>(),
        numel,
        static_cast<int>(bitwidth),
        qmin);
    check_cuda(cudaGetLastError(), "unpack_lowbit_kernel launch failed");

    return out;
}

std::tuple<at::Tensor, at::Tensor, int64_t> blockwise_quantize_pack_cuda(
    const at::Tensor& input,
    int64_t bitwidth,
    int64_t block_size,
    bool stochastic_rounding) {
    TORCH_CHECK(input.is_cuda(), "blockwise_quantize_pack_cuda expects CUDA tensor");
    TORCH_CHECK(input.is_floating_point(), "blockwise_quantize_pack_cuda expects floating tensor");
    TORCH_CHECK(input.scalar_type() == at::kFloat || input.scalar_type() == at::kHalf,
                "blockwise_quantize_pack_cuda supports float32/float16 tensors");
    TORCH_CHECK(bitwidth == 1 || bitwidth == 2 || bitwidth == 4 || bitwidth == 8,
                "CUDA fused blockwise packing currently supports 1/2/4/8 bit");
    TORCH_CHECK(block_size > 0, "block_size must be > 0");

    int bw = static_cast<int>(bitwidth);
    int bs = static_cast<int>(block_size);
    int per_byte = 8 / bw;
    TORCH_CHECK(bs % per_byte == 0,
                "CUDA fused blockwise packing requires block_size divisible by values per packed byte");

    auto flat = input.contiguous().view(-1);
    int64_t numel = flat.numel();
    int64_t num_blocks = (numel + bs - 1) / bs;
    int64_t num_bytes = (numel + per_byte - 1) / per_byte;
    auto packed = at::empty({num_bytes}, flat.options().dtype(at::kByte));
    auto scales = at::empty({num_blocks}, flat.options().dtype(at::kHalf));
    if (numel == 0) {
        return {packed, scales, numel};
    }

    auto bounds = quant_bounds(bw);
    int qmin = bounds.first;
    int qmax = bounds.second;
    int threads = 256;
    auto stream = at::cuda::getCurrentCUDAStream().stream();

    uint64_t seed = static_cast<uint64_t>(
        std::chrono::high_resolution_clock::now().time_since_epoch().count());
    seed ^= static_cast<uint64_t>(reinterpret_cast<uintptr_t>(flat.data_ptr()));

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(flat.scalar_type(), "blockwise_quantize_pack_cuda", [&] {
        blockwise_quantize_pack_kernel<scalar_t><<<
            static_cast<int>(num_blocks),
            threads,
            threads * sizeof(float),
            stream>>>(
            flat.data_ptr<scalar_t>(),
            packed.data_ptr<uint8_t>(),
            scales.data_ptr<at::Half>(),
            numel,
            bw,
            bs,
            qmin,
            qmax,
            per_byte,
            stochastic_rounding,
            seed);
    });
    check_cuda(cudaGetLastError(), "blockwise_quantize_pack_kernel launch failed");

    return {packed, scales, numel};
}

at::Tensor blockwise_unpack_dequantize_cuda(
    const at::Tensor& packed,
    const at::Tensor& scales,
    int64_t bitwidth,
    int64_t numel,
    int64_t block_size,
    at::ScalarType dtype) {
    TORCH_CHECK(packed.is_cuda(), "blockwise_unpack_dequantize_cuda expects CUDA packed tensor");
    TORCH_CHECK(scales.is_cuda(), "blockwise_unpack_dequantize_cuda expects CUDA scales tensor");
    TORCH_CHECK(packed.scalar_type() == at::kByte,
                "blockwise_unpack_dequantize_cuda expects uint8 packed tensor");
    TORCH_CHECK(scales.scalar_type() == at::kHalf,
                "blockwise_unpack_dequantize_cuda expects float16 scales tensor");
    TORCH_CHECK(dtype == at::kFloat || dtype == at::kHalf,
                "blockwise_unpack_dequantize_cuda supports float32/float16 output");
    TORCH_CHECK(bitwidth == 1 || bitwidth == 2 || bitwidth == 4 || bitwidth == 8,
                "CUDA fused blockwise unpacking currently supports 1/2/4/8 bit");
    TORCH_CHECK(block_size > 0, "block_size must be > 0");

    int bw = static_cast<int>(bitwidth);
    int bs = static_cast<int>(block_size);
    int per_byte = 8 / bw;
    TORCH_CHECK(bs % per_byte == 0,
                "CUDA fused blockwise unpacking requires block_size divisible by values per packed byte");

    auto out = at::empty({numel}, packed.options().dtype(dtype));
    if (numel == 0) {
        return out;
    }

    auto bounds = quant_bounds(bw);
    int qmin = bounds.first;
    int threads = 256;
    int blocks = static_cast<int>((numel + threads - 1) / threads);
    auto stream = at::cuda::getCurrentCUDAStream().stream();
    auto packed_contig = packed.contiguous();
    auto scales_contig = scales.contiguous();

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(out.scalar_type(), "blockwise_unpack_dequantize_cuda", [&] {
        blockwise_unpack_dequantize_kernel<scalar_t><<<blocks, threads, 0, stream>>>(
            packed_contig.data_ptr<uint8_t>(),
            scales_contig.data_ptr<at::Half>(),
            out.data_ptr<scalar_t>(),
            numel,
            bw,
            bs,
            qmin,
            per_byte);
    });
    check_cuda(cudaGetLastError(), "blockwise_unpack_dequantize_kernel launch failed");

    return out;
}

}  // namespace bitscom
