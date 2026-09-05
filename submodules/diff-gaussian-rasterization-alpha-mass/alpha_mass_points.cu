#include "alpha_mass_points.h"

#include <cooperative_groups.h>
#include <cuda_runtime.h>
#include <functional>
#include <stdexcept>

#include "cuda_rasterizer/config.h"
#include "cuda_rasterizer/auxiliary.h"
#include "cuda_rasterizer/rasterizer.h"
#include "cuda_rasterizer/rasterizer_impl.h"

namespace cg = cooperative_groups;

namespace {

std::function<char*(size_t)> resize_tensor(torch::Tensor& tensor) {
    return [&tensor](size_t bytes) {
        tensor.resize_({static_cast<long long>(bytes)});
        return reinterpret_cast<char*>(tensor.contiguous().data_ptr());
    };
}

__device__ __forceinline__ float warp_sum(unsigned warp, float value) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        value += __shfl_down_sync(warp, value, offset);
    }
    return value;
}

__global__ void __launch_bounds__(BLOCK_X * BLOCK_Y)
accumulate_kernel(
    const uint2* __restrict__ ranges,
    const uint32_t* __restrict__ point_list,
    int width,
    int height,
    const float2* __restrict__ means2d,
    const float4* __restrict__ conic_opacity,
    const float* __restrict__ final_transmittance,
    const uint32_t* __restrict__ last_contributor,
    const uint32_t* __restrict__ mask_bits,
    int mask_count,
    float min_opacity,
    int point_count,
    float* __restrict__ visible,
    float* __restrict__ inside,
    int* __restrict__ valid_pixels) {
    auto block = cg::this_thread_block();
    const uint32_t horizontal_blocks = (width + BLOCK_X - 1) / BLOCK_X;
    const uint2 pix_min = {block.group_index().x * BLOCK_X, block.group_index().y * BLOCK_Y};
    const uint2 pix = {pix_min.x + block.thread_index().x, pix_min.y + block.thread_index().y};
    const bool in_image = pix.x < width && pix.y < height;
    const uint32_t pixel_id = width * pix.y + pix.x;
    const uint2 range = ranges[block.group_index().y * horizontal_blocks + block.group_index().x];
    const int rounds = (range.y - range.x + BLOCK_SIZE - 1) / BLOCK_SIZE;
    int remaining = range.y - range.x;

    __shared__ int gaussian_ids[BLOCK_SIZE];
    __shared__ float2 positions[BLOCK_SIZE];
    __shared__ float4 conics[BLOCK_SIZE];

    const float opacity = in_image ? 1.0f - final_transmittance[pixel_id] : 0.0f;
    bool done = !in_image || !isfinite(opacity) || opacity < min_opacity;
    if (in_image && !done) atomicAdd(valid_pixels, 1);
    const float inverse_opacity = done ? 0.0f : 1.0f / opacity;
    const uint32_t pixel_bits = done ? 0u : mask_bits[pixel_id];
    const uint32_t last = done ? 0u : last_contributor[pixel_id];
    const float2 pixel = {static_cast<float>(pix.x), static_cast<float>(pix.y)};
    float transmittance = in_image ? final_transmittance[pixel_id] : 0.0f;
    uint32_t contributor = range.y - range.x;

    for (int round = 0; round < rounds; ++round, remaining -= BLOCK_SIZE) {
        const int all_done = __syncthreads_count(done);
        if (all_done == BLOCK_SIZE) break;
        const int progress = round * BLOCK_SIZE + block.thread_rank();
        if (range.x + progress < range.y) {
            const int id = point_list[range.y - progress - 1];
            gaussian_ids[block.thread_rank()] = id;
            positions[block.thread_rank()] = means2d[id];
            conics[block.thread_rank()] = conic_opacity[id];
        }
        block.sync();

        for (int j = 0; j < min(BLOCK_SIZE, remaining); ++j) {
            --contributor;
            float normalized = 0.0f;
            bool contributes = !done && contributor < last;
            if (contributes) {
                const float2 delta = {positions[j].x - pixel.x, positions[j].y - pixel.y};
                const float4 conic = conics[j];
                const float power = -0.5f * (conic.x * delta.x * delta.x + conic.z * delta.y * delta.y)
                                  - conic.y * delta.x * delta.y;
                contributes = power <= 0.0f;
                if (contributes) {
                    const float alpha = min(0.99f, conic.w * expf(power));
                    contributes = alpha >= 1.0f / 255.0f;
                    if (contributes) {
                        transmittance = transmittance / (1.0f - alpha);
                        normalized = alpha * transmittance * inverse_opacity;
                    }
                }
            }
            const int id = gaussian_ids[j];

            // Every lane in a warp visits the same tile Gaussian at a given j.
            // Reduce equal-target updates within the warp before touching global
            // memory. This preserves float32 accumulation semantics while
            // replacing up to 32 contended atomics by one.
            const unsigned warp = __activemask();
            const int lane = static_cast<int>(block.thread_rank() & 31);
            const float visible_total = warp_sum(warp, contributes ? normalized : 0.0f);
            if (lane == 0 && visible_total != 0.0f) atomicAdd(&visible[id], visible_total);

            uint32_t active = contributes ? pixel_bits : 0u;
            while (__any_sync(warp, active != 0u)) {
                const unsigned nonempty = __ballot_sync(warp, active != 0u);
                const int source_lane = __ffs(nonempty) - 1;
                const int mask = __shfl_sync(warp, __ffs(active) - 1, source_lane);
                const uint32_t bit = 1u << mask;
                const bool member = (active & bit) != 0u;
                const float total = warp_sum(warp, member ? normalized : 0.0f);
                if (lane == 0 && total != 0.0f) atomicAdd(&inside[mask * point_count + id], total);
                if (member) active &= ~bit;
            }
        }
    }
}

void launch_accumulation(
    dim3 grid, dim3 block, const uint2* ranges, const uint32_t* point_list,
    int width, int height, const float2* means2d, const float4* conic_opacity,
    const float* final_transmittance, const uint32_t* last_contributor,
    const uint32_t* mask_bits, int mask_count, float min_opacity, int point_count,
    float* visible, float* inside, int* valid_pixels) {
    accumulate_kernel<<<grid, block>>>(
        ranges, point_list, width, height, means2d, conic_opacity,
        final_transmittance, last_contributor, mask_bits, mask_count,
        min_opacity, point_count, visible, inside, valid_pixels);
    const cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        throw std::runtime_error(cudaGetErrorString(error));
    }
}

}  // namespace

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
AccumulateAlphaMassCUDA(
    const torch::Tensor& means3D,
    const torch::Tensor& opacity,
    const torch::Tensor& scales,
    const torch::Tensor& rotations,
    float scale_modifier,
    const torch::Tensor& cov3D_precomp,
    const torch::Tensor& viewmatrix,
    const torch::Tensor& projmatrix,
    float tan_fovx,
    float tan_fovy,
    int image_height,
    int image_width,
    const torch::Tensor& campos,
    const torch::Tensor& packed_masks,
    int mask_count,
    float min_opacity,
    bool prefiltered,
    bool debug) {
    TORCH_CHECK(means3D.is_cuda() && means3D.scalar_type() == torch::kFloat32, "means3D must be CUDA float32");
    TORCH_CHECK(packed_masks.is_cuda() && packed_masks.scalar_type() == torch::kInt32, "packed masks must be CUDA int32 bitsets");
    TORCH_CHECK(packed_masks.numel() == image_height * image_width, "packed mask shape mismatch");
    TORCH_CHECK(mask_count >= 0 && mask_count <= 32, "one fused pass supports 0..32 masks");

    const int point_count = means3D.size(0);
    auto floats = means3D.options().dtype(torch::kFloat32);
    auto bytes = means3D.options().dtype(torch::kUInt8);
    auto ints = means3D.options().dtype(torch::kInt32);
    auto colors = torch::ones({point_count, 3}, floats);
    auto background = torch::zeros({3}, floats);
    auto output = torch::zeros({3, image_height, image_width}, floats);
    auto radii = torch::zeros({point_count}, ints);
    auto geometry_buffer = torch::empty({0}, bytes);
    auto binning_buffer = torch::empty({0}, bytes);
    auto image_buffer = torch::empty({0}, bytes);
    auto geometry = resize_tensor(geometry_buffer);
    auto binning = resize_tensor(binning_buffer);
    auto image = resize_tensor(image_buffer);

    const float* scale_ptr = scales.numel() ? scales.contiguous().data_ptr<float>() : nullptr;
    const float* rotation_ptr = rotations.numel() ? rotations.contiguous().data_ptr<float>() : nullptr;
    const float* covariance_ptr = cov3D_precomp.numel() ? cov3D_precomp.contiguous().data_ptr<float>() : nullptr;
    const int rendered = CudaRasterizer::Rasterizer::forward(
        geometry, binning, image, point_count, 0, 0,
        background.data_ptr<float>(), image_width, image_height,
        means3D.contiguous().data_ptr<float>(), nullptr, colors.data_ptr<float>(),
        opacity.contiguous().data_ptr<float>(), scale_ptr, scale_modifier,
        rotation_ptr, covariance_ptr, viewmatrix.contiguous().data_ptr<float>(),
        projmatrix.contiguous().data_ptr<float>(), campos.contiguous().data_ptr<float>(),
        tan_fovx, tan_fovy, prefiltered, output.data_ptr<float>(), radii.data_ptr<int>(), debug);

    // The registered gradient reference accumulates color gradients with
    // float32 atomicAdd. Matching that dtype is both materially faster and
    // avoids comparing a high-precision sum against a differently rounded
    // reference sum.
    auto visible = torch::zeros({point_count}, floats);
    auto inside = torch::zeros({mask_count, point_count}, floats);
    auto valid_pixels = torch::zeros({}, ints);
    if (point_count && rendered) {
        char* geometry_ptr = reinterpret_cast<char*>(geometry_buffer.data_ptr());
        char* binning_ptr = reinterpret_cast<char*>(binning_buffer.data_ptr());
        char* image_ptr = reinterpret_cast<char*>(image_buffer.data_ptr());
        auto geometry_state = CudaRasterizer::GeometryState::fromChunk(geometry_ptr, point_count);
        auto binning_state = CudaRasterizer::BinningState::fromChunk(binning_ptr, rendered);
        auto image_state = CudaRasterizer::ImageState::fromChunk(image_ptr, image_width * image_height);
        dim3 grid((image_width + BLOCK_X - 1) / BLOCK_X, (image_height + BLOCK_Y - 1) / BLOCK_Y, 1);
        dim3 block(BLOCK_X, BLOCK_Y, 1);
        launch_accumulation(
            grid, block, image_state.ranges, binning_state.point_list,
            image_width, image_height, geometry_state.means2D,
            geometry_state.conic_opacity, image_state.accum_alpha,
            image_state.n_contrib,
            reinterpret_cast<uint32_t*>(packed_masks.contiguous().data_ptr<int>()),
            mask_count, min_opacity, point_count, visible.data_ptr<float>(),
            inside.data_ptr<float>(), valid_pixels.data_ptr<int>());
    }
    return std::make_tuple(visible, inside, valid_pixels);
}
