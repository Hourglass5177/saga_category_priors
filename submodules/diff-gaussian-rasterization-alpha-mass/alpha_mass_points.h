#pragma once

#include <torch/extension.h>
#include <tuple>

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
    bool debug);
