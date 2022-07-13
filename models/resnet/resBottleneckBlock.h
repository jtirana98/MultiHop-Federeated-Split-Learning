#pragma once

#include <torch/torch.h>

class ResidualBottleneckBlockImpl : public torch::nn::Module {
 public:
    ResidualBottleneckBlockImpl(int64_t in_channels, int64_t out_channels, int64_t stride = 1,
        bool downsample = false);
    torch::Tensor forward(torch::Tensor x);
    static const int expansion = 4; 
 private:
    torch::nn::Conv2d conv1;
    torch::nn::BatchNorm2d bn1;
    torch::nn::ReLU relu;
    torch::nn::Conv2d conv2;
    torch::nn::BatchNorm2d bn2;
    torch::nn::Conv2d conv3;
    torch::nn::BatchNorm2d bn3;
    torch::nn::Sequential shortcut;
    bool downsample;
};


TORCH_MODULE(ResidualBottleneckBlock);
