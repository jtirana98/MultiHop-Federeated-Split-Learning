#include "resBlock.h"

ResidualBlockImpl::ResidualBlockImpl(int64_t in_channels, int64_t out_channels, 
    int64_t stride, bool downsample) :
    conv1(torch::nn::Conv2dOptions(in_channels, out_channels, 3).stride(stride).padding(1).bias(false)),
    conv2(torch::nn::Conv2dOptions(in_channels, out_channels, 3).stride(1).padding(1).bias(false)),
    bn1(out_channels),
    bn2(out_channels),
    downsample(downsample) {

    register_module("conv1", conv1);
    register_module("conv2", conv2);
    register_module("relu", relu);
    register_module("bn1", bn1);
    register_module("bn2", bn2);

    if (downsample) {
        shortcut->push_back(torch::nn::Conv2d(torch::nn::Conv2dOptions(in_channels, out_channels, 1).stride(2).padding(1).bias(false)));
        shortcut->push_back(torch::nn::BatchNorm2d(out_channels));
        register_module("shortcut", shortcut);
    }
}

ResidualBlockImpl::forward(torch::Tensor x) {
    auto out = conv1->forward(x);
    out = bn1->forward(out);
    out = relu->forward(out);
    out = conv2->forward(out);
    out = bn2->forward(out);

    auto residual = downsampler ? downsampler->forward(x) : x;
    out += residual;
    out = relu->forward(out);

    return out;
}