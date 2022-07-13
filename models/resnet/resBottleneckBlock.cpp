#include "resBottleneckBlock.h"

ResidualBottleneckBlockImpl::ResidualBottleneckBlockImpl(int64_t in_channels, int64_t out_channels, int64_t stride, bool downsample) :
    conv1(torch::nn::Conv2dOptions(in_channels, out_channels/4, 1).stride(1).bias(false)),
    conv2(torch::nn::Conv2dOptions(out_channels/4, out_channels/4, 3).stride(stride).padding(1).bias(false)),
    conv3(torch::nn::Conv2dOptions(out_channels/4, out_channels, 1).stride(1).bias(false)),
    bn1(out_channels/4),
    bn2(out_channels/4),
    bn3(out_channels),
    downsample(downsample) {

    register_module("conv1", conv1);
    register_module("conv2", conv2);
    register_module("conv3", conv3);
    register_module("relu", relu);
    register_module("bn1", bn1);
    register_module("bn2", bn2);
    register_module("bn3", bn3);
    
    shortcut = torch::nn::Sequential();
    if (downsample || (in_channels != out_channels)) {
        shortcut->push_back(torch::nn::Conv2d(torch::nn::Conv2dOptions(in_channels, out_channels, 1).stride(stride).bias(false)));
        shortcut->push_back(torch::nn::BatchNorm2d(out_channels));
        register_module("shortcut", shortcut);
    }
}

torch::Tensor ResidualBottleneckBlockImpl::forward(torch::Tensor x) {
    auto out = conv1->forward(x);
    out = bn1->forward(out);
    out = relu->forward(out);
    out = conv2->forward(out);
    out = bn2->forward(out);
    out = relu->forward(out);
    out = conv3->forward(out);
    out = bn3->forward(out);
    auto residual = downsample ? shortcut->forward(x) : x;
    out += residual;
    out = relu->forward(out);

    return out;
}