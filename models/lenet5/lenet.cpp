#include "lenet.h"

static std::vector<int64_t> k_size = {2, 2};
static std::vector<int64_t> p_size = {0, 0};
static c10::optional<int64_t> divisor_override;

LeNet5Impl::LeNet5Impl(int num_classes, int in_channels) {
  conv_ = torch::nn::Sequential(
      torch::nn::Conv2d(torch::nn::Conv2dOptions(in_channels, 6, 5)), // layer 1
      torch::nn::Functional(torch::tanh), 
      torch::nn::AvgPool2d(torch::nn::AvgPool2dOptions( // layer 2
            torch::nn::AvgPool2dOptions(k_size).stride(k_size).padding(p_size).divisor_override(divisor_override))), 
      torch::nn::Conv2d(torch::nn::Conv2dOptions(6, 16, 5)),  // layer 3
      torch::nn::Functional(torch::tanh),
      torch::nn::AvgPool2d(torch::nn::AvgPool2dOptions( // layer 4
        torch::nn::AvgPool2dOptions(k_size).stride(k_size).padding(p_size).divisor_override(divisor_override))),
      torch::nn::Conv2d(torch::nn::Conv2dOptions(16, 120, 5)), // layer 5
      torch::nn::Functional(torch::tanh));
  register_module("conv", conv_);

  full_ = torch::nn::Sequential(
      torch::nn::Linear(torch::nn::LinearOptions(120, 84)), // layer 6
      torch::nn::Functional(torch::tanh),
      torch::nn::Linear(torch::nn::LinearOptions(84, num_classes)), // layer 7
      torch::nn::LogSoftmax(torch::nn::LogSoftmaxOptions(-1))); 
  register_module("full", full_);
}

torch::Tensor LeNet5Impl::forward(at::Tensor x) {
  auto output = conv_->forward(x);
  output = output.view({x.size(0), -1});
  output = full_->forward(output);
  return output;
}
