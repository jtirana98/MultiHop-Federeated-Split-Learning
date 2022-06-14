#pragma once

#include <torch/torch.h>


class VGGImpl : public torch::nn::Module {
  public:
    VGGImpl(torch::nn::Sequential features, int num_classes, double dropout);
    torch::Tensor forward(torch::Tensor x);
    
  private:
    torch::nn::Sequential features;
    torch::nn::Sequential avgpool;
    torch::nn::Sequential classifier;
};

TORCH_MODULE(VGG);