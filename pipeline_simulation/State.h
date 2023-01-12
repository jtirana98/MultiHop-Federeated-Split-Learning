#pragma once

#include <vector>
#include <torch/torch.h>
#include <torch/data/datasets/base.h>
#include <torch/data/example.h>
#include <torch/types.h>

class State {
 public:
    int client_id;
    std::vector<torch::nn::Sequential> layers;
    std::vector<torch::Tensor> detached_activations;
    std::vector<torch::Tensor> activations;
    torch::Tensor received_activation; // received from prev node
    std::vector<torch::optim::SGD *> optimizers;
    double learning_rate = 0.1;
    
    //  State() {};

    State(int client_id, std::vector<torch::nn::Sequential> layers, std::vector<torch::optim::SGD *> optimizers) :
      client_id(client_id),
      layers(layers),
      optimizers(optimizers) {
    }
};