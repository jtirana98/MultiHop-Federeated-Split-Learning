#pragma once

#include <iostream> 
#include <vector>
#include <torch/torch.h>
#include <torch/data/datasets/base.h>
#include <torch/data/example.h>
#include <torch/types.h>

class State {
 public:
    int client_id;
    std::vector<torch::nn::Sequential> layers;
    torch::Tensor prev_activations;
    
    State(int client_id, std::vector<torch::nn::Sequential> layers) :
        client_id(client_id),
        layers(layers) {}
};