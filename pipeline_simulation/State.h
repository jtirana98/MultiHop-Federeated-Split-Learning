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
    std::vector<torch::optim::Adam *> optimizers;
    double learning_rate = 0.1;
    
    //  State() {};

    State(int client_id, std::vector<torch::nn::Sequential> layers, std::vector<torch::optim::Adam *> optimizers) :
      client_id(client_id),
      layers(layers),
      optimizers(optimizers) {
    }
   /*
    void init_layers(std::vector<torch::nn::Sequential> layers_) {
        layers.clear();
        optimizers.clear();
        for (int i = 0; i < layers_.size(); i++) {
            layers.push_back(layers_[i]);
            optimizers.push_back(torch::optim::Adam(layers[i]->parameters(), torch::optim::AdamOptions(learning_rate)));
        }

        std::cout << " " << optimizers.size() << std::endl;
    }
    */

};