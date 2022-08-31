#pragma once

#include <torch/torch.h>
#include <torch/data/datasets/base.h>
#include <torch/data/example.h>
#include <torch/types.h>
#include <iostream>
#include <vector>
#include <string>
#include <utility>
#include "resnet.h"

std::array<int64_t, 4> getLayers(resnet_model model_option);
std::vector<torch::nn::Sequential> resnet_split(const std::array<int64_t, 4>& layers, 
                                                int64_t num_classes, 
                                                bool usebottleneck=false, 
                                                const std::vector<int>& split_points = std::vector<int>());

std::vector<torch::nn::Sequential> resnet_part(resnet_model model_option, 
                    int64_t num_classes, int start, int end);