#pragma once

#include <torch/torch.h>
#include <torch/data/datasets/base.h>
#include <torch/data/example.h>
#include <torch/types.h>
#include <iostream>
#include <vector>
#include <string>
#include <utility>
#include "lenet.h"

std::vector<torch::nn::Sequential> lenet_split( int64_t num_classes, 
                                                const std::vector<int>& split_points = std::vector<int>());

std::vector<torch::nn::Sequential> lenet_part(int64_t num_classes, int start, int end);