#pragma once

#include <torch/torch.h>
#include <torch/data/datasets/base.h>
#include <torch/data/example.h>
#include <torch/types.h>
#include <iostream>
#include <vector>
#include <string>
#include <utility>


#include "resBlock.h"
#include "resBottleneckBlock.h"


std::vector<torch::nn::Sequential> resnet_split(const std::array<int64_t, 4>& layers, int64_t num_classes, 
                                                bool usebottleneck=false, const std::vector<int>& split_points = std::vector<int>());