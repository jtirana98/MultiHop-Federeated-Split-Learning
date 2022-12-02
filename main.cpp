#include <torch/torch.h>
#include <torch/data/datasets/base.h>
#include <torch/data/example.h>
#include <torch/types.h>
#include <iostream>
#include <vector>
#include <string>

#include "models.h"

using torch::indexing::Slice;
using torch::indexing::None;
using torch::indexing::Ellipsis;

int main(int argc, char **argv) {

    train_resnet(CIFAR_10, resnet101, false, 128, std::vector<int>(), true, true);
} 