#pragma once

#include <torch/torch.h>
#include <torch/data/datasets/base.h>
#include <torch/data/example.h>
#include <torch/types.h>
#include <iostream>
#include <vector>
#include <string>
#include <utility>

#include "mylogging.h"
#include "mydataset.h"
#include "transform.h"

using transform::ConstantPad;
using transform::RandomCrop;
using transform::RandomHorizontalFlip;

void split_cifar(std::vector<torch::nn::Sequential> layers, int type, int batch_size, int avg_point_, double learning_rate, int num_epochs);
void split_mnist(std::vector<torch::nn::Sequential> layers, int batch_size, int avg_point_, double learning_rate, int num_epochs);