#pragma once
#include <torch/torch.h>
#include <torch/data/datasets/base.h>
#include <torch/data/example.h>
#include <torch/types.h>
#include <iostream>
#include <vector>

#include "mylogging.h"
#include "mydataset.h"
#include "resenet.h"
#include "transform.h"

using transform::ConstantPad;
using transform::RandomCrop;
using transform::RandomHorizontalFlip;

//const int64_t num_classes = 10;
const int64_t g_batch_size = 128;
const double learning_rate = 0.01;
const size_t num_epochs = 200;

enum resnet_model{
    resnet18,
    resnet34,
    resnet50,
    resnet101,
    resenet152
};

void train_resnet(dataset dataset_option, resenet_model model_option, bool split, int batch_size = g_batch_size, const std::vector<int>& split_points = std::vector<int>(), bool test = false);