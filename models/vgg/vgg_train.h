#pragma once
#include <torch/torch.h>
#include <torch/data/datasets/base.h>
#include <torch/data/example.h>
#include <torch/types.h>
#include <iostream>
#include <vector>

#include "mylogging.h"
#include "mydataset.h"
#include "vgg.h"
#include "vgg_help.h"
#include "transform.h"

using transform::ConstantPad;
using transform::RandomCrop;
using transform::RandomHorizontalFlip;

//const int64_t num_classes = 10;
const int64_t batch_size = 128;
const double learning_rate = 0.001;
const size_t num_epochs = 1;

enum vgg_model{
    v11,
    v11_bn,
    v13,
    v13_bn,
    v16,
    v16_bn,
    v19,
    v19_bn
};

void train_vgg(dataset dataset_option, vgg_model model_option, bool split, const std::vector<int>& split_points = std::vector<int>());