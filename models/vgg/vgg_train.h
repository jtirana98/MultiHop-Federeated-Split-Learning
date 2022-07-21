#pragma once
#include <torch/torch.h>
#include <torch/data/datasets/base.h>
#include <torch/data/example.h>
#include <torch/types.h>
#include <iostream>
#include <vector>

#include "mylogging.h"
#include "mydataset.h"
#include "split_training.h"
#include "vgg.h"
#include "vgg_help.h"
#include "transform.h"

using transform::ConstantPad;
using transform::RandomCrop;
using transform::RandomHorizontalFlip;

//const int64_t num_classes = 10;
const int64_t g_batch_size = 128;
const double learning_rate = 0.01;
const size_t num_epochs = 200;

void train_vgg(dataset dataset_option, vgg_model model_option, bool split, int batch_size = g_batch_size, const std::vector<int>& split_points = std::vector<int>(), bool test = false);