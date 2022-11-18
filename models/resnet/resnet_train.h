#pragma once
#include <torch/torch.h>
#include <torch/data/datasets/base.h>
#include <torch/data/example.h>
#include <torch/types.h>
#include <iostream>
#include <vector>
#include <string>

#include "mylogging.h"
#include "split_training.h"
#include "mydataset.h"
#include "resnet.h"
#include "resnet_split.h"
#include "transform.h"

using transform::ConstantPad;
using transform::RandomCrop;
using transform::RandomHorizontalFlip;

//const int64_t num_classes = 10;
//const int64_t g_batch_size = 128;
const double r_learning_rate = 0.01;
const size_t r_num_epochs = 100;

void train_resnet(dataset dataset_option, resnet_model model_option, bool split, int batch_size = 64, const std::vector<int>& split_points = std::vector<int>(), bool test = false);