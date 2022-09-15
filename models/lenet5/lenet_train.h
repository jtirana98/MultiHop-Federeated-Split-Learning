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
#include "lenet.h"
#include "lenet_help.h"
#include "transform.h"

using transform::ConstantPad;
using transform::RandomCrop;
using transform::RandomHorizontalFlip;

const double l_learning_rate = 0.01;
const size_t l_num_epochs = 200;

void train_lenet(dataset dataset_option, 
                bool split, int batch_size = 64, 
                const std::vector<int>& split_points = std::vector<int>(), 
                bool test = false);