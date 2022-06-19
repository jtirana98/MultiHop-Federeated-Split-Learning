#include <torch/torch.h>
#include <torch/data/datasets/base.h>
#include <torch/data/example.h>
#include <torch/types.h>
#include <iostream>
#include <vector>

#include "vgg_train.h"

int main(int argc, char **argv) {
    train_vgg(CIFAR_100, v11, true);

} 