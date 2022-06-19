#include <torch/torch.h>
#include <torch/data/datasets/base.h>
#include <torch/data/example.h>
#include <torch/types.h>
#include <iostream>
#include <vector>

#include "vgg_train.h"

int main(int argc, char **argv) {
    std::vector<int> splits{2, 4, 6, 8};
    train_vgg(CIFAR_10, v11, true, splits);

} 