#include <torch/torch.h>
#include <torch/data/datasets/base.h>
#include <torch/data/example.h>
#include <torch/types.h>
#include <iostream>
#include <vector>
#include <string>

#include "vgg_train.h"

int main(int argc, char **argv) {
    /*
    std::vector<int> splits{2, 4, 6, 8};
    std::vector<std::string> to_print{"vgg11", "vgg13", "vgg16", "vgg19"};
    std::vector<int> batches{32, 64, 128};
    std::vector<vgg_model> models{v11, v13, v16, v19};
    
    for(int i = 0; i < models.size(); i++) {
        for(int j = 0; j < batches.size(); j++) {
            std::cout << to_print[i] << " " << batches[j] << std::endl;
            train_vgg(CIFAR_10, models[i], false, batches[j]);
            std::cout << std::endl;
        }
    }
    
    */
   train_vgg(CIFAR_10, v11, false, 64);
} 