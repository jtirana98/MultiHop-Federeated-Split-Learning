#include <torch/torch.h>
#include <torch/data/datasets/base.h>
#include <torch/data/example.h>
#include <torch/types.h>
#include <iostream>
#include <vector>
#include <string>

#include "models.h"

//#define VGG
//#define RESNET
#define LENET

int main(int argc, char **argv) {
    std::vector<int> batches{32, 64, 128};
    std::vector<int> splits{2, 4, 6, 8};
    

    // VGG
    #ifdef VGG
    std::vector<std::string> to_print_vgg{"vgg11", "vgg13", "vgg16", "vgg19"};
    std::vector<vgg_model> models_vgg{v11, v13, v16, v19};
 
    for(int i = 0; i < models_vgg.size(); i++) {
        for(int j = 0; j < batches.size(); j++) {
            std::cout << to_print_vgg[i] << " " << batches[j] << std::endl;
            train_vgg(CIFAR_10, models_vgg[i], false, batches[j]);
            std::cout << std::endl;
            
            std::cout << "SPLIT" << std::endl;
            train_vgg(CIFAR_10, models_vgg[i], true, batches[j]);
            std::cout << std::endl;
            
        }
    }
    #endif

    // ResNet
    #ifdef RESNET
    std::vector<std::string> to_print_resnet{"resnet18", "resnet34", 
                                             "resnet50", "resnet101", "resenet152"};

    std::vector<resnet_model> models_resnet{resnet18, resnet34, 
                                            resnet50, resnet101, resenet152};
    
    for(int i = 0; i < models_resnet.size(); i++) {
        for(int j = 0; j < batches.size(); j++) {
            std::cout << to_print_resnet[i] << " " << batches[j] << std::endl;
            train_resnet(CIFAR_10, models_resnet[i], false, batches[j]);
            std::cout << std::endl;

            std::cout << "SPLIT" << std::endl;
            train_resnet(CIFAR_10, models_resnet[i], true, batches[j]);
            std::cout << std::endl;
        }
    }
    
    #endif

    // LeNet5
    #ifdef LENET
    std::vector<std::string> to_print_lenet{"LeNet-5"};
    
    for(int j = 0; j < batches.size(); j++) {
        std::cout << to_print_lenet[0] << " " << batches[j] << std::endl;
        /*
        train_lenet(CIFAR_10, false, batches[j]);
        std::cout << std::endl;
        
        std::cout << "SPLIT" << std::endl;
        train_lenet(CIFAR_10, true, batches[j]);
        std::cout << std::endl;
        */

       train_lenet(CIFAR_10, false, batches[j], std::vector<int>(), true);
    }
    
    
    #endif                               

} 