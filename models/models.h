#pragma once
#include <torch/torch.h>
#include <vector>

#include "vgg_train.h"
#include "resnet_train.h"
#include "lenet_train.h"


enum model_name {
    vgg,
    resnet,
    letnet
};

class ModelPart {
 public:   
    std::vector<torch::nn::Sequential> layers;

    ModelPart(model_name name, int model_, int start, int end, int num_classes) {
        switch (name) {
        case vgg:
            {
                vgg_model model_version_vgg = (vgg_model) model_;
                layers = vgg_part(model_version_vgg, num_classes, start-1, end);
            }
            break;
        case resnet:
            {
                resnet_model model_version_res = (resnet_model) model_;
                layers = resnet_part(model_version_res, num_classes, start-1, end);
            }
            break;
        case letnet:
            {
                layers = lenet_part(num_classes, start, end);
            }
            break;
        default:
            break;
        }
    }
};