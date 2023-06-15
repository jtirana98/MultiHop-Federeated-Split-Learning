#pragma once

#include <torch/torch.h>
#include <torch/data/datasets/base.h>
#include <torch/data/example.h>
#include <torch/types.h>
#include <iostream>
#include <vector>
#include <string>
#include "vgg.h"

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

static std::vector<int> typeA{64, -1, 128, -1, 256, 256, -1, 512, 512, -1, 512, 512, -1};
static std::vector<int> typeB{64, 64, -1, 128, 128, -1, 256, 256, -1, 512, 512, -1, 512, 512, -1};
static std::vector<int> typeD{64, 64, -1, 128, 128, -1, 256, 256, 256, -1, 512, 512, 512, -1, 512, 512, 512, -1};
static std::vector<int> typeE{64, 64, -1, 128, 128, -1, 256, 256, 256, 256, -1, 512, 512, 512, 512, -1, 512, 512, 512, 512, -1};


static std::unordered_map<std::string,std::vector<int>> table = { { "A", typeA}, {"B", typeB}, 
                                                                 { "D", typeD}, {"E", typeE} };

torch::nn::Sequential make_layers(std::string cfg, bool batch_norm, int in_channels);
VGG _vgg(std::string cfg, bool batch_norm, int num_classes, int in_channels);

// model types
VGG vgg11(int num_classes, int in_channels=3);
VGG vgg11_bn(int num_classes, int in_channels=3);
VGG vgg13(int num_classes, int in_channels=3);
VGG vgg13_bn(int num_classes, int in_channels=3);
VGG vgg16(int num_classes, int in_channels=3);
VGG vgg16_bn(int num_classes, int in_channels=3);
VGG vgg19(int num_classes, int in_channels=3);
VGG vgg19_bn(int num_classes, int in_channels=3);

// split models
std::vector<torch::nn::Sequential> _vgg_split(std::string cfg, bool batch_norm, int num_classes, const std::vector<int>& split_points = std::vector<int>(), int in_channels=3);
std::vector<torch::nn::Sequential> vgg11_split(int num_classes, const std::vector<int>& split_points = std::vector<int>(), int in_channels=3);
//VGG vgg11_bn(int num_classes);
std::vector<torch::nn::Sequential> vgg13_split(int num_classes, const std::vector<int>& split_points = std::vector<int>(), int in_channels=3);
//VGG vgg13_bn(int num_classes);
std::vector<torch::nn::Sequential> vgg16_split(int num_classes, const std::vector<int>& split_points = std::vector<int>(), int in_channels=3);
//VGG vgg16_bn(int num_classes);
std::vector<torch::nn::Sequential> vgg19_split(int num_classes, const std::vector<int>& split_points = std::vector<int>(), int in_channels=3);
//VGG vgg19_bn(int num_classes);

// get parts
std::vector<torch::nn::Sequential> vgg_part(vgg_model model, int num_classes, int start, int end, int in_channels=3);

