#include "vgg_help.h"
#include <typeinfo>

torch::nn::Sequential make_layers(std::string cfg, bool batch_norm) {
    auto layers = torch::nn::Sequential();
    int in_channels = 3;
    auto description = table.find(cfg)->second;

    for(int d : description) {
        if (d  != -1){ 
            layers->push_back(torch::nn::Conv2d(torch::nn::Conv2dOptions(in_channels, d, 3).padding(1)));
            if (batch_norm)
                layers->push_back(torch::nn::BatchNorm2d(d));
            layers->push_back(torch::nn::ReLU(true));
            in_channels = d;
        }
        else{
            layers->push_back(torch::nn::MaxPool2d(torch::nn::MaxPool2dOptions(2).stride(2).padding(1)));
        }
        
    }

    return layers;
}


VGG _vgg(std::string cfg, bool batch_norm, int num_classes) {
    auto features = make_layers(cfg, batch_norm);
    auto model =  VGG(features, num_classes, 0.5);
    return model;
}

// model types
VGG vgg11(int num_classes) {
    return _vgg("A", false, num_classes);
}

VGG vgg11_bn(int num_classes) {
    return _vgg("A", true, num_classes);
}

VGG vgg13(int num_classes) {
    return _vgg("B", false, num_classes);
}

VGG vgg13_bn(int num_classes) {
    return _vgg("B", true, num_classes);
}

VGG vgg16(int num_classes) {
    return _vgg("D", false, num_classes);
}

VGG vgg16_bn(int num_classes) {
    return _vgg("D", true, num_classes);
}

VGG vgg19(int num_classes) {
    return _vgg("E", false, num_classes);
}

VGG vgg19_bn(int num_classes) {
    return _vgg("E", true, num_classes);
}

// split models


std::vector<torch::nn::Sequential> _vgg_split(std::string cfg, bool batch_norm, int num_classes) {
    std::vector<torch::nn::Sequential> layers;

    int in_channels = 3;
    auto description = table.find(cfg)->second;
    double dropout = 0.5;

    for(int d : description) {
        auto layer = torch::nn::Sequential();
        if (d  != -1){ 
            layer->push_back(torch::nn::Conv2d(torch::nn::Conv2dOptions(in_channels, d, 3).padding(1)));
            if (batch_norm)
                layer->push_back(torch::nn::BatchNorm2d(d));
            layer->push_back(torch::nn::ReLU(true));
            in_channels = d;
        }
        else{
            layer->push_back(torch::nn::MaxPool2d(torch::nn::MaxPool2dOptions(2).stride(2).padding(1)));
        }
        layers.push_back(layer);
    }

    layers.push_back(torch::nn::Sequential(torch::nn::AdaptiveAvgPool2d(torch::nn::AdaptiveAvgPool2dOptions({7,7}))));
    layers.push_back(torch::nn::Sequential(torch::nn::Linear(torch::nn::LinearOptions(512 * 7 * 7, 4096)),
                                            torch::nn::ReLU(true),
                                            torch::nn::Dropout(dropout)));

    layers.push_back(torch::nn::Sequential(torch::nn::Linear(torch::nn::LinearOptions(4096, 4096)),
                                           torch::nn::ReLU(true)));
    layers.push_back(torch::nn::Sequential(torch::nn::Dropout(dropout),
                                           torch::nn::Linear(torch::nn::LinearOptions(4096, num_classes))));

    return layers;
}

std::vector<torch::nn::Sequential> vgg11_split(int num_classes) {
    return _vgg_split("A", false, num_classes);
}


std::vector<torch::nn::Sequential> vgg13_split(int num_classes) {
    return _vgg_split("B", false, num_classes);
}

std::vector<torch::nn::Sequential> vgg16_split(int num_classes) {
    return _vgg_split("D", false, num_classes);
}

std::vector<torch::nn::Sequential> vgg19_split(int num_classes) {
    return _vgg_split("E", false, num_classes);
}