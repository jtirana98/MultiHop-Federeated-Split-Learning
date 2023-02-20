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


std::vector<torch::nn::Sequential> _vgg_split(std::string cfg, bool batch_norm, int num_classes, const std::vector<int>& split_points) {
    std::vector<torch::nn::Sequential> layers;
    int in_channels = 3;
    auto description = table.find(cfg)->second;
    double dropout = 0.5;
    bool split_every_point = false, at_end=false;

    if (split_points.size() == 0) { //we split at each layer
        split_every_point = true;
    }

    int k = 1, l=0;
    auto layer = torch::nn::Sequential();
    bool new_split = true;
    for(int d : description) {
        if (new_split) {
            layer = torch::nn::Sequential(); // clean
            new_split = false;
        }

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
        if (split_every_point || (!split_every_point && !at_end && split_points[l] == k)) {
            layers.push_back(layer);
            new_split = true;
            l += 1;
            
            if (l == split_points.size())
                at_end = true;
        }
        
        k += 1;
    }

    // avg pool
    if (new_split) {
        layer = torch::nn::Sequential(); // clean
        new_split = false;
    }

    layer->push_back(torch::nn::AdaptiveAvgPool2d(torch::nn::AdaptiveAvgPool2dOptions({7,7})));

    if (split_every_point || (!split_every_point && !at_end && split_points[l] == k )) {
        layers.push_back(layer);
        new_split = true;
        l += 1;

        if (l == split_points.size())
            at_end = true;
    }
    else { // TODO: split because we need to flatted input
        layers.push_back(layer);
        new_split = true;
    }

    k += 1;

    // fully connected 1
    if (new_split) {
        layer = torch::nn::Sequential(); // clean
        new_split = false;
    }

    layer->push_back(torch::nn::Linear(torch::nn::LinearOptions(512 * 7 * 7, 4096)));
    layer->push_back(torch::nn::ReLU(true));
    
    if (split_every_point || (!split_every_point && !at_end && split_points[l] == k)) {
        layers.push_back(layer);
        new_split = true;
        l += 1;

        if (l == split_points.size())
            at_end = true;
    }

    k += 1;

    // fully connected 2

    if (new_split) {
        layer = torch::nn::Sequential(); // clean
        new_split = false;
    }

    layer->push_back(torch::nn::Dropout(dropout));
    layer->push_back(torch::nn::Linear(torch::nn::LinearOptions(4096, 4096)));
    layer->push_back(torch::nn::ReLU(true));

    if (split_every_point || (!split_every_point && !at_end && split_points[l] == k)) {
        layers.push_back(layer);
        new_split = true;
        l += 1;
        if (l == split_points.size())
            at_end = true;
    }

    k += 1;

    // fully connected 3 -- output layer

    if (new_split) {
        layer = torch::nn::Sequential(); // clean
        new_split = false;
    }

    layer->push_back(torch::nn::Dropout(dropout));
    layer->push_back(torch::nn::Linear(torch::nn::LinearOptions(4096, num_classes)));

    layers.push_back(layer);


    return layers;
}

std::vector<torch::nn::Sequential> vgg11_split(int num_classes, const std::vector<int>& split_points) {
    return _vgg_split("A", false, num_classes, split_points);
}


std::vector<torch::nn::Sequential> vgg13_split(int num_classes, const std::vector<int>& split_points) {
    return _vgg_split("B", false, num_classes, split_points);
}

std::vector<torch::nn::Sequential> vgg16_split(int num_classes, const std::vector<int>& split_points) {
    return _vgg_split("D", false, num_classes, split_points);
}

std::vector<torch::nn::Sequential> vgg19_split(int num_classes, const std::vector<int>& split_points) {
    return _vgg_split("E", false, num_classes, split_points);
}


std::vector<torch::nn::Sequential> vgg_part(vgg_model model, int num_classes, int start, int end) {
    std::string cfg;
    bool batch_norm = false;
    std::vector<torch::nn::Sequential> layers;
    std::vector<int> split_points;

    if (start == 0) {
        split_points.push_back(end);
    }
    else if (end == -1) {
        split_points.push_back(start);
    }
    else{
        split_points.push_back(start);
        split_points.push_back(end);
    }

    switch (model) {
    case v11:
        cfg = "A";
        break;
    case v13:
        cfg = "B";
        break;
    case v16:
        cfg = "D";
        break;
    case v19:
        cfg = "E";
    default:
        break;
    }

    auto parts = _vgg_split(cfg, false, num_classes, split_points);
    int sum=0;
    for (int i =0; i< parts.size(); i++) {
        sum = sum + parts[i]->size();
    }
    int first = 1;
    if (start == 0)
        first = 0;
    
    std::cout << start << " " << end << " " << parts.size() << std::endl;
    if (end > 12 || (end == -1 && start <= 12)) {
        layers.push_back(parts[first]);
        layers.push_back(parts[first + 1]);
    }
    else{
        if (end == -1 && end < 12)
            first = parts.size() - 1;
        layers.push_back(parts[first]);
    }
    
    return layers;
}
