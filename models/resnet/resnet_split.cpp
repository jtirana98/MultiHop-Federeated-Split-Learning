#include "resnet_split.h"
#include <typeinfo>

std::array<int64_t, 4> getLayers(resnet_model model_option) {
    switch (model_option) {
        std::array<int64_t, 4> layers;
    case resnet18:
        layers = std::array<int64_t, 4>({2, 2, 2, 2});
        return layers;
    case resnet34:
        layers = std::array<int64_t, 4>({3, 4, 6, 3});
        return layers;
    case resnet50:
        layers = std::array<int64_t, 4>({3, 4, 6, 3});
        return layers;
    case resnet101:
        layers = std::array<int64_t, 4>({3, 4, 23, 3});
        return layers;
    case resenet152:
        layers = std::array<int64_t, 4>({3, 8, 36, 3});
        return layers;

    }
}

std::vector<torch::nn::Sequential> resnet_split(const std::array<int64_t, 4>& layers, int64_t num_classes, 
    bool usebottleneck, const std::vector<int>& split_points, int in_channels) {

    bool downsample = false;
    int out_channels, blocks, stride;
    std::vector<torch::nn::Sequential> parts;
    bool split_every_point = (split_points.size() == 0), at_end = false;

    std::array<int64_t, 5> filter;
    if(usebottleneck)
        filter = std::array<int64_t, 5>({64, 256, 512, 1024, 2048});
    else
        filter = std::array<int64_t, 5>({64, 64, 128, 256, 512});

    int k = 1, l=0;
    auto part = torch::nn::Sequential();
    bool new_split = false;

    // layer 0
    part->push_back(torch::nn::Conv2d(torch::nn::Conv2dOptions(in_channels, 64, 7).stride(2).padding(3)));

    if (split_every_point || (!split_every_point && !at_end && split_points[l] == k)) {
        parts.push_back(part);
        new_split = true;
        l += 1;
        if (l == split_points.size())
            at_end = true;
    }
    k += 1;

    // layer 1
    if (new_split) {
        part = torch::nn::Sequential(); // clean
        new_split = false;
    }

    part->push_back(torch::nn::MaxPool2d(torch::nn::MaxPool2dOptions(3).stride(2).padding(1)));
    part->push_back(torch::nn::BatchNorm2d(torch::nn::BatchNorm2dOptions(64)));
    part->push_back(torch::nn::ReLU(true));

    if (split_every_point || (!split_every_point && !at_end && split_points[l] == k)) {
        parts.push_back(part);
        new_split = true;
        l += 1;
        
        if (l == split_points.size())
            at_end = true;
    }
    k += 1;


    // layers with res blocks
    for(int i=1; i<5; i++) {
        downsample = (i>1);
        in_channels = filter[i-1];
        out_channels = filter[i];
        blocks = layers[i-1];
        stride = (downsample) ? 2 : 1;
        if (in_channels != out_channels)
            downsample = true;

        for(int j=0; j != blocks; ++j){
            if (new_split) {
                part = torch::nn::Sequential(); // clean
                new_split = false;
            }
            if (usebottleneck)
                part->push_back(ResidualBottleneckBlock(in_channels, out_channels, stride, downsample));
            else
                part->push_back(ResidualBlock(in_channels, out_channels, stride, downsample));
            

            if (split_every_point || (!split_every_point && !at_end && split_points[l] == k)) {
                parts.push_back(part);
                new_split = true;
                l += 1;
                
                if (l == split_points.size())
                    at_end = true;
            }
            
            k += 1;

            if (j==0){
                downsample = false;
                in_channels = out_channels;
                stride = 1;
            }
        }
    }

    if (new_split) {
        part = torch::nn::Sequential(); // clean
        new_split = false;
    }

    part->push_back(torch::nn::AdaptiveAvgPool2d(torch::nn::AdaptiveAvgPool2dOptions(1)));

    if (split_every_point || (!split_every_point && !at_end && split_points[l] == k )) {
        parts.push_back(part);
        new_split = true;
        l += 1;

        if (l == split_points.size())
            at_end = true;
    }
    else { // split because we need to flatted input
        parts.push_back(part);
        new_split = true;
    }

    k += 1;

    // fully connected 1
    if (new_split) {
        part = torch::nn::Sequential(); // clean
        new_split = false;
    }
    int factor = (usebottleneck ? 4:1);
    part->push_back(torch::nn::Linear(torch::nn::LinearOptions(512*factor, num_classes)));
    //layer->push_back(torch::nn::ReLU(true));
    parts.push_back(part);                             

    return parts;
}

std::vector<torch::nn::Sequential> resnet_part(resnet_model model_option, int64_t num_classes, int start, int end, int in_channels) {
    auto layers_ = getLayers(model_option);
    bool usebottleneck = (model_option <=2) ? false : true;
    
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
    usebottleneck = false;
    auto parts = resnet_split(layers_, num_classes, usebottleneck, split_points, in_channels);
    
    int sum = 0;
    for (int i =0; i< parts.size(); i++) {
        sum = sum + parts[i]->size();
    }
    
    int first = 1;
    if (start == 0)
        first = 0;
    
    layers.push_back(parts[first]);
    
    if (end <sum-1 && end != -1) {
        return layers;
    }
    else {
        layers.push_back(parts[first + 1]);
    }
    
    return layers;
}