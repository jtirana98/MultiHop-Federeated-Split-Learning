#include "resnet_split.h"
#include <typeinfo>

std::vector<torch::nn::Sequential> resnet_split(const std::array<int64_t, 4>& layers, int64_t num_classes, 
    bool usebottleneck, const std::vector<int>& split_points) {

    bool downsample = false;
    int in_channels, out_channels, blocks, stride;
    std::vector<torch::nn::Sequential> parts;
    bool split_every_point = (split_points.size() == 0), at_end = false;

    std::array<int64_t, 5> filter;
        if(usebottleneck)
            filter = std::array<int64_t, 5>({64, 256, 512, 1024, 2048});
        else
            filter = std::array<int64_t, 5>({64, 64, 128, 256, 512});

    int k = 0, l=0;
    auto part = torch::nn::Sequential();
    bool new_split = true;

    // layer 0
    part->push_back(torch::nn::Conv2d(torch::nn::Conv2dOptions(3, 64, 7).stride(2).padding(3)));

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
    else { // TODO! split because we need to flatted input
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