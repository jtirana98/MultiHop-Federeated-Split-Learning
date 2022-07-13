#pragma once
#include <torch/torch.h>
#include <vector>
#include "resBlock.h"
#include "resBottleneckBlock.h"

template<typename Block> struct ResNet : torch::nn::Module {
    int64_t in_channels = 16;
    torch::nn::Sequential layer0;
    torch::nn::Sequential layer1;
    torch::nn::Sequential layer2;
    torch::nn::Sequential layer3;
    torch::nn::Sequential layer4;
    torch::nn::AdaptiveAvgPool2d avg_pool{1};
    torch::nn::Linear fc;
    

    
    ResNet(const std::array<int64_t, 4>& layers, int64_t num_classes, bool usebottleneck=false) :
        fc(512*(usebottleneck ? 4:1), num_classes)
     {
        std::array<int64_t, 5> filter;
        if(usebottleneck)
            filter = std::array<int64_t, 5>({64, 256, 512, 1024, 2048});
        else
            filter = std::array<int64_t, 5>({64, 64, 128, 256, 512});

        layer0->push_back(torch::nn::Conv2d(torch::nn::Conv2dOptions(3, 64, 7).stride(2).padding(3)));
        layer0->push_back(torch::nn::MaxPool2d(torch::nn::MaxPool2dOptions(3).stride(2).padding(1)));
        layer0->push_back(torch::nn::BatchNorm2d(torch::nn::BatchNorm2dOptions(64)));
        layer0->push_back(torch::nn::ReLU(true));
        
        layer1 = _make_layer(filter[0], filter[1], layers[0], false);
        layer2 = _make_layer(filter[1], filter[2], layers[1], true);
        layer3 = _make_layer(filter[2], filter[3], layers[2], true);
        layer4 = _make_layer(filter[3], filter[4], layers[3], true);
        
        register_module("layer0", layer0);
        register_module("layer1", layer1);
        register_module("layer2", layer2);
        register_module("layer3", layer3);
        register_module("layer4", layer4);
        register_module("avg_pool", avg_pool);
        register_module("fc", fc);
    }

    torch::Tensor forward(torch::Tensor x) {
        auto out = layer0->forward(x);
        out = layer1->forward(out);
        out = layer2->forward(out);
        out = layer3->forward(out);
        out = layer4->forward(out);
        out = avg_pool->forward(out);
        std::cout << "-"<< out.sizes() << "\t";
        out = out.view({out.size(0), -1});
        return fc->forward(out);
    }
    
    torch::nn::Sequential _make_layer(int64_t in_channels, int64_t out_channels, int64_t blocks, bool downsample) {
        torch::nn::Sequential layers;
        int stride = downsample ? 2 : 1;

        if (in_channels != out_channels)
            downsample = true;
        
        layers->push_back(Block(in_channels, out_channels, stride, downsample));

        in_channels = out_channels;

        for (int64_t i = 1; i != blocks; ++i) {
            layers->push_back(Block(out_channels, out_channels, 1, false));
        }

        return layers;
    }

};

