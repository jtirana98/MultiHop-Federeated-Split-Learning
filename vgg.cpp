#include "vgg.h"
#include <torch/torch.h>

VGGImpl::VGGImpl(torch::nn::Sequential features, int num_classes, double dropout) :
    avgpool(torch::nn::AdaptiveAvgPool2d(torch::nn::AdaptiveAvgPool2dOptions({7,7}))),
    classifier(torch::nn::Linear(torch::nn::LinearOptions(512 * 7 * 7, 4096)),
            torch::nn::ReLU(true),
            torch::nn::Dropout(dropout),
            torch::nn::Linear(torch::nn::LinearOptions(4096, 4096)),
            torch::nn::ReLU(true),
            torch::nn::Dropout(dropout),
            torch::nn::Linear(torch::nn::LinearOptions(4096, num_classes))),
    features(features)
{
    register_module("features", features);
    register_module("avgpool", avgpool);
    register_module("classifier", classifier);

    //std::cout << "features: " << features << std::endl;
    //std::cout << "avgpool: " << avgpool << std::endl;
    //std::cout << "classifier: " << classifier << std::endl; 
}

torch::Tensor VGGImpl::forward(torch::Tensor x) {
    auto output = features->forward(x);
    output = avgpool->forward(output);
    output = output.view({x.size(0), -1});
    output = classifier->forward(output);
    return output;
}
