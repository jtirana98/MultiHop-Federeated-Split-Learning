#include "lenet_help.h"

static std::vector<int64_t> k_size = {2, 2};
static std::vector<int64_t> p_size = {0, 0};
static c10::optional<int64_t> divisor_override;

std::vector<torch::nn::Sequential> lenet_split( int64_t num_classes, 
                                                const std::vector<int>& split_points) {

    std::vector<torch::nn::Sequential> parts;
    bool split_every_point = (split_points.size() == 0), at_end = false;

    int k = 1, l=0;
    auto part = torch::nn::Sequential();
    bool new_split = false;


    // layer 1
    part->push_back(torch::nn::Conv2d(torch::nn::Conv2dOptions(1, 6, 5)));
    part->push_back(torch::nn::Functional(torch::tanh));

    if (split_every_point || (!split_every_point && !at_end && split_points[l] == k)) {
        parts.push_back(part);
        new_split = true;
        l += 1;
        if (l == split_points.size())
            at_end = true;
    }
    k += 1;

    // layer 2

    if (new_split) {
        part = torch::nn::Sequential();
        new_split = false;
    }

     part->push_back(torch::nn::AvgPool2d(torch::nn::AvgPool2dOptions(
                                        torch::nn::AvgPool2dOptions(k_size).stride(k_size).padding(p_size).divisor_override(divisor_override))));

    
    if (split_every_point || (!split_every_point && !at_end && split_points[l] == k)) {
        parts.push_back(part);
        new_split = true;
        l += 1;
        if (l == split_points.size())
            at_end = true;
    }
    k += 1;

    // layer 3

    if (new_split) {
        part = torch::nn::Sequential();
        new_split = false;
    }

    part->push_back(torch::nn::Conv2d(torch::nn::Conv2dOptions(6, 16, 5)));
    part->push_back(torch::nn::Functional(torch::tanh));

    if (split_every_point || (!split_every_point && !at_end && split_points[l] == k)) {
        parts.push_back(part);
        new_split = true;
        l += 1;
        if (l == split_points.size())
            at_end = true;
    }
    k += 1;

    // layer 4

    if (new_split) {
        part = torch::nn::Sequential();
        new_split = false;
    }

    part->push_back(torch::nn::AvgPool2d(torch::nn::AvgPool2dOptions(
                                        torch::nn::AvgPool2dOptions(k_size).stride(k_size).padding(p_size).divisor_override(divisor_override))));
    
    if (split_every_point || (!split_every_point && !at_end && split_points[l] == k)) {
        parts.push_back(part);
        new_split = true;
        l += 1;
        if (l == split_points.size())
            at_end = true;
    }
    k += 1;

    // layer 5

    if (new_split) {
        part = torch::nn::Sequential();
        new_split = false;
    }

    part->push_back(torch::nn::Conv2d(torch::nn::Conv2dOptions(16, 120, 5)));
    part->push_back(torch::nn::Functional(torch::tanh));
    

    if (split_every_point || (!split_every_point && !at_end && split_points[l] == k)) {
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

    // layer 6

    if (new_split) {
        part = torch::nn::Sequential();
        new_split = false;
    }

    part->push_back(torch::nn::Linear(torch::nn::LinearOptions(120, 84)));
    part->push_back(torch::nn::Functional(torch::tanh));

    
    if (split_every_point || (!split_every_point && !at_end && split_points[l] == k)) {
        parts.push_back(part);
        new_split = true;
        l += 1;
        if (l == split_points.size())
            at_end = true;
    }
    k += 1;

    // layer 7

    if (new_split) {
        part = torch::nn::Sequential();
        new_split = false;
    }

    part->push_back(torch::nn::Linear(torch::nn::LinearOptions(84, num_classes)));
    part->push_back(torch::nn::LogSoftmax(torch::nn::LogSoftmaxOptions(num_classes)));


    return parts;
}

std::vector<torch::nn::Sequential> lenet_part(int64_t num_classes, int start, int end) {
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

    auto parts = lenet_split(num_classes, split_points);
    
    int sum = 0;
    for (int i =0; i < parts.size(); i++) {
        //std::cout << "new layer: "<< i+1 << " "<< parts[i] << std::endl;
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