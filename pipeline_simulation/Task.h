#ifndef _TASK_H_
#define _TASK_H_

#include <iostream> 
#include <torch/torch.h>
#include <torch/data/datasets/base.h>
#include <torch/data/example.h>
#include <torch/types.h>

enum operation {
    forward_=1,
    backward_=2,
    optimize_=3,
    refactoring_=4,
    aggregation_ = 5,
    noOp=6
};

struct refactoring_data {
    bool to_data_onwer=true;
    int start=0, end=0; // layers
    int prev=0, next=0; // inference path
    std::vector<int> data_owners;
    int dataset=0;
    int num_class=10;
    int model_name_=0, model_type_=0;
    std::vector<std::pair<int, std::string>> rooting_table;
};

class Task {
 public:
    int client_id;
    int prev_node;
    int model_part;
    int size_;
    bool check_=false;
    operation type;
    torch::Tensor values;
    torch::nn::Sequential model_part_;
    std::string model_parts;

    Task(int client_id, operation type, int prev_node) : 
        client_id(client_id),
        type(type),
        prev_node(prev_node) {}
    
    Task() {}

};

#endif