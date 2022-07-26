#ifndef _TASK_H_
#define _TASK_H_

#include <iostream> 
#include <torch/torch.h>
#include <torch/data/datasets/base.h>
#include <torch/data/example.h>
#include <torch/types.h>

enum operation {
    forward_,
    backward_,
    optimize_,
    refactoring_,
    noOp
};

struct refactoring_stuct {
    //model_type
    int start = -1;
    int end = -1;
    int data_owners = -1;
    int prev = -1;
    int next = -1;
};

class Task {
 public:
    int client_id;
    int prev_node;
    int size_;
    operation type;
    struct refactoring_stuct refactor_actions;
    torch::Tensor values;

    Task(int client_id, operation type, int prev_node) : 
        client_id(client_id),
        type(type),
        prev_node(prev_node) {}
    
    Task() {}

};

#endif