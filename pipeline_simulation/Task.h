#ifndef _TASK_H_
#define _TASK_H_

#include <iostream> 

enum operation {
    forward_,
    backward_,
    optimize_,
    refactoring_
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
    operation type;
    struct refactoring_stuct refactor_actions;
    // activations

    Task(int client_id, operation type) : 
        client_id(client_id),
        type(type) {}
    
    Task() {}

};

#endif