#pragma once

#include <iostream> 
#include <iterator> 
#include <map>
#include <string>
#include <thread>
#include <queue>
#include <chrono>
#include <mutex>
#include <condition_variable>
#include <atomic>
#include <vector>

#include <torch/torch.h>
#include <torch/data/datasets/base.h>
#include <torch/data/example.h>
#include <torch/types.h>

#include "models.h"
#include "State.h"
#include "network_layer.h"

class systemAPI  {
 public:
    network_layer my_network_layer;
    std::thread rcv, snd;

    std::vector<int> inference_path;
    int myid;
    bool is_data_owner;
    // apply for compute node
    std::map<int, State> clients_state;
    std::vector<int> clients;

    // apply for data owner
    int batch_size=32, rounds=50;
    std::vector<State> parts;
    double learning_rate = 0.1;
    
    systemAPI(bool is_data_owner, int myid) : 
    is_data_owner(is_data_owner),
    myid(myid)/*,
    rcv(&network_layer::receiver, &my_network_layer)
    snd(&network_layer::receiver, &my_network_layer)
    */
    { };

    //void refactoring();
    Task exec(Task task, torch::Tensor& targer);
    /*
    void terminate() {
        rcv.join();
        snd.join();
    }
    */
//private:    
    void init_state_vector(model_name name, int model_, int num_class, int start, int end);
    void init_model_sate(model_name name, int model_, int num_class, int start, int end);

};
