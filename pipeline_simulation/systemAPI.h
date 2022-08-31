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
    std::thread rcv_thread, snd_thread;

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
    double running_loss = 0.0;
    double num_correct = 0;
    int batch_index = 0;
    
    systemAPI(bool is_data_owner, int myid, std::string log_dir) : 
    is_data_owner(is_data_owner),
    myid(myid),
    my_network_layer(myid, log_dir),
    rcv_thread(&network_layer::receiver, &my_network_layer),
    snd_thread(&network_layer::sender, &my_network_layer) { };

    void refactor(refactoring_data refactor_message);
    Task exec(Task task, torch::Tensor& targer);

    void zero_metrics() {
        running_loss = 0.0;
        num_correct = 0;
        batch_index = 0;
    }
    
    
    void terminate() {
        rcv_thread.join();
        snd_thread.join();
        my_network_layer.terminate();
    }
private:    
    void init_state_vector(model_name name, int model_, int num_class, int start, int end);
    void init_model_sate(model_name name, int model_, int num_class, int start, int end);

};
