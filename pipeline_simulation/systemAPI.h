#ifndef _UTILS_H_
#define _UTILS_H_

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
    std::vector<int> inference_path;
    std::map<int, State> clients_state;
    std::vector<int> clients;

    systemAPI() {};
    //void refactoring();
    //exec(task)

    //template <typename T>
    void init_state(model_name name, int model_, int num_class, int start, int end);

};
#endif