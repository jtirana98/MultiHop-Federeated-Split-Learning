#include "systemAPI.h"
#include <type_traits>
#include <stdlib.h>

int main(int argc, char **argv) {
    if (argc <= 1) {
        std::cerr << "Wrong number of arguments" << std::endl;
        return 1;
    }
    int myID= atoi(argv[1]), next_node;
    systemAPI sys_(false, myID);
    Task next_task;
    torch::Tensor tmp;

    // POINT 10 -  wait for init refactoring
    auto refactor_message = sys_.my_network_layer.check_new_refactor_task();
    sys_.refactor(refactor_message);
    // POINT 11

    // ----------- DRAFT ------------
    //sys_.init_state(vgg, vgg_model::v11, 10, 1, 3);
    //sys_.init_state_vector(resnet, resnet_model::resnet18, 10, 9, 13);
    //auto layers = sys_.clients_state.find(0)->second.layers;
    //std::cout << sys_.clients_state.find(1)->second.client_id << std::endl;
    //for (int i = 0; i< layers.size(); i++) {
    //    std::cout << "new layer: "<< i+1 << " "<< layers[i] << std::endl;
    //}
    // -------- DRAFT ------------
    
    while (true) {
        // POINT 13
        next_task = sys_.my_network_layer.check_new_task();
        
        // check if it is a refactoring task
        // else ...
        //std::cout << "new task" << std::endl;

        // POINT 14
        auto task = sys_.exec(next_task, tmp);
        // POINT 15
        
        if (task.type != noOp) {
            bool keep_connection = true;
            next_node = (next_task.type == forward_) ? sys_.inference_path[0] : sys_.inference_path[1];
            if (next_node == -1) {
                keep_connection = false;
                next_node = task.client_id;
            }
            sys_.my_network_layer.new_message(task, next_node, keep_connection);
        }
    }
}